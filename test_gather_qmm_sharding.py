#!/opt/homebrew/bin/python3.14
"""
Test: Sharded gather_mm / gather_qmm + sum == Full gather_mm / gather_qmm

Verifies that the tensor-parallel sharding pattern used for QuantizedSwitchLinear
(Kimi K2.5 / DeepSeek V3 MoE) produces correct results when the two shards are
summed.

Test A: Float SwitchGLU (no quantization)
Test B: Quantized SwitchGLU (Q4 + affine)
"""

import sys

sys.path.insert(0, "/Users/hw/mlx-lm-server")

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear, SwiGLU

# ---------------------------------------------------------------------------
# Model dimensions from Kimi K2.5
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 7168
MOE_INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS = 16  # reduced from 384 for speed
NUM_EXPERTS_PER_TOK = 8
GROUP_SIZE = 32
BITS = 4
MODE = "affine"

BATCH = 1
SEQ_LEN = 4

# Thresholds
FLOAT_MAX_DIFF_THRESH = 0.01
QUANT_MAX_DIFF_THRESH = 0.1


def make_routing_indices(batch, seq_len, num_experts, top_k, *, key=None):
    """Generate random top-k routing indices."""
    if key is None:
        key = mx.random.key(42)
    logits = mx.random.normal((batch * seq_len, num_experts), key=key)
    # top-k per token
    indices = mx.argpartition(logits, kth=num_experts - top_k, axis=-1)[
        :, -top_k:
    ]
    indices = mx.sort(indices, axis=-1)
    indices = indices.reshape(batch, seq_len, top_k)
    return indices


def report(name, y_full, y_combined, threshold):
    """Print comparison stats and PASS/FAIL."""
    diff = y_full - y_combined
    max_diff = mx.max(mx.abs(diff)).item()
    mean_diff = mx.mean(mx.abs(diff)).item()
    l2_full = mx.sqrt(mx.sum(mx.square(y_full))).item()
    l2_combined = mx.sqrt(mx.sum(mx.square(y_combined))).item()
    l2_diff = mx.sqrt(mx.sum(mx.square(diff))).item()

    # Mean relative error (avoid div by zero)
    abs_full = mx.abs(y_full)
    rel_err = mx.where(abs_full > 1e-8, mx.abs(diff) / abs_full, mx.zeros_like(diff))
    mean_rel = mx.mean(rel_err).item()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  y_full shape:     {y_full.shape}")
    print(f"  y_combined shape: {y_combined.shape}")
    print(f"  L2 norm (full):     {l2_full:.6f}")
    print(f"  L2 norm (combined): {l2_combined:.6f}")
    print(f"  L2 norm (diff):     {l2_diff:.6f}")
    print(f"  Max abs diff:       {max_diff:.8f}")
    print(f"  Mean abs diff:      {mean_diff:.8f}")
    print(f"  Mean relative err:  {mean_rel:.8f}")
    print(f"  Threshold:          {threshold}")

    passed = max_diff < threshold
    if passed:
        print(f"  Result: PASS (max_diff {max_diff:.8f} < {threshold})")
    else:
        print(f"  Result: FAIL (max_diff {max_diff:.8f} >= {threshold})")
    print(f"{'='*60}")
    return passed


# ===========================================================================
# Test A: Float SwitchGLU
# ===========================================================================
def test_float_switchglu():
    print("\n\n>>> TEST A: Float SwitchGLU (no quantization)")
    print(f"    dims: {HIDDEN_SIZE} -> {MOE_INTERMEDIATE_SIZE}, "
          f"{N_ROUTED_EXPERTS} experts, top-{NUM_EXPERTS_PER_TOK}")

    mx.random.seed(123)

    # 1. Create SwitchGLU
    glu = SwitchGLU(
        HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, N_ROUTED_EXPERTS,
        activation=SwiGLU(), bias=False,
    )
    mx.eval(glu.parameters())

    # 2. Create input and routing indices
    x = mx.random.normal((BATCH, SEQ_LEN, HIDDEN_SIZE))
    inds = make_routing_indices(BATCH, SEQ_LEN, N_ROUTED_EXPERTS, NUM_EXPERTS_PER_TOK)
    mx.eval(x, inds)

    print(f"    x shape: {x.shape}, inds shape: {inds.shape}")

    # 3. Full forward
    y_full = glu(x, inds)
    mx.eval(y_full)
    print(f"    y_full shape: {y_full.shape}")

    # 4. Manual sharded forward using gather_mm directly
    # Retrieve full weights (E, O, I)
    gate_w = glu.gate_proj.weight  # (E, O, I)
    up_w = glu.up_proj.weight      # (E, O, I)
    down_w = glu.down_proj.weight   # (E, I_hidden, O_hidden=hidden_size) wait...

    # SwitchLinear stores weight as (E, output_dims, input_dims)
    # gate_proj: input=7168, output=2048 -> weight (E, 2048, 7168)
    # up_proj:   input=7168, output=2048 -> weight (E, 2048, 7168)
    # down_proj: input=2048, output=7168 -> weight (E, 7168, 2048)
    print(f"    gate_proj.weight shape: {gate_w.shape}")
    print(f"    up_proj.weight shape:   {up_w.shape}")
    print(f"    down_proj.weight shape: {down_w.shape}")

    # Sharding:
    # all-to-sharded (gate, up): split axis 1 (output_dims) by 2
    # sharded-to-all (down): split axis -1 (input_dims) by 2
    half_out = MOE_INTERMEDIATE_SIZE // 2  # 1024

    # Rank 0 shards
    gate_w_r0 = gate_w[:, :half_out, :]       # (E, 1024, 7168)
    up_w_r0 = up_w[:, :half_out, :]           # (E, 1024, 7168)
    down_w_r0 = down_w[:, :, :half_out]       # (E, 7168, 1024)

    # Rank 1 shards
    gate_w_r1 = gate_w[:, half_out:, :]       # (E, 1024, 7168)
    up_w_r1 = up_w[:, half_out:, :]           # (E, 1024, 7168)
    down_w_r1 = down_w[:, :, half_out:]       # (E, 7168, 1024)

    # Prepare x for gather_mm: expand_dims like SwitchGLU.__call__
    x_exp = mx.expand_dims(x, (-2, -3))  # (B, S, 1, 1, hidden)
    # For small token count (<64), SwitchGLU does NOT sort, so we use
    # sorted_indices=False directly.

    # SwitchLinear.__call__ does: gather_mm(x, weight.swapaxes(-1, -2), rhs_indices=indices)
    # weight (E, O, I).swapaxes(-1, -2) -> (E, I, O)
    # gather_mm(x @ (I,O)) -> output (O,)

    # --- Rank 0 forward ---
    x_gate_r0 = mx.gather_mm(
        x_exp, gate_w_r0.swapaxes(-1, -2), rhs_indices=inds, sorted_indices=False
    )  # -> (..., 1024)
    x_up_r0 = mx.gather_mm(
        x_exp, up_w_r0.swapaxes(-1, -2), rhs_indices=inds, sorted_indices=False
    )  # -> (..., 1024)
    # activation: silu(gate) * up
    x_act_r0 = nn.silu(x_gate_r0) * x_up_r0  # (..., 1024)
    y_r0 = mx.gather_mm(
        x_act_r0, down_w_r0.swapaxes(-1, -2), rhs_indices=inds, sorted_indices=False
    )  # -> (..., 7168)

    # --- Rank 1 forward ---
    x_gate_r1 = mx.gather_mm(
        x_exp, gate_w_r1.swapaxes(-1, -2), rhs_indices=inds, sorted_indices=False
    )
    x_up_r1 = mx.gather_mm(
        x_exp, up_w_r1.swapaxes(-1, -2), rhs_indices=inds, sorted_indices=False
    )
    x_act_r1 = nn.silu(x_gate_r1) * x_up_r1
    y_r1 = mx.gather_mm(
        x_act_r1, down_w_r1.swapaxes(-1, -2), rhs_indices=inds, sorted_indices=False
    )

    # 5. Combine: all-reduce (sum)
    y_combined = (y_r0 + y_r1).squeeze(-2)
    mx.eval(y_combined)

    return report("Test A: Float SwitchGLU", y_full, y_combined, FLOAT_MAX_DIFF_THRESH)


# ===========================================================================
# Test B: Quantized SwitchGLU (Q4 + affine)
# ===========================================================================
def test_quantized_switchglu():
    print("\n\n>>> TEST B: Quantized SwitchGLU (Q4 + affine, group_size=32)")
    print(f"    dims: {HIDDEN_SIZE} -> {MOE_INTERMEDIATE_SIZE}, "
          f"{N_ROUTED_EXPERTS} experts, top-{NUM_EXPERTS_PER_TOK}")
    print(f"    bits={BITS}, group_size={GROUP_SIZE}, mode={MODE}")

    mx.random.seed(456)

    # 1. Create float SwitchGLU, then quantize
    glu = SwitchGLU(
        HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, N_ROUTED_EXPERTS,
        activation=SwiGLU(), bias=False,
    )
    mx.eval(glu.parameters())

    # Convert each SwitchLinear to QuantizedSwitchLinear
    gate_q = glu.gate_proj.to_quantized(group_size=GROUP_SIZE, bits=BITS, mode=MODE)
    up_q = glu.up_proj.to_quantized(group_size=GROUP_SIZE, bits=BITS, mode=MODE)
    down_q = glu.down_proj.to_quantized(group_size=GROUP_SIZE, bits=BITS, mode=MODE)
    mx.eval(gate_q.parameters(), up_q.parameters(), down_q.parameters())

    # 2. Input and routing indices
    x = mx.random.normal((BATCH, SEQ_LEN, HIDDEN_SIZE))
    inds = make_routing_indices(BATCH, SEQ_LEN, N_ROUTED_EXPERTS, NUM_EXPERTS_PER_TOK)
    mx.eval(x, inds)

    print(f"    x shape: {x.shape}, inds shape: {inds.shape}")
    print(f"    gate_q weight: {gate_q.weight.shape}, scales: {gate_q.scales.shape}, "
          f"biases: {gate_q.biases.shape if gate_q.biases is not None else None}")
    print(f"    down_q weight: {down_q.weight.shape}, scales: {down_q.scales.shape}")

    # 3. Full forward (using quantized layers, mimicking SwitchGLU forward)
    x_exp = mx.expand_dims(x, (-2, -3))

    x_up_full = up_q(x_exp, inds)
    x_gate_full = gate_q(x_exp, inds)
    x_act_full = nn.silu(x_gate_full) * x_up_full
    y_full = down_q(x_act_full, inds)
    y_full = y_full.squeeze(-2)
    mx.eval(y_full)
    print(f"    y_full shape: {y_full.shape}")

    # 4. Shard quantized weights
    # QuantizedSwitchLinear stores:
    #   weight: (E, O, packed_I)  — packed along input dim
    #   scales: (E, O, I//gs)
    #   biases: (E, O, I//gs)     — quantization biases (affine mode)
    #
    # all-to-sharded (gate, up): split axis 1 (output_dims)
    #   weight (E, O, packed_I) -> split axis 1
    #   scales (E, O, I//gs)    -> split axis 1
    #   biases (E, O, I//gs)    -> split axis 1
    #
    # sharded-to-all (down): split axis -1 (input_dims)
    #   weight (E, O, packed_I) -> split axis -1
    #   scales (E, O, I//gs)    -> split axis -1
    #   biases (E, O, I//gs)    -> split axis -1

    half_out = gate_q.output_dims // 2  # 1024

    # gate_proj shard (all-to-sharded: split output dim, axis 1)
    gate_w_r0 = gate_q.weight[:, :half_out, :]
    gate_s_r0 = gate_q.scales[:, :half_out, :]
    gate_b_r0 = gate_q.biases[:, :half_out, :] if gate_q.biases is not None else None
    gate_w_r1 = gate_q.weight[:, half_out:, :]
    gate_s_r1 = gate_q.scales[:, half_out:, :]
    gate_b_r1 = gate_q.biases[:, half_out:, :] if gate_q.biases is not None else None

    # up_proj shard (all-to-sharded: split output dim, axis 1)
    up_w_r0 = up_q.weight[:, :half_out, :]
    up_s_r0 = up_q.scales[:, :half_out, :]
    up_b_r0 = up_q.biases[:, :half_out, :] if up_q.biases is not None else None
    up_w_r1 = up_q.weight[:, half_out:, :]
    up_s_r1 = up_q.scales[:, half_out:, :]
    up_b_r1 = up_q.biases[:, half_out:, :] if up_q.biases is not None else None

    # down_proj shard (sharded-to-all: split input dim, axis -1)
    # weight (E, 7168, packed_2048) -> split axis -1
    # The packed dim has 2048/8=256 uint32 values (4-bit, 8 values per uint32)
    # Half of that is 128
    half_down_packed = down_q.weight.shape[-1] // 2
    half_down_scales = down_q.scales.shape[-1] // 2

    down_w_r0 = down_q.weight[:, :, :half_down_packed]
    down_s_r0 = down_q.scales[:, :, :half_down_scales]
    down_b_r0 = down_q.biases[:, :, :half_down_scales] if down_q.biases is not None else None
    down_w_r1 = down_q.weight[:, :, half_down_packed:]
    down_s_r1 = down_q.scales[:, :, half_down_scales:]
    down_b_r1 = down_q.biases[:, :, half_down_scales:] if down_q.biases is not None else None

    print(f"    Shard shapes:")
    print(f"      gate_w_r0: {gate_w_r0.shape}, gate_s_r0: {gate_s_r0.shape}")
    print(f"      down_w_r0: {down_w_r0.shape}, down_s_r0: {down_s_r0.shape}")

    # 5. Sharded forward using gather_qmm
    # --- Rank 0 ---
    x_gate_r0 = mx.gather_qmm(
        x_exp, gate_w_r0, gate_s_r0, gate_b_r0,
        rhs_indices=inds, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode=MODE, sorted_indices=False,
    )
    x_up_r0 = mx.gather_qmm(
        x_exp, up_w_r0, up_s_r0, up_b_r0,
        rhs_indices=inds, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode=MODE, sorted_indices=False,
    )
    x_act_r0 = nn.silu(x_gate_r0) * x_up_r0  # (..., 1024)
    y_r0 = mx.gather_qmm(
        x_act_r0, down_w_r0, down_s_r0, down_b_r0,
        rhs_indices=inds, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode=MODE, sorted_indices=False,
    )

    # --- Rank 1 ---
    x_gate_r1 = mx.gather_qmm(
        x_exp, gate_w_r1, gate_s_r1, gate_b_r1,
        rhs_indices=inds, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode=MODE, sorted_indices=False,
    )
    x_up_r1 = mx.gather_qmm(
        x_exp, up_w_r1, up_s_r1, up_b_r1,
        rhs_indices=inds, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode=MODE, sorted_indices=False,
    )
    x_act_r1 = nn.silu(x_gate_r1) * x_up_r1  # (..., 1024)
    y_r1 = mx.gather_qmm(
        x_act_r1, down_w_r1, down_s_r1, down_b_r1,
        rhs_indices=inds, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode=MODE, sorted_indices=False,
    )

    # 6. Combine shards
    y_combined = (y_r0 + y_r1).squeeze(-2)
    mx.eval(y_combined)

    return report(
        "Test B: Quantized SwitchGLU (Q4/affine)",
        y_full, y_combined, QUANT_MAX_DIFF_THRESH,
    )


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Sharded gather_mm / gather_qmm Correctness Test")
    print("=" * 60)
    print(f"  MLX version: {mx.__version__}")

    results = []
    results.append(("Test A (float)", test_float_switchglu()))
    results.append(("Test B (quantized)", test_quantized_switchglu()))

    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    print("=" * 60)

    if all_pass:
        print("\n  All tests passed.\n")
    else:
        print("\n  Some tests FAILED.\n")
        sys.exit(1)
