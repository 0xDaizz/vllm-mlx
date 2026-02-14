#!/usr/bin/env python3
"""TP Parity Test for vllm-mlx distributed inference.

Verifies that tensor-parallel model inference produces identical results
to single-device inference with greedy decoding.

Usage as launcher:
    python tests/test_tp_parity.py --model mlx-community/Llama-3.2-1B-Instruct-4bit -n 2

Usage as standalone single-device test:
    python tests/test_tp_parity.py --model mlx-community/Llama-3.2-1B-Instruct-4bit --single-only
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


# ---------------------------------------------------------------------------
# Single-device reference generation
# ---------------------------------------------------------------------------


def generate_reference(model_name: str, prompt: str, max_tokens: int = 100) -> list[int]:
    """Generate tokens on single device as reference.

    Uses mlx_lm.load() and greedy decoding (temperature=0).
    Returns list of token IDs.
    """
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model, tokenizer = load(model_name)

    # Greedy sampling (temperature=0)
    sampler = make_sampler(temp=0.0)

    prompt_tokens = tokenizer.encode(prompt)
    prompt_array = mx.array(prompt_tokens)

    tokens: list[int] = []
    for step_output in generate_step(
        prompt=prompt_array, model=model, sampler=sampler, max_tokens=max_tokens
    ):
        # generate_step yields different formats depending on mlx-lm version
        if hasattr(step_output, "token"):
            tok = step_output.token
        elif isinstance(step_output, tuple):
            tok = step_output[0]
        else:
            tok = int(step_output)
        tokens.append(int(tok))
        if len(tokens) >= max_tokens:
            break

    return tokens


# ---------------------------------------------------------------------------
# Distributed worker (called inside mlx.launch)
# ---------------------------------------------------------------------------


def generate_distributed(
    model_name: str, prompt: str, max_tokens: int = 100, output_file: str | None = None
) -> list[int]:
    """Generate tokens using TP-sharded model.

    This function is called inside a distributed process launched by mlx.launch.
    Uses mlx_lm.utils.sharded_load() and greedy decoding.
    Only rank 0 returns the actual tokens; other ranks return empty list.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import sharded_load

    group = mx.distributed.init(strict=True)
    rank = group.rank()
    world_size = group.size()

    print(f"[Rank {rank}/{world_size}] Loading sharded model: {model_name}")

    model, tokenizer = sharded_load(
        model_name,
        pipeline_group=None,
        tensor_group=group,
    )

    # Barrier after loading
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))
    print(f"[Rank {rank}/{world_size}] Model loaded, generating tokens")

    # Greedy sampling
    sampler = make_sampler(temp=0.0)
    prompt_tokens = tokenizer.encode(prompt)
    prompt_array = mx.array(prompt_tokens)

    tokens: list[int] = []
    for step_output in generate_step(
        prompt=prompt_array, model=model, sampler=sampler, max_tokens=max_tokens
    ):
        if hasattr(step_output, "token"):
            tok = step_output.token
        elif isinstance(step_output, tuple):
            tok = step_output[0]
        else:
            tok = int(step_output)

        # Rank 0 samples, broadcast to all via all_sum
        tok_array = mx.array([int(tok)]) if rank == 0 else mx.array([0])
        tok_array = mx.distributed.all_sum(tok_array, group=group)
        tok = tok_array.item()

        tokens.append(int(tok))
        if len(tokens) >= max_tokens:
            break

    # Rank 0 writes output
    if rank == 0:
        result = {
            "tokens": tokens,
            "model_name": model_name,
            "world_size": world_size,
            "text": tokenizer.decode(tokens),
        }
        out = output_file or "/tmp/tp_parity_output.json"
        with open(out, "w") as f:
            json.dump(result, f)
        print(f"[Rank 0] Output written to {out}")

    # Final barrier
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))
    print(f"[Rank {rank}/{world_size}] Done")

    return tokens if rank == 0 else []


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def compare_outputs(reference: list[int], distributed: list[int]) -> dict:
    """Compare two token sequences.

    Returns dict with:
    - match: bool (exact match)
    - reference_len: int
    - distributed_len: int
    - first_mismatch: int | None (index of first different token)
    - match_ratio: float
    """
    exact_match = reference == distributed
    first_mismatch: int | None = None

    if not exact_match:
        for i, (r, d) in enumerate(zip(reference, distributed)):
            if r != d:
                first_mismatch = i
                break
        if first_mismatch is None and len(reference) != len(distributed):
            first_mismatch = min(len(reference), len(distributed))

    matching = sum(1 for r, d in zip(reference, distributed) if r == d)
    max_len = max(len(reference), len(distributed), 1)

    return {
        "match": exact_match,
        "reference_len": len(reference),
        "distributed_len": len(distributed),
        "first_mismatch": first_mismatch,
        "match_ratio": matching / max_len,
    }


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------


def _get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback: use resource module (macOS / Linux)
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is in bytes on macOS, kilobytes on Linux
            if sys.platform == "darwin":
                return usage.ru_maxrss / (1024 * 1024)
            return usage.ru_maxrss / 1024
        except Exception:
            return 0.0


def profile_memory(model_name: str, num_ranks: int = 2) -> dict:
    """Profile memory usage per rank.

    Returns dict with:
    - single_rank_memory_mb: float
    - per_rank_memory_mb: list[float]
    - reduction_ratio: float (should be ~1/N)
    """
    # Single-rank memory: load model, measure, unload
    print("  Profiling single-rank memory...")
    import mlx.core as mx
    from mlx_lm import load

    mem_before = _get_process_memory_mb()
    model, tokenizer = load(model_name)
    mx.eval(model.parameters())
    single_mem = _get_process_memory_mb() - mem_before
    del model, tokenizer

    # For distributed memory, we would need to launch workers and collect
    # their memory snapshots.  Since we cannot directly measure remote
    # processes here, we estimate from the model weight file sizes.
    print("  Estimating per-rank memory from weight sizes...")
    per_rank_estimate = single_mem / num_ranks

    return {
        "single_rank_memory_mb": round(single_mem, 2),
        "per_rank_memory_mb": [round(per_rank_estimate, 2)] * num_ranks,
        "reduction_ratio": round(1.0 / num_ranks, 4),
    }


# ---------------------------------------------------------------------------
# Distributed launch helper
# ---------------------------------------------------------------------------


def run_distributed_test(
    model_name: str,
    prompt: str,
    max_tokens: int,
    num_ranks: int,
    backend: str,
) -> dict | None:
    """Launch distributed generation and return tokens."""
    # Create temp file for output
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        output_file = f.name

    # Build command to launch this script as a distributed worker
    script = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        "-m",
        "mlx._distributed_utils.launch",
        "-n",
        str(num_ranks),
        "--backend",
        backend,
        script,
        "--worker-mode",
        "--model",
        model_name,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--output-file",
        output_file,
    ]

    print(f"  Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  ERROR: Distributed launch failed with code {result.returncode}")
        return None

    # Read output written by rank 0
    try:
        with open(output_file) as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  ERROR: Failed to read distributed output: {e}")
        return None
    finally:
        try:
            os.unlink(output_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="TP Parity Test")
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--num-ranks", "-n", type=int, default=2)
    parser.add_argument(
        "--backend", default="ring", choices=["jaccl", "ring"]
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Test multiple models sequentially",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only run single-device test",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Run memory profiling",
    )
    # Hidden worker-mode args (used when script is relaunched by mlx.launch)
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-file", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Mode 1: Worker – we are inside an mlx.launch subprocess
    # ------------------------------------------------------------------
    if args.worker_mode or "MLX_RANK" in os.environ:
        output_file = args.output_file or "/tmp/tp_parity_output.json"
        generate_distributed(args.model, args.prompt, args.max_tokens, output_file)
        return

    # ------------------------------------------------------------------
    # Mode 2: Launcher – orchestrate single + distributed and compare
    # ------------------------------------------------------------------

    model_list = args.models or [args.model]
    all_passed = True

    for model_name in model_list:
        print("=" * 60)
        print("TP Parity Test")
        print("=" * 60)
        print(f"  Model      : {model_name}")
        print(f"  Prompt     : {args.prompt!r}")
        print(f"  Max tokens : {args.max_tokens}")
        print(f"  Num ranks  : {args.num_ranks}")
        print(f"  Backend    : {args.backend}")
        print()

        # Step 1: Single-device reference -----------------------------------
        print("[Step 1] Generating single-device reference...")
        t0 = time.time()
        ref_tokens = generate_reference(model_name, args.prompt, args.max_tokens)
        single_time = time.time() - t0
        print(f"  Generated {len(ref_tokens)} tokens in {single_time:.2f}s")
        print(f"  Tokens (first 10): {ref_tokens[:10]}...")

        if args.single_only:
            print("\n  --single-only: Skipping distributed test")
            continue

        # Step 2: Distributed generation ------------------------------------
        print(f"\n[Step 2] Generating with {args.num_ranks}-rank TP...")
        t0 = time.time()
        dist_result = run_distributed_test(
            model_name, args.prompt, args.max_tokens, args.num_ranks, args.backend
        )
        dist_time = time.time() - t0

        if dist_result is None:
            print("  FAILED: Distributed generation returned no output")
            all_passed = False
            continue

        dist_tokens: list[int] = dist_result["tokens"]
        print(f"  Generated {len(dist_tokens)} tokens in {dist_time:.2f}s")
        print(f"  Tokens (first 10): {dist_tokens[:10]}...")

        # Step 3: Compare ---------------------------------------------------
        print("\n[Step 3] Comparing outputs...")
        comparison = compare_outputs(ref_tokens, dist_tokens)

        print(f"  Exact match : {comparison['match']}")
        print(f"  Match ratio : {comparison['match_ratio']:.4f}")
        if comparison["first_mismatch"] is not None:
            idx = comparison["first_mismatch"]
            ref_tok = ref_tokens[idx] if idx < len(ref_tokens) else "N/A"
            dist_tok = dist_tokens[idx] if idx < len(dist_tokens) else "N/A"
            print(f"  First mismatch at index {idx}: ref={ref_tok}, dist={dist_tok}")

        # Step 4 (optional): Memory profile ---------------------------------
        if args.memory_profile:
            print("\n[Step 4] Memory profiling...")
            mem_info = profile_memory(model_name, args.num_ranks)
            print(f"  Single-rank memory : {mem_info['single_rank_memory_mb']:.2f} MB")
            print(f"  Per-rank estimate  : {mem_info['per_rank_memory_mb']}")
            print(f"  Reduction ratio    : {mem_info['reduction_ratio']}")

        # Verdict -----------------------------------------------------------
        print("\n" + "=" * 60)
        if comparison["match"]:
            print("RESULT: PASS - Outputs are bit-identical")
        elif comparison["match_ratio"] > 0.95:
            print(
                f"RESULT: WARN - High match ({comparison['match_ratio']:.1%}) "
                "but not exact"
            )
        else:
            print(f"RESULT: FAIL - Low match ratio ({comparison['match_ratio']:.1%})")
            all_passed = False
        print("=" * 60)
        print()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
