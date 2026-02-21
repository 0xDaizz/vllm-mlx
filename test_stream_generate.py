#!/opt/homebrew/bin/python3.14
"""
Minimal diagnostic script for distributed TP generation using stream_generate().

Tests whether sharded_load + stream_generate (the DIRECT mlx-lm generate API)
produces correct output across 2 ranks, bypassing BatchGenerator entirely.
This isolates whether corruption bugs are in BatchGenerator or in the model.

Two approaches are tested:
  A) stream_generate() — high-level, handles tokenization/detokenization
  B) generate_step() — low-level, raw token generator

Both use a custom sampler that broadcasts rank 0's choice to all ranks,
preventing independent sampling → KV cache divergence.

Usage (on each node, with appropriate env vars):

  Rank 0 (hwstudio1):
    export MLX_RANK=0
    export MLX_DISTRIBUTED_BACKEND=jaccl
    export MLX_JACCL_COORDINATOR=192.168.0.105:32323
    export VLLM_MLX_MODEL=$HOME/models/Kimi-K2.5-4bit
    /opt/homebrew/bin/python3.14 test_stream_generate.py

  Rank 1 (hwstudio2):
    export MLX_RANK=1
    export MLX_DISTRIBUTED_BACKEND=jaccl
    export MLX_JACCL_COORDINATOR=192.168.0.105:32323
    export VLLM_MLX_MODEL=$HOME/models/Kimi-K2.5-4bit
    /opt/homebrew/bin/python3.14 test_stream_generate.py

Both ranks MUST be started (rank 0 first, then rank 1 within ~30s).
Only rank 0 prints the generated output.
"""

from __future__ import annotations

import os
import sys
import time
import logging

# ---------------------------------------------------------------------------
# 0. Read configuration from environment
# ---------------------------------------------------------------------------
rank = int(os.environ.get("MLX_RANK", "0"))
backend = os.environ.get("MLX_DISTRIBUTED_BACKEND", "jaccl")
model_path = os.environ.get("VLLM_MLX_MODEL", "")

if not model_path:
    print("ERROR: Set VLLM_MLX_MODEL to the model path.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [Rank {rank}] %(levelname)s: %(message)s",
)
log = logging.getLogger("test_stream_generate")

# ---------------------------------------------------------------------------
# 1. Initialize MLX distributed group
# ---------------------------------------------------------------------------
import mlx.core as mx

log.info("Initializing distributed group (backend=%s) ...", backend)
group = mx.distributed.init(backend=backend, strict=True)
assert group.rank() == rank, f"Expected rank {rank}, got {group.rank()}"
world_size = group.size()
log.info("Distributed init OK: rank=%d, world_size=%d", rank, world_size)

# ---------------------------------------------------------------------------
# 2. Load sharded model (collective operation -- all ranks must call this)
# ---------------------------------------------------------------------------
from mlx_lm.utils import sharded_load

log.info("Loading sharded model: %s", model_path)
t0 = time.perf_counter()

model, tokenizer = sharded_load(
    model_path,
    pipeline_group=None,
    tensor_group=group,
)

t1 = time.perf_counter()
log.info("Model loaded in %.1fs", t1 - t0)

# Barrier after load
mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))
log.info("Post-load barrier passed")

# ---------------------------------------------------------------------------
# 3. Build a synchronized sampler for distributed generation
#    Without this, each rank samples independently -> KV cache corruption.
# ---------------------------------------------------------------------------
_step_count = [0]


def synced_greedy_sampler(logprobs: mx.array) -> mx.array:
    """Greedy sampler that broadcasts rank 0's token to all ranks."""
    sampled = mx.argmax(logprobs, axis=-1)

    if world_size > 1:
        pre_sync = sampled.item() if sampled.size == 1 else sampled.tolist()
        if group.rank() > 0:
            sampled = mx.zeros_like(sampled)
        sampled = mx.distributed.all_sum(sampled, group=group)
        mx.eval(sampled)
        post_sync = sampled.item() if sampled.size == 1 else sampled.tolist()

        _step_count[0] += 1
        if _step_count[0] <= 55:
            log.info(
                "[synced_sampler] step=%d pre=%s post=%s",
                _step_count[0], pre_sync, post_sync,
            )

    return sampled


# ---------------------------------------------------------------------------
# 4. Prepare prompt with chat template
# ---------------------------------------------------------------------------
messages = [
    {"role": "user", "content": "What is 2+2? Answer briefly."},
]

prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
prompt_tokens = tokenizer.encode(prompt_text)

if rank == 0:
    log.info("Prompt text:\n%s", prompt_text)
    log.info("Prompt tokens (%d): %s", len(prompt_tokens), prompt_tokens[:20])

# ---------------------------------------------------------------------------
# 5A. Test with stream_generate() — high-level API
# ---------------------------------------------------------------------------
from mlx_lm.generate import stream_generate, generate_step

log.info("=" * 60)
log.info("TEST A: stream_generate() (high-level API)")
log.info("=" * 60)

_step_count[0] = 0
t_gen_start = time.perf_counter()
generated_tokens_a = []

for response in stream_generate(
    model,
    tokenizer,
    prompt_tokens,  # pass token IDs directly
    max_tokens=50,
    sampler=synced_greedy_sampler,
):
    generated_tokens_a.append(response.token)
    if rank == 0:
        print(
            f"  [A] token {len(generated_tokens_a):3d}: "
            f"id={response.token:6d}  text={response.text!r}"
            f"  finish={response.finish_reason}",
            flush=True,
        )

t_gen_end = time.perf_counter()

if rank == 0:
    gen_time = t_gen_end - t_gen_start
    tps = len(generated_tokens_a) / gen_time if gen_time > 0 else 0
    print()
    print("=" * 60)
    print(f"[A] stream_generate: {len(generated_tokens_a)} tokens in {gen_time:.2f}s ({tps:.1f} tok/s)")
    # Decode all tokens for final output
    full_text_a = tokenizer.decode(generated_tokens_a)
    print(f"[A] Output: {full_text_a}")
    print("=" * 60)

# Barrier between tests
mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))
log.info("Barrier between test A and test B passed")

# ---------------------------------------------------------------------------
# 5B. Test with generate_step() — low-level API
# ---------------------------------------------------------------------------
log.info("=" * 60)
log.info("TEST B: generate_step() (low-level API)")
log.info("=" * 60)

_step_count[0] = 0
t_gen_start = time.perf_counter()
generated_tokens_b = []

prompt_array = mx.array(prompt_tokens)

for token_id, logprobs in generate_step(
    prompt_array,
    model,
    max_tokens=50,
    sampler=synced_greedy_sampler,
):
    generated_tokens_b.append(token_id)
    if rank == 0:
        decoded = tokenizer.decode([token_id])
        print(
            f"  [B] token {len(generated_tokens_b):3d}: "
            f"id={token_id:6d}  text={decoded!r}",
            flush=True,
        )
    # Check for EOS
    if token_id in tokenizer.eos_token_ids:
        break

t_gen_end = time.perf_counter()

if rank == 0:
    gen_time = t_gen_end - t_gen_start
    tps = len(generated_tokens_b) / gen_time if gen_time > 0 else 0
    full_text_b = tokenizer.decode(generated_tokens_b)
    print()
    print("=" * 60)
    print(f"[B] generate_step: {len(generated_tokens_b)} tokens in {gen_time:.2f}s ({tps:.1f} tok/s)")
    print(f"[B] Output: {full_text_b}")
    print("=" * 60)

# ---------------------------------------------------------------------------
# 6. Compare results
# ---------------------------------------------------------------------------
if rank == 0:
    print()
    print("=" * 60)
    print("COMPARISON:")
    print(f"  [A] stream_generate tokens: {generated_tokens_a}")
    print(f"  [B] generate_step   tokens: {generated_tokens_b}")
    match = generated_tokens_a == generated_tokens_b
    print(f"  Match: {match}")
    if not match:
        print("  WARNING: Outputs differ! This may indicate a bug.")
    print("=" * 60)

log.info("Done.")
