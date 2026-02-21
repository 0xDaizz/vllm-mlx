#!/opt/homebrew/bin/python3.14
"""
Minimal diagnostic script for distributed TP generation.

Tests whether sharded_load + BatchGenerator produces correct output
across 2 ranks (tensor-parallel) WITHOUT the full vllm-mlx server pipeline.

Usage (on each node, with appropriate env vars):

  Rank 0 (hwstudio1):
    export MLX_RANK=0
    export MLX_DISTRIBUTED_BACKEND=jaccl
    export MLX_JACCL_COORDINATOR=192.168.0.105:32323
    export VLLM_MLX_MODEL=$HOME/models/Kimi-K2.5-4bit
    /opt/homebrew/bin/python3.14 test_direct_generate.py

  Rank 1 (hwstudio2):
    export MLX_RANK=1
    export MLX_DISTRIBUTED_BACKEND=jaccl
    export MLX_JACCL_COORDINATOR=192.168.0.105:32323
    export VLLM_MLX_MODEL=$HOME/models/Kimi-K2.5-4bit
    /opt/homebrew/bin/python3.14 test_direct_generate.py

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
log = logging.getLogger("test_direct_generate")

# ---------------------------------------------------------------------------
# 1. Initialize MLX distributed group (same as distributed_launcher.py ~L730)
# ---------------------------------------------------------------------------
import mlx.core as mx

log.info("Initializing distributed group (backend=%s) ...", backend)
group = mx.distributed.init(backend=backend, strict=True)
assert group.rank() == rank, f"Expected rank {rank}, got {group.rank()}"
world_size = group.size()
log.info("Distributed init OK: rank=%d, world_size=%d", rank, world_size)

# ---------------------------------------------------------------------------
# 2. Load sharded model (collective operation — all ranks must call this)
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
# 3. Create BatchGenerator
# ---------------------------------------------------------------------------
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import BatchGenerator

sampler = make_sampler(temp=0.0)  # greedy for reproducibility

batch_gen = BatchGenerator(
    model=model,
    max_tokens=50,
    stop_tokens=tokenizer.eos_token_ids,
    sampler=sampler,
)

log.info("BatchGenerator created")

# ---------------------------------------------------------------------------
# 4. Install _synced_step monkey-patch (same as distributed_launcher.py ~L538)
#    Without this, each rank samples independently -> KV cache corruption.
# ---------------------------------------------------------------------------
if world_size > 1:
    _orig_step = batch_gen._step
    _step_count = [0]

    def _synced_step(input_tokens, prompt_cache, samplers, logits_processors, tokens):
        sampled, logprobs = _orig_step(
            input_tokens, prompt_cache, samplers, logits_processors, tokens
        )
        pre_sync = sampled.item() if sampled.size == 1 else sampled.tolist()
        if group.rank() > 0:
            sampled = mx.zeros_like(sampled)
        sampled = mx.distributed.all_sum(sampled, group=group)
        mx.eval(sampled)
        post_sync = sampled.item() if sampled.size == 1 else sampled.tolist()
        _step_count[0] += 1
        if _step_count[0] <= 50:
            log.info(
                "[_synced_step] step=%d pre=%s post=%s",
                _step_count[0], pre_sync, post_sync,
            )
        return sampled, logprobs

    batch_gen._step = _synced_step
    log.info("_synced_step monkey-patch installed")

# ---------------------------------------------------------------------------
# 5. Prepare prompt with chat template
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
# 6. Generate tokens
# ---------------------------------------------------------------------------
log.info("Starting generation (max 50 tokens) ...")
t_gen_start = time.perf_counter()

uids = batch_gen.insert([prompt_tokens], max_tokens=50)
generated_tokens = []

while True:
    responses = batch_gen.next()
    if not responses:
        break
    for r in responses:
        generated_tokens.append(r.token)
        if rank == 0:
            decoded = tokenizer.decode([r.token])
            print(
                f"  token {len(generated_tokens):3d}: "
                f"id={r.token:6d}  text={decoded!r}"
                f"  finish={r.finish_reason}",
                flush=True,
            )
        if r.finish_reason is not None:
            break
    else:
        continue
    break

t_gen_end = time.perf_counter()

# ---------------------------------------------------------------------------
# 7. Print final output (rank 0 only)
# ---------------------------------------------------------------------------
if rank == 0:
    full_text = tokenizer.decode(generated_tokens)
    gen_time = t_gen_end - t_gen_start
    tps = len(generated_tokens) / gen_time if gen_time > 0 else 0
    print("\n" + "=" * 60)
    print(f"Generated {len(generated_tokens)} tokens in {gen_time:.2f}s ({tps:.1f} tok/s)")
    print(f"Output: {full_text}")
    print("=" * 60)

batch_gen.close()
log.info("Done.")
