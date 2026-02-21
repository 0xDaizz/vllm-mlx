#!/opt/homebrew/bin/python3.14
"""
Test script: compare single-node inference vs TP=2 sharded inference.

This verifies that tensor-parallel sharding produces the same output as
single-node for the deepseek_v3 architecture (Moonlight-16B-A3B).

Usage:
    Single-node:
        python test_single_vs_tp.py --mode single

    TP=2 (run on each rank):
        MLX_RANK=0 MLX_DISTRIBUTED_BACKEND=jaccl python test_single_vs_tp.py --mode tp
        MLX_RANK=1 MLX_DISTRIBUTED_BACKEND=jaccl python test_single_vs_tp.py --mode tp
"""

import sys

import argparse
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model, load_tokenizer, sharded_load
from mlx_lm.generate import generate_step

MODEL_PATH = "/Users/hw/models/Moonlight-16B-A3B-Instruct-4-bit"
MAX_TOKENS = 100
PROMPT_MESSAGES = [{"role": "user", "content": "Write a haiku about machine learning."}]


def build_prompt(tokenizer):
    """Build the prompt tokens using chat template if available."""
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            PROMPT_MESSAGES, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = PROMPT_MESSAGES[0]["content"]
    prompt_tokens = tokenizer.encode(prompt_text)
    return prompt_text, prompt_tokens


def run_single():
    """Run single-node inference (no sharding)."""
    print("=" * 60)
    print("MODE: single (no sharding)")
    print(f"Model: {MODEL_PATH}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = load_tokenizer(MODEL_PATH, tokenizer_config_extra={"trust_remote_code": True})

    # Load model (full, unsharded)
    print("Loading model (single, unsharded)...")
    t0 = time.perf_counter()
    model, config = load_model(Path(MODEL_PATH))
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Build prompt
    prompt_text, prompt_tokens = build_prompt(tokenizer)
    print(f"\nPrompt ({len(prompt_tokens)} tokens):")
    print(prompt_text)
    print("-" * 60)

    # Greedy sampler
    sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)

    # Get EOS token IDs
    eos_token_ids = set()
    if hasattr(tokenizer, "_eos_token_ids"):
        eos_token_ids = set(tokenizer._eos_token_ids)
    elif tokenizer.eos_token_id is not None:
        eos_token_ids = {tokenizer.eos_token_id}
    print(f"EOS token IDs: {eos_token_ids}")

    # Generate
    print(f"\nGenerating (max {MAX_TOKENS} tokens, greedy)...")
    prompt_array = mx.array(prompt_tokens)
    generated_tokens = []
    t_start = time.perf_counter()

    for token_id, logprobs in generate_step(
        prompt=prompt_array,
        model=model,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
    ):
        generated_tokens.append(token_id)
        decoded = tokenizer.decode([token_id])
        print(f"  token {len(generated_tokens):3d}: id={token_id:6d}  |{decoded}|")

        if token_id in eos_token_ids:
            print("  [EOS reached]")
            break

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # Final output
    final_text = tokenizer.decode(generated_tokens)
    tok_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print(f"Generated {len(generated_tokens)} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
    print(f"\nToken IDs: {generated_tokens}")
    print(f"\nDecoded text:\n{final_text}")
    print("=" * 60)


def run_tp():
    """Run TP=2 sharded inference."""
    # Initialize distributed group
    backend = os.environ.get("MLX_DISTRIBUTED_BACKEND", "jaccl")
    print(f"Initializing distributed group (backend={backend})...")
    group = mx.distributed.init(backend=backend, strict=True)
    rank = group.rank()
    world_size = group.size()

    if rank == 0:
        print("=" * 60)
        print(f"MODE: tp (tensor parallel, world_size={world_size})")
        print(f"Model: {MODEL_PATH}")
        print("=" * 60)

    # Load model with sharding
    if rank == 0:
        print("Loading model (sharded)...")
    t0 = time.perf_counter()
    model, tokenizer = sharded_load(
        MODEL_PATH,
        tensor_group=group,
    )
    load_time = time.perf_counter() - t0
    if rank == 0:
        print(f"Model loaded in {load_time:.1f}s")

    # Build prompt
    prompt_text, prompt_tokens = build_prompt(tokenizer)
    if rank == 0:
        print(f"\nPrompt ({len(prompt_tokens)} tokens):")
        print(prompt_text)
        print("-" * 60)

    # Synced greedy sampler: rank 0 picks argmax, broadcast via all_sum
    _step_count = [0]

    def synced_greedy_sampler(logprobs):
        _step_count[0] += 1
        step = _step_count[0]

        # Debug: logits stats on both ranks
        if step <= 20:
            lp_min = logprobs.min().item()
            lp_max = logprobs.max().item()
            has_nan = mx.any(mx.isnan(logprobs)).item()
            has_inf = mx.any(mx.isinf(logprobs)).item()
            print(f"  [Rank {rank}] step={step} logprobs shape={logprobs.shape} "
                  f"min={lp_min:.4f} max={lp_max:.4f} nan={has_nan} inf={has_inf}")

        sampled = mx.argmax(logprobs, axis=-1)

        if step <= 20:
            pre_val = sampled.item()
            print(f"  [Rank {rank}] step={step} pre-sync argmax={pre_val}")
            if pre_val >= 163840:
                print(f"  [Rank {rank}] *** ALERT: argmax {pre_val} >= vocab_size! ***")
                print(f"  [Rank {rank}]   logprobs dtype={logprobs.dtype} shape={logprobs.shape}")
                print(f"  [Rank {rank}]   logprobs[:5]={logprobs[0,:5].tolist()}")
                print(f"  [Rank {rank}]   logprobs[-5:]={logprobs[0,-5:].tolist()}")

        if world_size > 1:
            if group.rank() > 0:
                sampled = mx.zeros_like(sampled)
            sampled = mx.distributed.all_sum(sampled, group=group)
            mx.eval(sampled)

        if step <= 20:
            post_val = sampled.item()
            print(f"  [Rank {rank}] step={step} post-sync token={post_val}")

        return sampled

    # Get EOS token IDs
    eos_token_ids = set()
    if hasattr(tokenizer, "_eos_token_ids"):
        eos_token_ids = set(tokenizer._eos_token_ids)
    elif tokenizer.eos_token_id is not None:
        eos_token_ids = {tokenizer.eos_token_id}
    if rank == 0:
        print(f"EOS token IDs: {eos_token_ids}")

    # Generate
    if rank == 0:
        print(f"\nGenerating (max {MAX_TOKENS} tokens, synced greedy)...")
    prompt_array = mx.array(prompt_tokens)
    generated_tokens = []
    t_start = time.perf_counter()

    for token_id, logprobs in generate_step(
        prompt=prompt_array,
        model=model,
        max_tokens=MAX_TOKENS,
        sampler=synced_greedy_sampler,
    ):
        generated_tokens.append(token_id)
        # Force full evaluation every step to prevent lazy graph from holding RDMA buffers
        mx.eval(logprobs)
        if rank == 0:
            try:
                decoded = tokenizer.decode([token_id])
            except (KeyError, Exception) as e:
                decoded = f"<INVALID:{e}>"
            print(f"  token {len(generated_tokens):3d}: id={token_id:6d}  |{decoded}|")

        if len(generated_tokens) >= MAX_TOKENS:
            break
        if token_id in eos_token_ids:
            if rank == 0:
                print("  [EOS reached]")
            break

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # Final output (rank 0 only)
    if rank == 0:
        final_text = tokenizer.decode(generated_tokens)
        tok_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print(f"Generated {len(generated_tokens)} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
        print(f"\nToken IDs: {generated_tokens}")
        print(f"\nDecoded text:\n{final_text}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare single-node vs TP sharded inference for Moonlight-16B"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "tp"],
        required=True,
        help="'single' for unsharded inference, 'tp' for tensor-parallel sharded inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model path (default: MODEL_PATH constant)",
    )
    args = parser.parse_args()

    global MODEL_PATH
    if args.model:
        MODEL_PATH = args.model

    if args.mode == "single":
        run_single()
    elif args.mode == "tp":
        run_tp()


if __name__ == "__main__":
    main()
