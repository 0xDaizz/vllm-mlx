#!/usr/bin/env python3
"""GLM-5-4bit TP=2 Benchmark Script
Measures: TTFT, prefill tok/s, decode tok/s, batch throughput.
"""

import urllib.request
import json
import time
import concurrent.futures
import sys

BASE_URL = "http://100.120.177.62:8000/v1/chat/completions"
MODEL = "/Users/hw/models/GLM-5-4bit"


def chat_request(prompt, max_tokens, temperature=0):
    """Send a chat completion request and return timing + usage info."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL, data=data, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - start

    usage = result["usage"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "elapsed": elapsed,
    }


def run_single_request_benchmarks():
    """Part 1: Single request benchmarks with varying output lengths.

    Strategy for TTFT/prefill measurement:
    - First send max_tokens=1 to measure approximate prefill time (TTFT).
    - Then send full request to measure total time.
    - decode time = total - TTFT estimate
    - prefill tok/s = prompt_tokens / TTFT
    - decode tok/s = completion_tokens / decode_time
    """
    tests = [
        ("Short", 300,
         "Explain what a hash table is and how it works."),
        ("Medium", 1000,
         "Write a detailed tutorial on building a REST API with Python Flask. "
         "Include code examples for CRUD operations, error handling, and authentication."),
        ("Long", 3000,
         "Write a comprehensive guide to modern CPU architecture. Cover pipelining, "
         "branch prediction, cache hierarchy, out-of-order execution, SIMD, and "
         "multi-core design. Include historical context and future trends."),
        ("Very Long", 10000,
         "Write an extremely detailed textbook chapter on the history and evolution of "
         "programming languages from the 1950s to 2025. Cover FORTRAN, LISP, C, C++, "
         "Java, Python, Rust, and all major paradigm shifts. Include code examples in "
         "each language and discuss their design philosophies, strengths, and weaknesses."),
    ]

    results = []
    for name, max_tok, prompt in tests:
        print(f"\n[Single] {name} (max_tokens={max_tok})", flush=True)
        try:
            # Step 1: Measure TTFT (approximate) with max_tokens=1
            print(f"  Measuring TTFT (max_tokens=1)...", flush=True)
            r1 = chat_request(prompt, 1)
            ttft_s = r1["elapsed"]
            prompt_tokens = r1["prompt_tokens"]
            prefill_tok_s = prompt_tokens / ttft_s if ttft_s > 0 else 0
            print(f"  TTFT ~{ttft_s*1000:.0f} ms, prompt_tokens={prompt_tokens}, "
                  f"prefill={prefill_tok_s:.1f} tok/s", flush=True)

            # Step 2: Full generation
            print(f"  Running full generation (max_tokens={max_tok})...", flush=True)
            r2 = chat_request(prompt, max_tok)
            total_s = r2["elapsed"]
            completion_tokens = r2["completion_tokens"]

            # Decode time = total - prefill (TTFT)
            decode_time = total_s - ttft_s
            decode_tok_s = completion_tokens / decode_time if decode_time > 0 else 0

            print(f"  Done: {completion_tokens} tokens in {total_s:.2f}s "
                  f"(decode={decode_tok_s:.1f} tok/s)", flush=True)

            results.append({
                "name": name,
                "max_tokens": max_tok,
                "actual_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "ttft_ms": ttft_s * 1000,
                "prefill_tok_s": prefill_tok_s,
                "decode_tok_s": decode_tok_s,
                "total_s": total_s,
            })
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append({
                "name": name,
                "max_tokens": max_tok,
                "actual_tokens": 0,
                "prompt_tokens": 0,
                "ttft_ms": 0,
                "prefill_tok_s": 0,
                "decode_tok_s": 0,
                "total_s": 0,
                "error": str(e),
            })

    return results


def run_batch_benchmarks():
    """Part 2: Batch throughput benchmarks with concurrent requests."""
    prompt = "Write a short essay about artificial intelligence in healthcare."
    max_tokens = 500
    batch_sizes = [1, 2, 4, 8]

    # First measure TTFT for this prompt to better estimate per-request decode speed
    print(f"\n  Measuring TTFT for batch prompt (max_tokens=1)...", flush=True)
    r_ttft = chat_request(prompt, 1)
    batch_ttft_s = r_ttft["elapsed"]
    print(f"  Batch prompt TTFT ~{batch_ttft_s*1000:.0f} ms", flush=True)

    results = []
    for batch_size in batch_sizes:
        print(f"\n[Batch] {batch_size} concurrent requests (max_tokens={max_tokens})...",
              flush=True)
        try:
            individual_results = []
            wall_start = time.perf_counter()

            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(chat_request, prompt, max_tokens)
                    for _ in range(batch_size)
                ]
                for i, f in enumerate(concurrent.futures.as_completed(futures)):
                    r = f.result()
                    individual_results.append(r)
                    tok_s = r["completion_tokens"] / r["elapsed"] if r["elapsed"] > 0 else 0
                    print(f"  Req {i+1}/{batch_size}: {r['completion_tokens']} tok, "
                          f"{r['elapsed']:.2f}s, {tok_s:.1f} tok/s", flush=True)

            wall_time = time.perf_counter() - wall_start

            total_tokens = sum(r["completion_tokens"] for r in individual_results)
            throughput = total_tokens / wall_time if wall_time > 0 else 0

            # Average per-request decode tok/s (subtracting TTFT estimate)
            decode_speeds = []
            for r in individual_results:
                dt = r["elapsed"] - batch_ttft_s
                if dt > 0:
                    decode_speeds.append(r["completion_tokens"] / dt)
            avg_decode = sum(decode_speeds) / len(decode_speeds) if decode_speeds else 0

            print(f"  Total: {total_tokens} tok in {wall_time:.2f}s, "
                  f"throughput={throughput:.1f} tok/s, avg_decode={avg_decode:.1f} tok/s",
                  flush=True)

            results.append({
                "batch_size": batch_size,
                "total_tokens": total_tokens,
                "wall_time": wall_time,
                "throughput": throughput,
                "avg_decode": avg_decode,
                "individual": individual_results,
            })
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append({
                "batch_size": batch_size,
                "total_tokens": 0,
                "wall_time": 0,
                "throughput": 0,
                "avg_decode": 0,
                "error": str(e),
            })

    return results


def print_results(single_results, batch_results):
    """Print results as markdown tables."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: GLM-5-4bit, TP=2, Barebone (no MTP/KV cache)")
    print("=" * 80)

    print("\n### Single Request Results\n")
    print("| Test | max_tokens | actual_tokens | TTFT (ms) | prefill (tok/s) | decode (tok/s) | total (s) |")
    print("|------|-----------|--------------|-----------|----------------|---------------|-----------|")
    for r in single_results:
        if "error" in r:
            print(f"| {r['name']} | {r['max_tokens']} | ERROR | - | - | - | {r.get('error','')} |")
        else:
            print(f"| {r['name']} | {r['max_tokens']} | {r['actual_tokens']} | "
                  f"{r['ttft_ms']:.0f} | {r['prefill_tok_s']:.1f} | "
                  f"{r['decode_tok_s']:.1f} | {r['total_s']:.2f} |")

    print("\n### Batch Throughput Results\n")
    print("| Batch Size | Total Tokens | Wall Time (s) | Throughput (tok/s) | Avg Decode (tok/s) |")
    print("|-----------|-------------|--------------|-------------------|-------------------|")
    for r in batch_results:
        if "error" in r:
            print(f"| {r['batch_size']} | ERROR | - | - | - |")
        else:
            print(f"| {r['batch_size']} | {r['total_tokens']} | {r['wall_time']:.2f} | "
                  f"{r['throughput']:.1f} | {r['avg_decode']:.1f} |")


if __name__ == "__main__":
    print("=" * 80)
    print("GLM-5-4bit TP=2 Benchmark")
    print(f"Server: http://100.120.177.62:8000")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\n--- Part 1: Single Request Benchmarks ---")
    single_results = run_single_request_benchmarks()

    print("\n\n--- Part 2: Batch Throughput Benchmarks ---")
    batch_results = run_batch_benchmarks()

    print_results(single_results, batch_results)
