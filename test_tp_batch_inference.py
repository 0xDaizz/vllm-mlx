#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
TP=2 Batch Inference Verification Test

Verifies that tensor-parallel batch inference works correctly on
Kimi K2.5 across 2x Mac Studio (hwstudio1 + hwstudio2) via TB5 RDMA.

Test scenarios:
    1. Single request  -- verify basic TP inference works
    2. 4 concurrent    -- verify batching works correctly
    3. 8 concurrent    -- stress test

Run from the control MacBook:
    python test_tp_batch_inference.py
    python test_tp_batch_inference.py --skip-server-management
    python test_tp_batch_inference.py --server-url http://100.120.177.62:8000

Server configuration:
    Model:               ~/models/Kimi-K2.5
    Hosts:               hwstudio1, hwstudio2
    Backend:             JACCL (TB5 RDMA)
    Continuous Batching: Enabled
    KV Cache:            FP8 quantized
"""
from __future__ import annotations

import argparse
import os
import time
import traceback

from test_common import (
    DEFAULT_SERVER_URL,
    TP_HOSTS,
    RequestResult,
    collect_server_logs,
    send_chat_request,
    send_concurrent_requests,
    setup_logging,
    start_tp_server,
    stop_all_servers,
    wait_for_health,
    write_results_md,
)

# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

PROMPTS_4 = [
    "Explain general relativity briefly. What are its key predictions?",
    "What is photosynthesis? Describe the light-dependent and light-independent reactions.",
    "Compare Python and Rust programming languages in terms of performance, safety, and ecosystem.",
    "How does a modern CPU pipeline work? Explain pipelining, branch prediction, and OOO execution.",
]

PROMPTS_8 = PROMPTS_4 + [
    "Describe the history and evolution of jazz music from its origins in New Orleans to modern fusion.",
    "Explain the basics of machine learning: supervised, unsupervised, and reinforcement learning with examples.",
    "Discuss the causes and consequences of the French Revolution, including social and economic factors.",
    "Explain how blockchain technology works, including consensus mechanisms and smart contracts.",
]


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_single_request(server_url: str, logger) -> RequestResult:
    """Test 1: Single request to verify basic TP inference functionality."""
    logger.info("Sending single request...")
    messages = [{"role": "user", "content": "Explain quantum computing in 200 words."}]

    result = send_chat_request(
        server_url, messages, max_tokens=256, temperature=0.7,
    )

    logger.info(
        "Single request: success=%s prompt_tok=%d completion_tok=%d "
        "ttft=%.1fms decode=%.1ftok/s total=%.2fs",
        result.success, result.prompt_tokens, result.completion_tokens,
        result.ttft_ms, result.decode_tok_s, result.total_time_s,
    )

    if result.success:
        logger.debug("Output text (%d chars): %s",
                      len(result.output_text), result.output_text[:500])
    else:
        logger.error("Single request failed: %s", result.error)

    return result


def test_concurrent_4(server_url: str, logger) -> list[RequestResult]:
    """Test 2: 4 concurrent requests to verify batching."""
    logger.info("Sending 4 concurrent requests...")
    messages_list = [[{"role": "user", "content": p}] for p in PROMPTS_4]

    results = send_concurrent_requests(
        server_url, messages_list,
        max_tokens=256, concurrency=4, temperature=0.7,
    )

    successes = sum(1 for r in results if r.success)
    logger.info(
        "4 concurrent: %d/%d succeeded, avg_decode=%.1f tok/s",
        successes, len(results),
        sum(r.decode_tok_s for r in results if r.success) / max(successes, 1),
    )

    for i, r in enumerate(results):
        logger.debug(
            "  [%d] success=%s tokens=%d decode=%.1ftok/s output_len=%d",
            i, r.success, r.completion_tokens, r.decode_tok_s,
            len(r.output_text),
        )

    return results


def test_concurrent_8(server_url: str, logger) -> list[RequestResult]:
    """Test 3: 8 concurrent requests for stress testing."""
    logger.info("Sending 8 concurrent requests...")
    messages_list = [[{"role": "user", "content": p}] for p in PROMPTS_8]

    results = send_concurrent_requests(
        server_url, messages_list,
        max_tokens=256, concurrency=8, temperature=0.7,
    )

    successes = sum(1 for r in results if r.success)
    total_tokens = sum(r.completion_tokens for r in results if r.success)
    total_time = max(r.total_time_s for r in results) if results else 0
    aggregate_throughput = total_tokens / total_time if total_time > 0 else 0

    logger.info(
        "8 concurrent: %d/%d succeeded, total_tokens=%d, "
        "wall_time=%.1fs, aggregate=%.1f tok/s",
        successes, len(results), total_tokens,
        total_time, aggregate_throughput,
    )

    for i, r in enumerate(results):
        logger.debug(
            "  [%d] success=%s tokens=%d decode=%.1ftok/s output_len=%d",
            i, r.success, r.completion_tokens, r.decode_tok_s,
            len(r.output_text),
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify batch inference on TP=2 with Kimi K2.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s
  %(prog)s --skip-server-management
  %(prog)s --server-url http://100.120.177.62:8000 --timeout 900
""",
    )
    parser.add_argument(
        "--server-url", default=DEFAULT_SERVER_URL,
        help=f"Server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--skip-server-management", action="store_true",
        help="Skip starting/stopping server (assume already running)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Server startup timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    logger = setup_logging("tp_batch_inference")
    errors: list[str] = []
    all_results: list[tuple[str, list[RequestResult]]] = []

    logger.info("=" * 60)
    logger.info("TP Batch Inference Verification Test")
    logger.info("=" * 60)
    logger.info("Server URL: %s", args.server_url)
    logger.info("Hosts: %s", TP_HOSTS)
    logger.info("Skip server management: %s", args.skip_server_management)

    # -----------------------------------------------------------------------
    # Server startup
    # -----------------------------------------------------------------------
    if args.skip_server_management:
        if not wait_for_health(TP_HOSTS[0], timeout=30):
            logger.error("Server not reachable at %s", TP_HOSTS[0])
            return 1
    else:
        logger.info("Stopping any existing servers...")
        stop_all_servers(TP_HOSTS)
        time.sleep(5)

        logger.info("Starting TP server for batch inference test...")
        server_args = [
            "--continuous-batching",
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits", "8",
            "--port", "8000",
        ]
        if not start_tp_server(
            TP_HOSTS, os.path.expanduser("~/models/Kimi-K2.5"), server_args, timeout=args.timeout,
        ):
            errors.append("Failed to start TP server")
            logger.error("Server startup failed -- writing error report")
            config = {
                "Model": "Kimi K2.5 (612GB MoE, 4-bit)",
                "TP": "2 (hwstudio1 + hwstudio2)",
                "Backend": "JACCL (TB5 RDMA)",
                "Continuous Batching": "Enabled",
                "KV Cache Quantization": "FP8",
                "Status": "SERVER STARTUP FAILED",
            }
            server_logs = collect_server_logs(TP_HOSTS)
            write_results_md(
                "tp_batch_inference", config, all_results, errors,
                server_logs=server_logs,
            )
            return 1

    # -----------------------------------------------------------------------
    # Run tests
    # -----------------------------------------------------------------------
    try:
        # Test 1: Single request
        logger.info("")
        logger.info("=" * 40)
        logger.info("Test 1: Single request")
        logger.info("=" * 40)
        try:
            result = test_single_request(args.server_url, logger)
            all_results.append(("Single Request", [result]))
            if not result.success:
                errors.append(f"Test 1 (single request): request failed - {result.error}")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Test 1 failed: %s\n%s", exc, tb)
            errors.append(f"Test 1 (single request): {exc}")

        time.sleep(2)

        # Test 2: 4 concurrent
        logger.info("")
        logger.info("=" * 40)
        logger.info("Test 2: 4 concurrent requests")
        logger.info("=" * 40)
        try:
            results_4 = test_concurrent_4(args.server_url, logger)
            all_results.append(("4 Concurrent Requests", results_4))
            failed = [r for r in results_4 if not r.success]
            if failed:
                for r in failed:
                    errors.append(f"Test 2 (4 concurrent): request failed - {r.error}")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Test 2 failed: %s\n%s", exc, tb)
            errors.append(f"Test 2 (4 concurrent): {exc}")

        time.sleep(2)

        # Test 3: 8 concurrent
        logger.info("")
        logger.info("=" * 40)
        logger.info("Test 3: 8 concurrent requests")
        logger.info("=" * 40)
        try:
            results_8 = test_concurrent_8(args.server_url, logger)
            all_results.append(("8 Concurrent Requests", results_8))
            failed = [r for r in results_8 if not r.success]
            if failed:
                for r in failed:
                    errors.append(f"Test 3 (8 concurrent): request failed - {r.error}")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Test 3 failed: %s\n%s", exc, tb)
            errors.append(f"Test 3 (8 concurrent): {exc}")

    finally:
        # -------------------------------------------------------------------
        # Collect logs & shut down
        # -------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 40)
        logger.info("Collecting server logs...")
        logger.info("=" * 40)
        server_logs = collect_server_logs(TP_HOSTS)

        if not args.skip_server_management:
            logger.info("Stopping servers...")
            stop_all_servers(TP_HOSTS)

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------
    config = {
        "Model": "Kimi K2.5 (612GB MoE, 4-bit)",
        "TP": "2 (hwstudio1 + hwstudio2)",
        "Backend": "JACCL (TB5 RDMA)",
        "Continuous Batching": "Enabled",
        "KV Cache Quantization": "FP8",
    }

    md_path = write_results_md(
        "tp_batch_inference", config, all_results, errors,
        server_logs=server_logs,
    )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info("Results file: %s", md_path)
    logger.info("Errors: %d", len(errors))
    for e in errors:
        logger.info("  - %s", e)

    total_ok = sum(
        1 for _, section_results in all_results
        for r in section_results if r.success
    )
    total_count = sum(len(section_results) for _, section_results in all_results)
    logger.info("Requests: %d/%d successful", total_ok, total_count)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
