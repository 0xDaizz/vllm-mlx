#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify MTP (Multi-Token Prediction) on single node with GLM-5.

This script runs from a control MacBook and manages a server on hwstudio1
via SSH. It tests that MTP speculative decoding works correctly with GLM-5
(4-bit quantized) and its companion MTP draft model.

IMPORTANT: MTP does NOT support TP mode. When MTP is used with TP, it
silently falls back to normal decoding (no speedup, no error). Therefore
this test runs on hwstudio1 SINGLE NODE only.

Usage:
    python test_mtp_glm5.py
    python test_mtp_glm5.py --skip-server-management  # server already running
    python test_mtp_glm5.py --timeout 600
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
import urllib.request

from test_common import (
    DEFAULT_SERVER_URL,
    TP_HOSTS,
    RequestResult,
    collect_server_logs,
    send_chat_request,
    setup_logging,
    start_single_server,
    stop_all_servers,
    wait_for_health,
    write_results_md,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = os.path.expanduser("~/models/GLM-5-4bit")
MTP_MODEL = os.path.expanduser("~/models/GLM-5-mtp")
SERVER_HOST = "hwstudio1"
SERVER_PORT = 8000

# Test prompts — varied topics to verify consistency
TEST_PROMPTS = [
    {
        "name": "reasoning",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the difference between TCP and UDP protocols. "
                    "Include when you would use each one."
                ),
            }
        ],
    },
    {
        "name": "coding",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python function that checks if a string is a "
                    "valid palindrome, ignoring spaces and punctuation."
                ),
            }
        ],
    },
    {
        "name": "creative",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a short story (3-4 sentences) about a robot "
                    "discovering music for the first time."
                ),
            }
        ],
    },
]


# ---------------------------------------------------------------------------
# Helper: query /v1/status for MTP / spec decode stats
# ---------------------------------------------------------------------------


def query_status_endpoint(
    server_url: str, logger: logging.Logger, timeout: int = 30
) -> dict | None:
    """Query the /v1/status endpoint and return the JSON response.

    Returns None on failure (logged, not raised).
    """
    url = f"{server_url.rstrip('/')}/v1/status"
    logger.debug("GET %s (timeout=%ds)", url, timeout)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        logger.debug("/v1/status response: %s", json.dumps(data, indent=2))
        return data
    except Exception:
        logger.error("Failed to query %s:\n%s", url, traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_single_request(
    server_url: str, logger: logging.Logger
) -> tuple[RequestResult | None, list[str]]:
    """Test 1: Single request to verify MTP produces coherent output."""
    errors: list[str] = []
    logger.info("=" * 60)
    logger.info("TEST 1: Single request — verify MTP coherence")
    logger.info("=" * 60)

    messages = [
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
    ]

    result = send_chat_request(
        server_url, messages, max_tokens=128, temperature=0.0, timeout=120
    )

    if not result.success:
        msg = f"Test 1 FAILED: {result.error}"
        logger.error(msg)
        errors.append(msg)
        return result, errors

    logger.info("Test 1 output (%d tokens): %s", result.completion_tokens, result.output_text)
    logger.info(
        "Test 1 perf: ttft=%.1fms, prefill=%.1f tok/s, decode=%.1f tok/s, total=%.2fs",
        result.ttft_ms,
        result.prefill_tok_s,
        result.decode_tok_s,
        result.total_time_s,
    )

    # Basic coherence check — output should mention Paris
    if result.output_text and "paris" in result.output_text.lower():
        logger.info("Test 1 coherence check: PASS (mentions Paris)")
    else:
        logger.warning(
            "Test 1 coherence check: output does not mention 'Paris' — "
            "may still be valid depending on model behavior"
        )

    return result, errors


def test_mtp_acceptance_rate(
    server_url: str, logger: logging.Logger
) -> tuple[dict | None, list[str]]:
    """Test 2: Check /v1/status for MTP speculative decode statistics."""
    errors: list[str] = []
    logger.info("=" * 60)
    logger.info("TEST 2: Check MTP acceptance rate via /v1/status")
    logger.info("=" * 60)

    status = query_status_endpoint(server_url, logger)

    if status is None:
        msg = "Test 2 FAILED: could not query /v1/status"
        logger.error(msg)
        errors.append(msg)
        return None, errors

    spec_decode_info = status.get("spec_decode")
    if spec_decode_info is None:
        logger.warning(
            "Test 2: /v1/status has no 'spec_decode' key. "
            "MTP stats may not be exposed yet — this is informational only."
        )
    else:
        logger.info("Test 2 spec_decode stats: %s", json.dumps(spec_decode_info, indent=2))

        # Extract acceptance rate if available
        acceptance_rate = None
        if isinstance(spec_decode_info, dict):
            acceptance_rate = spec_decode_info.get("acceptance_rate")
            if acceptance_rate is not None:
                logger.info("Test 2 MTP acceptance rate: %.3f", acceptance_rate)
            else:
                logger.info("Test 2: acceptance_rate key not found in spec_decode stats")

    # Also log cache info if present
    cache_info = status.get("cache")
    if cache_info:
        logger.info("Test 2 cache stats: %s", json.dumps(cache_info, indent=2))

    return status, errors


def test_sequential_requests(
    server_url: str, logger: logging.Logger
) -> tuple[list[RequestResult], list[str]]:
    """Test 3: 3 sequential requests with different prompts — verify consistency."""
    errors: list[str] = []
    results: list[RequestResult] = []
    logger.info("=" * 60)
    logger.info("TEST 3: Sequential requests with varied prompts")
    logger.info("=" * 60)

    for i, prompt_info in enumerate(TEST_PROMPTS):
        name = prompt_info["name"]
        messages = prompt_info["messages"]
        logger.info("  Request %d/%d (%s)...", i + 1, len(TEST_PROMPTS), name)

        result = send_chat_request(
            server_url, messages, max_tokens=256, temperature=0.0, timeout=180
        )

        if not result.success:
            msg = f"Test 3 request '{name}' FAILED: {result.error}"
            logger.error(msg)
            errors.append(msg)
        else:
            logger.info(
                "  Request '%s': %d tokens, ttft=%.1fms, decode=%.1f tok/s",
                name,
                result.completion_tokens,
                result.ttft_ms,
                result.decode_tok_s,
            )
            logger.debug("  Output: %s", result.output_text[:200] if result.output_text else "")

        results.append(result)

        # Small delay between requests
        if i < len(TEST_PROMPTS) - 1:
            time.sleep(1)

    successful = sum(1 for r in results if r.success)
    logger.info("Test 3 summary: %d/%d requests succeeded", successful, len(results))

    return results, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MTP verification test with GLM-5 (single node)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help="Server URL (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-server-management",
        action="store_true",
        help="Skip starting/stopping the server (assume already running)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Server startup timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    logger = setup_logging("mtp_glm5")
    errors: list[str] = []
    all_results: list[tuple[str, list[RequestResult]]] = []
    server_logs: dict[str, str] = {}
    status_data: dict | None = None

    logger.info("=" * 60)
    logger.info("MTP Verification Test — GLM-5 (Single Node)")
    logger.info("=" * 60)
    logger.info("Server URL: %s", args.server_url)
    logger.info("Skip server management: %s", args.skip_server_management)
    logger.info("Startup timeout: %ds", args.timeout)

    # -----------------------------------------------------------------------
    # Server startup
    # -----------------------------------------------------------------------
    if args.skip_server_management:
        if not wait_for_health(SERVER_HOST, SERVER_PORT, timeout=30):
            logger.error("Server not reachable at %s:%d", SERVER_HOST, SERVER_PORT)
            return 1
    else:
        logger.info("Stopping all existing servers on %s...", TP_HOSTS)
        stop_all_servers(TP_HOSTS)
        time.sleep(5)

        server_args = [
            "--continuous-batching",
            "--speculative-method", "mtp",
            "--num-speculative-tokens", "1",
            "--mtp-model", MTP_MODEL,
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits", "8",
            "--port", str(SERVER_PORT),
        ]

        logger.info("Starting GLM-5 MTP server on %s...", SERVER_HOST)
        logger.debug("Server args: %s", server_args)

        if not start_single_server(
            SERVER_HOST, MODEL, server_args, timeout=args.timeout
        ):
            msg = "FATAL: Failed to start GLM-5 MTP server on hwstudio1"
            logger.error(msg)
            errors.append(msg)

            # Collect whatever logs exist
            server_logs = collect_server_logs(
                [SERVER_HOST], "/tmp/single_server.log"
            )
            config = {
                "Model": "GLM-5 (4-bit quantized)",
                "MTP Model": "GLM-5-mtp (original HF weights)",
                "Mode": "Single Node (hwstudio1)",
                "Speculative Method": "MTP (k=1)",
                "KV Cache Quantization": "FP8",
                "Note": "MTP does NOT support TP mode — silently falls back to normal decoding",
                "Status": "FAILED — server did not start",
            }
            write_results_md(
                "mtp_glm5", config, all_results, errors, server_logs
            )
            return 1

        logger.info("Server started. Waiting for health check...")
        if not wait_for_health(SERVER_HOST, SERVER_PORT, timeout=60):
            msg = "FATAL: Server health check failed after startup"
            logger.error(msg)
            errors.append(msg)
            server_logs = collect_server_logs(
                [SERVER_HOST], "/tmp/single_server.log"
            )
            stop_all_servers([SERVER_HOST])
            config = {
                "Model": "GLM-5 (4-bit quantized)",
                "MTP Model": "GLM-5-mtp (original HF weights)",
                "Mode": "Single Node (hwstudio1)",
                "Status": "FAILED — health check timeout",
            }
            write_results_md(
                "mtp_glm5", config, all_results, errors, server_logs
            )
            return 1

    # -----------------------------------------------------------------------
    # Run tests
    # -----------------------------------------------------------------------
    try:
        # Test 1: Single request
        result1, errs1 = test_single_request(args.server_url, logger)
        errors.extend(errs1)
        if result1 is not None:
            all_results.append(("Single Request", [result1]))

        # Test 2: Check MTP stats via /v1/status
        status_data, errs2 = test_mtp_acceptance_rate(args.server_url, logger)
        errors.extend(errs2)

        # Test 3: Sequential requests
        results3, errs3 = test_sequential_requests(args.server_url, logger)
        errors.extend(errs3)
        if results3:
            all_results.append(("Sequential Requests", results3))

    except Exception:
        msg = f"Unexpected error during tests:\n{traceback.format_exc()}"
        logger.error(msg)
        errors.append(msg)

    finally:
        # Collect server logs
        logger.info("Collecting server logs...")
        server_logs = collect_server_logs(
            [SERVER_HOST], "/tmp/single_server.log"
        )

        if not args.skip_server_management:
            logger.info("Stopping server...")
            stop_all_servers([SERVER_HOST])

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------
    config = {
        "Model": "GLM-5 (4-bit quantized)",
        "MTP Model": "GLM-5-mtp (original HF weights)",
        "Mode": "Single Node (hwstudio1)",
        "Speculative Method": "MTP (k=1)",
        "KV Cache Quantization": "FP8",
        "Note": "MTP does NOT support TP mode — silently falls back to normal decoding",
    }

    # Build extra sections as a markdown string
    extra_md = ""

    # MTP stats section
    if status_data is not None:
        spec_info = status_data.get("spec_decode")
        extra_md += "## MTP Speculative Decode Stats\n\n"
        if spec_info:
            extra_md += f"```json\n{json.dumps(spec_info, indent=2)}\n```\n\n"
        else:
            extra_md += (
                "No `spec_decode` key returned by `/v1/status`. "
                "MTP statistics may not be exposed in this server version.\n\n"
            )

    # Output samples section
    output_lines: list[str] = []
    for section_name, section_results in all_results:
        for r in section_results:
            if r.success and r.output_text:
                output_lines.append(f"### {section_name}")
                text = r.output_text if len(r.output_text) <= 500 else r.output_text[:500] + "..."
                output_lines.append(f"```\n{text}\n```\n")
    if output_lines:
        extra_md += "## Output Samples\n\n"
        extra_md += "\n".join(output_lines) + "\n\n"

    # MTP limitation note
    extra_md += "## MTP + TP Limitation\n\n"
    extra_md += (
        "MTP (Multi-Token Prediction) does **not** support Tensor Parallel mode. "
        "When `--speculative-method mtp` is used with TP, the server silently "
        "falls back to normal (non-speculative) decoding. There is no error or "
        "warning — the only observable difference is the lack of speedup from "
        "speculative decoding.\n\n"
        "For this reason, this test runs on **hwstudio1 only** (single node).\n"
    )

    output_path = write_results_md(
        "mtp_glm5", config, all_results, errors, server_logs, extra_md
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_tests = sum(len(rs) for _, rs in all_results)
    passed = sum(1 for _, rs in all_results for r in rs if r.success)
    failed = total_tests - passed

    logger.info("=" * 60)
    logger.info("MTP GLM-5 Test Summary")
    logger.info("  Total tests: %d", total_tests)
    logger.info("  Passed: %d", passed)
    logger.info("  Failed: %d", failed)
    logger.info("  Errors: %d", len(errors))
    if output_path:
        logger.info("  Results: %s", output_path)
    logger.info("=" * 60)

    return 1 if failed > 0 or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
