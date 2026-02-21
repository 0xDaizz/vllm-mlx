#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
TP=2 Speculative Decoding Verification Test

Verifies that draft-model speculative decoding works correctly on
TP=2 with Kimi K2.5 (target) + Moonlight 16B (draft) across 2x Mac
Studio (hwstudio1 + hwstudio2) via TB5 RDMA.

IMPORTANT: Only k=1 (--num-speculative-tokens 1) is tested.
    k>=2 has a known Bug 7 -- deterministic deadlock at step 50
    caused by cache_idx=242 all_sum deadlock. See
    docs/spec_decode_tp_bugs.md for the full investigation.

Test scenarios:
    1. Single request    -- verify spec decode + TP works
    2. Acceptance rate   -- query /v1/status for spec decode stats
    3. 3 sequential      -- verify consistency across requests
    4. Output quality    -- check output is coherent (not garbage)

Run from the control MacBook:
    python test_tp_spec_decode.py
    python test_tp_spec_decode.py --skip-server-management
    python test_tp_spec_decode.py --server-url http://100.120.177.62:8000

Server configuration:
    Target model:          ~/models/Kimi-K2.5
    Draft model:           ~/models/Moonlight-16B-A3B-Instruct-4-bit
    Hosts:                 hwstudio1, hwstudio2
    Backend:               JACCL (TB5 RDMA)
    Speculative method:    draft_model
    Num speculative tokens: 1  (k>=2 deadlocks -- Bug 7)
    Continuous Batching:   Enabled
    KV Cache:              FP8 quantized
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
import urllib.error
import urllib.request

from test_common import (
    DEFAULT_SERVER_URL,
    TP_HOSTS,
    RequestResult,
    collect_server_logs,
    send_chat_request,
    setup_logging,
    start_tp_server,
    stop_all_servers,
    wait_for_health,
    write_results_md,
)


# ---------------------------------------------------------------------------
# Quality heuristics
# ---------------------------------------------------------------------------

# Minimum distinct-word ratio to consider output "coherent".
# Repetitive garbage (e.g., "the the the the") has a very low ratio.
_MIN_DISTINCT_WORD_RATIO = 0.20

# Minimum output length (characters) to bother checking quality.
_MIN_OUTPUT_LEN = 40


def _check_output_quality(text: str, logger: logging.Logger) -> tuple[bool, str]:
    """Check whether *text* looks like coherent natural language.

    Returns:
        ``(is_coherent, reason)``
    """
    if not text or len(text.strip()) < _MIN_OUTPUT_LEN:
        reason = f"Output too short ({len(text.strip())} chars)"
        logger.warning("Quality check: %s", reason)
        return False, reason

    words = text.split()
    if not words:
        return False, "No words in output"

    # Check distinct-word ratio
    distinct = len(set(w.lower() for w in words))
    ratio = distinct / len(words)
    logger.debug(
        "Quality check: %d words, %d distinct, ratio=%.3f (threshold=%.3f)",
        len(words), distinct, ratio, _MIN_DISTINCT_WORD_RATIO,
    )

    if ratio < _MIN_DISTINCT_WORD_RATIO:
        reason = (
            f"Low distinct-word ratio {ratio:.3f} < {_MIN_DISTINCT_WORD_RATIO} "
            f"({distinct}/{len(words)} words) -- likely repetitive garbage"
        )
        logger.warning("Quality check FAIL: %s", reason)
        return False, reason

    # Check for null bytes or binary garbage
    if "\x00" in text:
        reason = "Output contains null bytes"
        logger.warning("Quality check FAIL: %s", reason)
        return False, reason

    logger.debug("Quality check PASS: ratio=%.3f, len=%d", ratio, len(text))
    return True, "OK"


# ---------------------------------------------------------------------------
# Spec decode status query
# ---------------------------------------------------------------------------


def query_spec_decode_status(server_url: str, logger: logging.Logger) -> dict | None:
    """Query ``GET /v1/status`` and return the ``spec_decode`` section.

    Returns:
        The ``spec_decode`` dict from the status response, or ``None``
        if the endpoint is unavailable or does not report spec decode stats.
    """
    url = f"{server_url.rstrip('/')}/v1/status"
    logger.info("Querying spec decode status: %s", url)

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            logger.debug("Status response (%d bytes): %s", len(body), body[:500])
            data = json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to query /v1/status: %s", exc)
        return None

    spec_decode = data.get("spec_decode")
    if spec_decode:
        logger.info("Spec decode stats: %s", json.dumps(spec_decode, indent=2))
    else:
        logger.warning("No spec_decode section in /v1/status response")
        logger.debug("Full status response: %s", json.dumps(data, indent=2))

    return spec_decode


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_single_request(server_url: str, logger: logging.Logger) -> RequestResult:
    """Test 1: Single request to verify spec decode + TP works."""
    logger.info("Sending single request (spec decode + TP)...")
    messages = [
        {"role": "user", "content": "Explain how neural networks learn through backpropagation in 200 words."},
    ]

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


def test_acceptance_rate(
    server_url: str, logger: logging.Logger,
) -> dict | None:
    """Test 2: Query /v1/status for spec decode acceptance rate.

    This test does NOT send a new request; it reads the stats accumulated
    from all previous requests in this session.
    """
    return query_spec_decode_status(server_url, logger)


def test_sequential_3(
    server_url: str, logger: logging.Logger,
) -> list[RequestResult]:
    """Test 3: 3 sequential requests to verify consistency."""
    prompts = [
        "What is the theory of general relativity? Explain briefly.",
        "Describe the process of photosynthesis step by step.",
        "Compare Python and Rust as programming languages.",
    ]

    results: list[RequestResult] = []
    for i, prompt in enumerate(prompts, 1):
        logger.info("Sequential request %d/3...", i)
        messages = [{"role": "user", "content": prompt}]
        result = send_chat_request(
            server_url, messages, max_tokens=256, temperature=0.7,
        )
        logger.info(
            "  Request %d: success=%s tokens=%d decode=%.1ftok/s",
            i, result.success, result.completion_tokens, result.decode_tok_s,
        )
        if result.success:
            logger.debug("  Output (%d chars): %s",
                          len(result.output_text), result.output_text[:300])
        else:
            logger.error("  Request %d failed: %s", i, result.error)

        results.append(result)
        # Small pause between sequential requests
        time.sleep(1)

    return results


def test_output_quality(
    results: list[RequestResult], logger: logging.Logger,
) -> list[str]:
    """Test 4: Verify output text is coherent (not corrupted garbage).

    Checks all successful results from previous tests.

    Returns:
        A list of quality-issue descriptions (empty = all OK).
    """
    logger.info("Checking output quality for %d results...", len(results))
    issues: list[str] = []

    for i, r in enumerate(results, 1):
        if not r.success:
            logger.debug("Skipping failed result %d", i)
            continue

        is_coherent, reason = _check_output_quality(r.output_text, logger)
        if not is_coherent:
            msg = f"Result {i}: quality issue -- {reason}"
            logger.warning(msg)
            issues.append(msg)
        else:
            logger.debug("Result %d: quality OK", i)

    if not issues:
        logger.info("All outputs passed quality check")
    else:
        logger.warning("%d output(s) have quality issues", len(issues))

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify speculative decoding on TP=2 with Kimi K2.5 + Moonlight 16B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s
  %(prog)s --skip-server-management
  %(prog)s --server-url http://100.120.177.62:8000 --timeout 900

NOTE: Only k=1 is tested. k>=2 has a known Bug 7 (deadlock at step 50).
      See docs/spec_decode_tp_bugs.md for details.
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

    logger = setup_logging("tp_spec_decode")
    errors: list[str] = []
    all_results: list[tuple[str, list[RequestResult]]] = []
    all_request_results: list[RequestResult] = []  # flat, for quality check
    spec_decode_stats: dict | None = None

    logger.info("=" * 60)
    logger.info("TP Speculative Decoding Verification Test")
    logger.info("=" * 60)
    logger.info("Server URL: %s", args.server_url)
    logger.info("Hosts: %s", TP_HOSTS)
    logger.info("Skip server management: %s", args.skip_server_management)
    logger.info("")
    logger.info("LIMITATION: Only k=1 is tested (--num-speculative-tokens 1).")
    logger.info("k>=2 has a known Bug 7: deterministic deadlock at step 50.")
    logger.info("See docs/spec_decode_tp_bugs.md for the full investigation.")

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

        logger.info("Starting TP server with speculative decoding...")
        server_args = [
            "--continuous-batching",
            "--speculative-method", "draft_model",
            "--draft-model", os.path.expanduser("~/models/Moonlight-16B-A3B-Instruct-4-bit"),
            "--num-speculative-tokens", "1",
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits", "8",
            "--port", "8000",
        ]
        if not start_tp_server(
            TP_HOSTS, os.path.expanduser("~/models/Kimi-K2.5"), server_args, timeout=args.timeout,
        ):
            errors.append("Failed to start TP server with spec decode")
            logger.error("Server startup failed -- writing error report")
            config = {
                "Target Model": "Kimi K2.5 (612GB MoE, 4-bit)",
                "Draft Model": "Moonlight 16B A3B Instruct (4-bit)",
                "TP": "2 (hwstudio1 + hwstudio2)",
                "Backend": "JACCL (TB5 RDMA)",
                "Speculative Method": "draft_model",
                "Num Speculative Tokens": "1 (k>=2 deadlocks -- Bug 7)",
                "Continuous Batching": "Enabled",
                "KV Cache Quantization": "FP8",
                "Status": "SERVER STARTUP FAILED",
            }
            server_logs = collect_server_logs(TP_HOSTS)
            write_results_md(
                "tp_spec_decode", config, all_results, errors,
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
        logger.info("Test 1: Single request (spec decode + TP)")
        logger.info("=" * 40)
        try:
            result = test_single_request(args.server_url, logger)
            all_results.append(("Single Request (Spec Decode)", [result]))
            all_request_results.append(result)
            if not result.success:
                errors.append(f"Test 1 (single request): request failed - {result.error}")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Test 1 failed: %s\n%s", exc, tb)
            errors.append(f"Test 1 (single request): {exc}")

        time.sleep(2)

        # Test 2: Acceptance rate
        logger.info("")
        logger.info("=" * 40)
        logger.info("Test 2: Check spec decode acceptance rate")
        logger.info("=" * 40)
        try:
            spec_decode_stats = test_acceptance_rate(args.server_url, logger)
            if spec_decode_stats is None:
                logger.warning(
                    "Could not retrieve spec decode stats. The /v1/status "
                    "endpoint may not be available or spec decode may not be active."
                )
                errors.append(
                    "Test 2 (acceptance rate): /v1/status did not return spec_decode stats"
                )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Test 2 failed: %s\n%s", exc, tb)
            errors.append(f"Test 2 (acceptance rate): {exc}")

        time.sleep(2)

        # Test 3: 3 sequential requests
        logger.info("")
        logger.info("=" * 40)
        logger.info("Test 3: 3 sequential requests")
        logger.info("=" * 40)
        try:
            results_seq = test_sequential_3(args.server_url, logger)
            all_results.append(("3 Sequential Requests", results_seq))
            all_request_results.extend(results_seq)
            failed = [r for r in results_seq if not r.success]
            if failed:
                for r in failed:
                    errors.append(f"Test 3 (3 sequential): request failed - {r.error}")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Test 3 failed: %s\n%s", exc, tb)
            errors.append(f"Test 3 (3 sequential): {exc}")

        time.sleep(1)

        # Re-check acceptance rate after more requests
        logger.info("")
        logger.info("Re-checking acceptance rate after sequential requests...")
        try:
            spec_decode_stats = test_acceptance_rate(args.server_url, logger)
        except Exception as exc:
            logger.warning("Post-test acceptance rate check failed: %s", exc)

        # Test 4: Output quality
        logger.info("")
        logger.info("=" * 40)
        logger.info("Test 4: Verify output quality")
        logger.info("=" * 40)
        try:
            quality_issues = test_output_quality(all_request_results, logger)
            if quality_issues:
                for issue in quality_issues:
                    errors.append(f"Test 4 (quality): {issue}")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Test 4 failed: %s\n%s", exc, tb)
            errors.append(f"Test 4 (output quality): {exc}")

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
    # Build acceptance rate section for the report
    # -----------------------------------------------------------------------
    extra_sections = ""
    if spec_decode_stats:
        extra_sections = "## Speculative Decoding Stats\n\n"
        extra_sections += "| Metric | Value |\n"
        extra_sections += "|--------|-------|\n"
        for key, value in spec_decode_stats.items():
            if isinstance(value, float):
                extra_sections += f"| {key} | {value:.4f} |\n"
            elif isinstance(value, list):
                formatted = ", ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in value)
                extra_sections += f"| {key} | [{formatted}] |\n"
            else:
                extra_sections += f"| {key} | {value} |\n"
        extra_sections += "\n"
        extra_sections += (
            "> **Note:** Only k=1 (1 speculative token) is tested. "
            "k>=2 has a known Bug 7 (deterministic deadlock at step 50 "
            "due to cache_idx=242 all_sum deadlock). "
            "See `docs/spec_decode_tp_bugs.md` for the full investigation.\n"
        )

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------
    config = {
        "Target Model": "Kimi K2.5 (612GB MoE, 4-bit)",
        "Draft Model": "Moonlight 16B A3B Instruct (4-bit)",
        "TP": "2 (hwstudio1 + hwstudio2)",
        "Backend": "JACCL (TB5 RDMA)",
        "Speculative Method": "draft_model",
        "Num Speculative Tokens": "1 (k>=2 deadlocks -- Bug 7)",
        "Continuous Batching": "Enabled",
        "KV Cache Quantization": "FP8",
    }

    md_path = write_results_md(
        "tp_spec_decode", config, all_results, errors,
        server_logs=server_logs,
        extra_sections=extra_sections,
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

    if spec_decode_stats:
        ar = spec_decode_stats.get("acceptance_rate", "N/A")
        logger.info("Acceptance rate: %s", ar)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
