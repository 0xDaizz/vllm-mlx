#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify hash-based KV cache prefix matching on TP=2 with Kimi K2.5.

This script runs from a control MacBook and manages servers on hwstudio1
and hwstudio2 via SSH. It tests that prefix caching works correctly
under tensor-parallel distributed inference: a long shared prefix should
be cached after the first request, making subsequent requests with the
same prefix significantly faster (lower TTFT, higher prefill tok/s).

Phase 1: In-memory prefix caching
  - Request A: cache miss (first time seeing the prefix)
  - Request B: cache hit (same prefix, different question)
  - Request C: cache hit (same prefix, third question)

Phase 2 (optional): SSD persistence
  - Stop server (triggers automatic SSD save)
  - Restart server (triggers automatic SSD load)
  - Request D: should be a cache hit from SSD-restored cache

Usage:
    python test_tp_kv_cache.py
    python test_tp_kv_cache.py --skip-server-management
    python test_tp_kv_cache.py --skip-phase2
    python test_tp_kv_cache.py --timeout 600
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
    get_host_ip,
    send_chat_request,
    setup_logging,
    start_tp_server,
    stop_all_servers,
    wait_for_health,
    write_results_md,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = os.path.expanduser("~/models/Kimi-K2.5")
SERVER_PORT = 8000

SERVER_ARGS = [
    "--continuous-batching",
    "--kv-cache-quantization",
    "--kv-cache-quantization-bits", "8",
    "--prefix-cache-size", "500",
    "--port", str(SERVER_PORT),
]

# ---------------------------------------------------------------------------
# Long prefix generation (~5500-7500 tokens, ~30000 characters)
#
# Each paragraph is ~300 words / ~400 tokens.
# 13 sections * ~400 tokens = ~5200 tokens (actual count depends on tokenizer).
# The chars/4 heuristic overestimates; real token count is typically lower.
# ---------------------------------------------------------------------------

BASE_PARAGRAPH = """
Quantum computing represents a fundamental shift in computational paradigm, leveraging \
quantum mechanical phenomena such as superposition and entanglement to process information \
in ways that classical computers cannot efficiently replicate. Unlike classical bits, which \
exist in a definite state of either 0 or 1, quantum bits (qubits) can exist in a \
superposition of both states simultaneously, enabling quantum computers to explore multiple \
solution paths in parallel. This property becomes exponentially powerful as the number of \
qubits increases: while n classical bits can represent exactly one of 2^n states at any \
given time, n qubits can represent a superposition of all 2^n states simultaneously.

The practical implications of quantum computing extend across numerous fields. In \
cryptography, Shor's algorithm threatens current RSA encryption by efficiently factoring \
large numbers. In drug discovery, quantum simulation can model molecular interactions with \
unprecedented accuracy. In optimization, quantum annealing and variational quantum \
eigensolvers offer potential speedups for complex combinatorial problems encountered in \
logistics, finance, and machine learning. Google's demonstration of quantum supremacy in \
2019 with their Sycamore processor, completing a specific calculation in 200 seconds that \
would take classical supercomputers approximately 10,000 years, marked a significant \
milestone in the field.

However, significant challenges remain before quantum computing achieves broad practical \
utility. Decoherence — the loss of quantum information due to environmental interference — \
limits the duration of quantum computations. Error correction requires significant \
overhead, with current estimates suggesting that thousands of physical qubits may be needed \
to create a single logical qubit with sufficiently low error rates. The extreme cooling \
requirements of superconducting qubits (operating near absolute zero at approximately 15 \
millikelvin) present engineering challenges for scaling. Despite these obstacles, major \
technology companies and research institutions continue to invest heavily in quantum \
computing research, with IBM, Google, Microsoft, and numerous startups racing to achieve \
practical quantum advantage across various application domains.
"""  # ~300 words, ~400 tokens

# Build long prefix: 13 sections to target ~6000-7500 tokens
_NUM_SECTIONS = 13
LONG_PREFIX = "\n\n".join(
    [
        f"=== Section {i + 1}: Quantum Computing Overview (Part {i + 1} of {_NUM_SECTIONS}) ===\n"
        + BASE_PARAGRAPH.strip()
        for i in range(_NUM_SECTIONS)
    ]
)

# Approximate token count for logging (1 token ~ 4 chars in English)
_APPROX_PREFIX_TOKENS = len(LONG_PREFIX) // 4

# Questions that reference the long prefix
QUESTIONS = [
    "Based on the text above, what are the three main challenges preventing quantum computing from achieving broad practical utility?",
    "Summarize all the practical applications of quantum computing mentioned across the sections above. List each application domain and the specific quantum technique or algorithm involved.",
    "What is quantum supremacy, who demonstrated it, and what were the specific details of the demonstration described in the text?",
    "Compare and contrast the different qubit technologies described in the text: superconducting circuits, trapped ions, photonic qubits, and topological qubits. What are the tradeoffs?",
]


# ---------------------------------------------------------------------------
# Helper: query /v1/status for cache stats
# ---------------------------------------------------------------------------


def query_status(
    server_url: str, logger: logging.Logger, timeout: int = 30
) -> dict | None:
    """Query /v1/status and return the JSON response, or None on error."""
    url = f"{server_url.rstrip('/')}/v1/status"
    logger.debug("GET %s", url)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        logger.debug("/v1/status: %s", json.dumps(data, indent=2))
        return data
    except Exception:
        logger.error("Failed to query %s:\n%s", url, traceback.format_exc())
        return None


def log_cache_stats(
    server_url: str, label: str, logger: logging.Logger
) -> dict | None:
    """Query and log cache stats with a label."""
    status = query_status(server_url, logger)
    if status and status.get("cache"):
        logger.info("[%s] cache stats: %s", label, json.dumps(status["cache"], indent=2))
    return status


# ---------------------------------------------------------------------------
# Phase 1: In-memory prefix caching tests
# ---------------------------------------------------------------------------


def run_phase1(
    server_url: str, logger: logging.Logger
) -> tuple[list[dict], list[str]]:
    """Phase 1: In-memory prefix caching.

    Returns (results_list, errors_list) where each result dict has:
        label, type, result (RequestResult), cache_stats (dict|None)
    """
    errors: list[str] = []
    results: list[dict] = []

    logger.info("=" * 60)
    logger.info("PHASE 1: In-memory prefix caching")
    logger.info("=" * 60)
    logger.info("Prefix length: ~%d characters (~%d tokens)", len(LONG_PREFIX), _APPROX_PREFIX_TOKENS)

    # Log initial cache state
    log_cache_stats(server_url, "before-phase1", logger)

    # --- Request A: Cache MISS (first time) ---
    logger.info("-" * 40)
    logger.info("Request A: Cache MISS (first request with long prefix)")
    logger.info("Question: %s", QUESTIONS[0][:80])

    messages_a = [
        {"role": "system", "content": LONG_PREFIX},
        {"role": "user", "content": QUESTIONS[0]},
    ]

    t_start_a = time.monotonic()
    result_a = send_chat_request(
        server_url, messages_a, max_tokens=256, temperature=0.0, timeout=600
    )
    t_elapsed_a = time.monotonic() - t_start_a

    if not result_a.success:
        msg = f"Request A FAILED: {result_a.error}"
        logger.error(msg)
        errors.append(msg)
    else:
        logger.info(
            "Request A: ttft=%.1fms, prefill=%.1f tok/s, decode=%.1f tok/s, "
            "total=%.2fs, prompt_tokens=%d, completion_tokens=%d",
            result_a.ttft_ms, result_a.prefill_tok_s, result_a.decode_tok_s,
            result_a.total_time_s, result_a.prompt_tokens, result_a.completion_tokens,
        )
        logger.debug("Request A output: %s", result_a.output_text[:200] if result_a.output_text else "")

    cache_a = log_cache_stats(server_url, "after-A", logger)
    results.append({
        "label": "A",
        "type": "Cache MISS",
        "result": result_a,
        "cache_stats": cache_a.get("cache") if cache_a else None,
    })

    # Small delay to ensure cache is fully committed
    time.sleep(2)

    # --- Request B: Cache HIT (same prefix, different question) ---
    logger.info("-" * 40)
    logger.info("Request B: Cache HIT (same prefix, question #2)")
    logger.info("Question: %s", QUESTIONS[1][:80])

    messages_b = [
        {"role": "system", "content": LONG_PREFIX},
        {"role": "user", "content": QUESTIONS[1]},
    ]

    t_start_b = time.monotonic()
    result_b = send_chat_request(
        server_url, messages_b, max_tokens=256, temperature=0.0, timeout=600
    )
    t_elapsed_b = time.monotonic() - t_start_b

    if not result_b.success:
        msg = f"Request B FAILED: {result_b.error}"
        logger.error(msg)
        errors.append(msg)
    else:
        logger.info(
            "Request B: ttft=%.1fms, prefill=%.1f tok/s, decode=%.1f tok/s, "
            "total=%.2fs, prompt_tokens=%d, completion_tokens=%d",
            result_b.ttft_ms, result_b.prefill_tok_s, result_b.decode_tok_s,
            result_b.total_time_s, result_b.prompt_tokens, result_b.completion_tokens,
        )
        logger.debug("Request B output: %s", result_b.output_text[:200] if result_b.output_text else "")

    cache_b = log_cache_stats(server_url, "after-B", logger)
    results.append({
        "label": "B",
        "type": "Cache HIT",
        "result": result_b,
        "cache_stats": cache_b.get("cache") if cache_b else None,
    })

    time.sleep(2)

    # --- Request C: Cache HIT (same prefix, question #3) ---
    logger.info("-" * 40)
    logger.info("Request C: Cache HIT (same prefix, question #3)")
    logger.info("Question: %s", QUESTIONS[2][:80])

    messages_c = [
        {"role": "system", "content": LONG_PREFIX},
        {"role": "user", "content": QUESTIONS[2]},
    ]

    t_start_c = time.monotonic()
    result_c = send_chat_request(
        server_url, messages_c, max_tokens=256, temperature=0.0, timeout=600
    )
    t_elapsed_c = time.monotonic() - t_start_c

    if not result_c.success:
        msg = f"Request C FAILED: {result_c.error}"
        logger.error(msg)
        errors.append(msg)
    else:
        logger.info(
            "Request C: ttft=%.1fms, prefill=%.1f tok/s, decode=%.1f tok/s, "
            "total=%.2fs, prompt_tokens=%d, completion_tokens=%d",
            result_c.ttft_ms, result_c.prefill_tok_s, result_c.decode_tok_s,
            result_c.total_time_s, result_c.prompt_tokens, result_c.completion_tokens,
        )
        logger.debug("Request C output: %s", result_c.output_text[:200] if result_c.output_text else "")

    cache_c = log_cache_stats(server_url, "after-C", logger)
    results.append({
        "label": "C",
        "type": "Cache HIT",
        "result": result_c,
        "cache_stats": cache_c.get("cache") if cache_c else None,
    })

    # --- Phase 1 summary ---
    logger.info("-" * 40)
    logger.info("Phase 1 Summary:")
    if result_a.success and result_b.success:
        if result_a.ttft_ms and result_b.ttft_ms and result_a.ttft_ms > 0:
            speedup = result_a.ttft_ms / result_b.ttft_ms
            logger.info(
                "  TTFT speedup (A->B): %.1fx (%.1fms -> %.1fms)",
                speedup, result_a.ttft_ms, result_b.ttft_ms,
            )
        if result_a.prefill_tok_s and result_b.prefill_tok_s and result_a.prefill_tok_s > 0:
            prefill_ratio = result_b.prefill_tok_s / result_a.prefill_tok_s
            logger.info(
                "  Prefill throughput ratio (B/A): %.1fx (%.1f -> %.1f tok/s)",
                prefill_ratio, result_a.prefill_tok_s, result_b.prefill_tok_s,
            )
    if result_a.success and result_c.success:
        if result_a.ttft_ms and result_c.ttft_ms and result_a.ttft_ms > 0:
            speedup_c = result_a.ttft_ms / result_c.ttft_ms
            logger.info(
                "  TTFT speedup (A->C): %.1fx (%.1fms -> %.1fms)",
                speedup_c, result_a.ttft_ms, result_c.ttft_ms,
            )

    return results, errors


# ---------------------------------------------------------------------------
# Phase 2: SSD persistence (optional)
# ---------------------------------------------------------------------------


def run_phase2(
    server_url: str, logger: logging.Logger, timeout: int = 600
) -> tuple[list[dict], list[str]]:
    """Phase 2: SSD persistence — stop server, restart, test cache reload.

    Returns (results_list, errors_list).
    """
    errors: list[str] = []
    results: list[dict] = []

    logger.info("=" * 60)
    logger.info("PHASE 2: SSD persistence (optional)")
    logger.info("=" * 60)

    # Step 1: Stop server (triggers automatic SSD save)
    logger.info("Stopping server to trigger SSD cache save...")
    stop_all_servers(TP_HOSTS)

    # Wait for clean shutdown + SSD save to complete
    logger.info("Waiting 15s for clean shutdown and SSD save...")
    time.sleep(15)

    # Step 2: Restart server with same config
    logger.info("Restarting TP server...")
    log_file = "/tmp/tp_server_phase2.log"

    if not start_tp_server(
        TP_HOSTS, MODEL, SERVER_ARGS, log_file=log_file, timeout=timeout
    ):
        msg = "Phase 2 FAILED: Could not restart TP server"
        logger.error(msg)
        errors.append(msg)
        return results, errors

    host_ip = get_host_ip(TP_HOSTS[0])
    if not wait_for_health(TP_HOSTS[0], SERVER_PORT, timeout=120):
        msg = "Phase 2 FAILED: Server health check failed after restart"
        logger.error(msg)
        errors.append(msg)
        return results, errors

    logger.info("Server restarted and healthy. Waiting 5s for SSD cache load...")
    time.sleep(5)

    # Log cache state after restart
    log_cache_stats(server_url, "after-restart", logger)

    # Step 3: Request D — should be a cache hit from SSD-restored cache
    logger.info("-" * 40)
    logger.info("Request D: SSD cache hit (same prefix, question #4)")
    logger.info("Question: %s", QUESTIONS[3][:80])

    messages_d = [
        {"role": "system", "content": LONG_PREFIX},
        {"role": "user", "content": QUESTIONS[3]},
    ]

    t_start_d = time.monotonic()
    result_d = send_chat_request(
        server_url, messages_d, max_tokens=256, temperature=0.0, timeout=600
    )
    t_elapsed_d = time.monotonic() - t_start_d

    if not result_d.success:
        msg = f"Request D FAILED: {result_d.error}"
        logger.error(msg)
        errors.append(msg)
    else:
        logger.info(
            "Request D: ttft=%.1fms, prefill=%.1f tok/s, decode=%.1f tok/s, "
            "total=%.2fs, prompt_tokens=%d, completion_tokens=%d",
            result_d.ttft_ms, result_d.prefill_tok_s, result_d.decode_tok_s,
            result_d.total_time_s, result_d.prompt_tokens, result_d.completion_tokens,
        )
        logger.debug("Request D output: %s", result_d.output_text[:200] if result_d.output_text else "")

    cache_d = log_cache_stats(server_url, "after-D", logger)
    results.append({
        "label": "D",
        "type": "SSD Load",
        "result": result_d,
        "cache_stats": cache_d.get("cache") if cache_d else None,
    })

    return results, errors


# ---------------------------------------------------------------------------
# Results formatting helpers
# ---------------------------------------------------------------------------


def build_comparison_table(
    phase1_results: list[dict],
    phase2_results: list[dict],
    logger: logging.Logger,
) -> str:
    """Build a markdown comparison table from all results."""
    all_entries = phase1_results + phase2_results

    # Find baseline TTFT (Request A) for speedup calculation
    baseline_ttft: float | None = None
    for entry in all_entries:
        if entry["label"] == "A" and entry["result"].success:
            baseline_ttft = entry["result"].ttft_ms
            break

    lines = [
        "| Request | Type | TTFT (ms) | Prefill tok/s | Decode tok/s | Total (s) | Speedup |",
        "|---------|------|-----------|---------------|--------------|-----------|---------|",
    ]

    for entry in all_entries:
        r: RequestResult = entry["result"]
        label = entry["label"]
        rtype = entry["type"]

        if not r.success:
            lines.append(
                f"| {label} | {rtype} | FAILED | - | - | - | - |"
            )
            continue

        ttft_str = f"{r.ttft_ms:.1f}" if r.ttft_ms else "-"
        prefill_str = f"{r.prefill_tok_s:.1f}" if r.prefill_tok_s else "-"
        decode_str = f"{r.decode_tok_s:.1f}" if r.decode_tok_s else "-"
        total_str = f"{r.total_time_s:.2f}" if r.total_time_s else "-"

        if baseline_ttft and r.ttft_ms and r.ttft_ms > 0:
            speedup = baseline_ttft / r.ttft_ms
            speedup_str = f"{speedup:.1f}x" if label != "A" else "1.0x (baseline)"
        else:
            speedup_str = "-"

        lines.append(
            f"| {label} | {rtype} | {ttft_str} | {prefill_str} | "
            f"{decode_str} | {total_str} | {speedup_str} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="KV cache prefix matching test on TP=2 with Kimi K2.5",
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
        "--skip-phase2",
        action="store_true",
        help="Skip Phase 2 (SSD persistence test)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Server startup timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    logger = setup_logging("tp_kv_cache")
    errors: list[str] = []
    phase1_results: list[dict] = []
    phase2_results: list[dict] = []
    server_logs: dict[str, str] = {}

    logger.info("=" * 60)
    logger.info("KV Cache Prefix Matching Test — TP=2, Kimi K2.5")
    logger.info("=" * 60)
    logger.info("Server URL: %s", args.server_url)
    logger.info("Skip server management: %s", args.skip_server_management)
    logger.info("Skip Phase 2: %s", args.skip_phase2)
    logger.info("Startup timeout: %ds", args.timeout)
    logger.info("Prefix length: ~%d characters (~%d tokens)", len(LONG_PREFIX), _APPROX_PREFIX_TOKENS)

    # -----------------------------------------------------------------------
    # Server startup
    # -----------------------------------------------------------------------
    log_file = "/tmp/tp_server.log"

    if args.skip_server_management:
        if not wait_for_health(TP_HOSTS[0], SERVER_PORT, timeout=30):
            logger.error("Server not reachable at %s:%d", TP_HOSTS[0], SERVER_PORT)
            return 1
    else:
        logger.info("Stopping all existing servers on %s...", TP_HOSTS)
        stop_all_servers(TP_HOSTS)
        time.sleep(5)

        logger.info("Starting TP=2 server with Kimi K2.5...")
        logger.debug("Server args: %s", SERVER_ARGS)

        if not start_tp_server(
            TP_HOSTS, MODEL, SERVER_ARGS, log_file=log_file, timeout=args.timeout
        ):
            msg = "FATAL: Failed to start TP server"
            logger.error(msg)
            errors.append(msg)

            server_logs = collect_server_logs(TP_HOSTS, log_file)
            config = {
                "Model": "Kimi K2.5 (612GB MoE)",
                "Mode": "TP=2 (hwstudio1 + hwstudio2, JACCL RDMA)",
                "KV Cache Quantization": "FP8",
                "Prefix Cache Size": "500 blocks",
                "Status": "FAILED — server did not start",
            }
            write_results_md("tp_kv_cache", config, [], errors, server_logs)
            return 1

        logger.info("Server started. Waiting for health check...")
        if not wait_for_health(TP_HOSTS[0], SERVER_PORT, timeout=120):
            msg = "FATAL: Server health check failed after startup"
            logger.error(msg)
            errors.append(msg)
            server_logs = collect_server_logs(TP_HOSTS, log_file)
            stop_all_servers(TP_HOSTS)
            config = {
                "Model": "Kimi K2.5 (612GB MoE)",
                "Mode": "TP=2 (hwstudio1 + hwstudio2, JACCL RDMA)",
                "Status": "FAILED — health check timeout",
            }
            write_results_md("tp_kv_cache", config, [], errors, server_logs)
            return 1

    # -----------------------------------------------------------------------
    # Phase 1: In-memory prefix caching
    # -----------------------------------------------------------------------
    try:
        phase1_results, phase1_errors = run_phase1(args.server_url, logger)
        errors.extend(phase1_errors)
    except Exception:
        msg = f"Phase 1 unexpected error:\n{traceback.format_exc()}"
        logger.error(msg)
        errors.append(msg)

    # -----------------------------------------------------------------------
    # Phase 2: SSD persistence (optional)
    # -----------------------------------------------------------------------
    if not args.skip_phase2 and not args.skip_server_management:
        try:
            phase2_results, phase2_errors = run_phase2(
                args.server_url, logger, timeout=args.timeout
            )
            errors.extend(phase2_errors)
        except Exception:
            msg = f"Phase 2 unexpected error:\n{traceback.format_exc()}"
            logger.error(msg)
            errors.append(msg)
    elif args.skip_phase2:
        logger.info("Phase 2 skipped (--skip-phase2)")
    elif args.skip_server_management:
        logger.info("Phase 2 skipped (requires server management)")

    # -----------------------------------------------------------------------
    # Collect server logs
    # -----------------------------------------------------------------------
    logger.info("Collecting server logs...")
    # Try both log files
    server_logs = collect_server_logs(TP_HOSTS, log_file)
    if phase2_results:
        phase2_logs = collect_server_logs(TP_HOSTS, "/tmp/tp_server_phase2.log")
        for host, log_content in phase2_logs.items():
            if log_content:
                server_logs[f"{host}_phase2"] = log_content

    if not args.skip_server_management:
        logger.info("Stopping servers...")
        stop_all_servers(TP_HOSTS)

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------
    config = {
        "Model": "Kimi K2.5 (612GB MoE, deepseek_v3 architecture)",
        "Mode": "TP=2 (hwstudio1 + hwstudio2, JACCL RDMA over TB5)",
        "KV Cache Quantization": "FP8 (8-bit)",
        "Prefix Cache Size": "500 blocks",
        "Prefix Length": f"~{_APPROX_PREFIX_TOKENS} tokens (~{len(LONG_PREFIX)} chars)",
        "Continuous Batching": "Enabled",
    }

    # Build results as (section_name, [RequestResult]) tuples for write_results_md
    results_for_md: list[tuple[str, list[RequestResult]]] = []
    if phase1_results:
        phase1_rr = [entry["result"] for entry in phase1_results]
        results_for_md.append(("Phase 1: In-Memory Prefix Caching", phase1_rr))
    if phase2_results:
        phase2_rr = [entry["result"] for entry in phase2_results]
        results_for_md.append(("Phase 2: SSD Persistence", phase2_rr))

    # Build extra sections as a single markdown string
    extra_md = ""

    # Comparison table
    comparison_table = build_comparison_table(phase1_results, phase2_results, logger)
    extra_md += "## Prefix Cache Speedup Comparison\n\n"
    extra_md += comparison_table + "\n\n"

    # Phase 1 analysis
    analysis_lines: list[str] = []
    baseline_entry = next((e for e in phase1_results if e["label"] == "A"), None)
    hit_entries = [e for e in phase1_results if e["label"] in ("B", "C")]

    if baseline_entry and baseline_entry["result"].success:
        baseline = baseline_entry["result"]
        for hit_entry in hit_entries:
            if hit_entry["result"].success:
                hit = hit_entry["result"]
                label = hit_entry["label"]

                if baseline.ttft_ms and hit.ttft_ms and hit.ttft_ms > 0:
                    ttft_speedup = baseline.ttft_ms / hit.ttft_ms
                    analysis_lines.append(
                        f"- **Request {label} TTFT speedup**: {ttft_speedup:.1f}x "
                        f"({baseline.ttft_ms:.1f}ms -> {hit.ttft_ms:.1f}ms)"
                    )
                if baseline.prefill_tok_s and hit.prefill_tok_s and baseline.prefill_tok_s > 0:
                    prefill_ratio = hit.prefill_tok_s / baseline.prefill_tok_s
                    analysis_lines.append(
                        f"- **Request {label} prefill throughput**: {prefill_ratio:.1f}x "
                        f"({baseline.prefill_tok_s:.1f} -> {hit.prefill_tok_s:.1f} tok/s)"
                    )

    if analysis_lines:
        extra_md += "## Phase 1 Analysis (In-Memory Cache)\n\n"
        extra_md += "\n".join(analysis_lines) + "\n\n"

    # Phase 2 analysis
    if phase2_results and baseline_entry and baseline_entry["result"].success:
        baseline = baseline_entry["result"]
        ssd_lines: list[str] = []
        for ssd_entry in phase2_results:
            if ssd_entry["result"].success:
                ssd = ssd_entry["result"]
                label = ssd_entry["label"]
                if baseline.ttft_ms and ssd.ttft_ms and ssd.ttft_ms > 0:
                    ssd_speedup = baseline.ttft_ms / ssd.ttft_ms
                    ssd_lines.append(
                        f"- **Request {label} (SSD) TTFT speedup**: {ssd_speedup:.1f}x "
                        f"({baseline.ttft_ms:.1f}ms -> {ssd.ttft_ms:.1f}ms)"
                    )
        if ssd_lines:
            extra_md += "## Phase 2 Analysis (SSD Persistence)\n\n"
            extra_md += "\n".join(ssd_lines) + "\n\n"
        elif not phase2_results[0]["result"].success:
            extra_md += "## Phase 2 Analysis (SSD Persistence)\n\n"
            extra_md += "Phase 2 request failed. SSD persistence could not be verified.\n\n"
    elif not phase2_results and not args.skip_phase2 and not args.skip_server_management:
        extra_md += "## Phase 2 Analysis (SSD Persistence)\n\n"
        extra_md += "Phase 2 was not executed due to earlier errors.\n\n"

    # Cache stats across requests
    cache_stats_lines: list[str] = []
    for entry in phase1_results + phase2_results:
        cs = entry.get("cache_stats")
        if cs:
            cache_stats_lines.append(
                f"**After Request {entry['label']}**:\n```json\n{json.dumps(cs, indent=2)}\n```\n"
            )
    if cache_stats_lines:
        extra_md += "## Cache Stats (from /v1/status)\n\n"
        extra_md += "\n".join(cache_stats_lines) + "\n\n"

    output_path = write_results_md(
        "tp_kv_cache", config, results_for_md, errors, server_logs, extra_md
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    all_entries = phase1_results + phase2_results
    total_tests = len(all_entries)
    passed = sum(1 for e in all_entries if e["result"].success)
    failed = total_tests - passed

    logger.info("=" * 60)
    logger.info("KV Cache Test Summary")
    logger.info("  Total requests: %d", total_tests)
    logger.info("  Passed: %d", passed)
    logger.info("  Failed: %d", failed)
    logger.info("  Errors: %d", len(errors))
    if output_path:
        logger.info("  Results: %s", output_path)
    logger.info("=" * 60)

    return 1 if failed > 0 or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
