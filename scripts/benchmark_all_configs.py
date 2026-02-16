#!/opt/homebrew/bin/python3.14
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive single-node benchmark for ALL vllm-mlx configurations.

Tests every combination of server config (simple, continuous-batching,
ngram speculative decoding, draft-model speculative decoding) against
four prompt lengths (short / medium / long / very_long).

Designed for Mac Studio M4 Ultra 512GB.  Uses ONLY Python stdlib
(urllib, json, subprocess, signal, time, argparse) -- no pip deps.

Usage:
    # Run ALL configs
    python3.14 scripts/benchmark_all_configs.py

    # Run a single config
    python3.14 scripts/benchmark_all_configs.py --config-id ngram-k3

    # Run a single prompt level
    python3.14 scripts/benchmark_all_configs.py --prompt-level short

    # External server (don't start/stop automatically)
    python3.14 scripts/benchmark_all_configs.py --no-server-management
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

# ============================================================================
# Constants
# ============================================================================

PYTHON = "/opt/homebrew/bin/python3.14"
SERVER_MODULE = "vllm_mlx.server"

MAIN_MODEL = os.path.expanduser("~/models/Moonlight-16B-A3B-Instruct-4-bit/")
DRAFT_MODEL = os.path.expanduser("~/models/Kimi-K2-Instruct-DRAFT-0.6B-MLX/")

DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_MAX_TOKENS = 256
DEFAULT_OUTPUT_JSON = "/tmp/benchmark_results.json"
DEFAULT_WARMUP = 1

HEALTH_POLL_INTERVAL = 3  # seconds between health-check polls
HEALTH_TIMEOUT = 180  # max seconds to wait for server readiness
REQUEST_TIMEOUT = 120  # per-request timeout
INTER_CONFIG_DELAY = 10  # seconds between configs for memory release
SIGTERM_GRACE = 10  # seconds to wait after SIGTERM before SIGKILL

# ============================================================================
# Prompts
# ============================================================================

PROMPTS: dict[str, str] = {
    "short": "Write a haiku about the ocean.",
    "medium": (
        "Explain the theory of general relativity, including its key predictions "
        "such as gravitational time dilation, the bending of light around massive "
        "objects, and gravitational waves. How have these predictions been "
        "experimentally verified?"
    ),
    "long": (
        "Quantum computing represents a paradigm shift in computational power. "
        "Unlike classical computers that use bits representing 0 or 1, quantum "
        "computers use qubits that can exist in superposition states. This allows "
        "quantum computers to explore many possible solutions simultaneously "
        "through quantum parallelism. Key quantum phenomena include entanglement, "
        "where qubits become correlated regardless of distance, and interference, "
        "which amplifies correct solutions while canceling incorrect ones. Major "
        "quantum algorithms include Shor's algorithm for factoring large numbers "
        "(threatening RSA encryption), Grover's algorithm for searching unsorted "
        "databases with quadratic speedup, and variational quantum eigensolvers "
        "for chemistry simulations. Current challenges include decoherence, error "
        "correction overhead, and the difficulty of scaling up qubit counts while "
        "maintaining coherence times. Explain the current state of quantum "
        "computing hardware, comparing superconducting qubits, trapped ions, and "
        "photonic approaches."
    ),
    "very_long": (
        "Climate change represents one of the most complex challenges facing "
        "humanity in the 21st century. Global average temperatures have risen "
        "approximately 1.1 degrees Celsius above pre-industrial levels, with the "
        "last decade being the warmest on record. The Intergovernmental Panel on "
        "Climate Change (IPCC) projects that without significant intervention, "
        "temperatures could rise by 2.5 to 4.5 degrees Celsius by 2100, leading "
        "to catastrophic consequences including sea level rise of up to one meter, "
        "more frequent extreme weather events, and disruption of agricultural "
        "systems that feed billions.\n\n"
        "Renewable energy deployment has accelerated dramatically: solar photovoltaic "
        "costs have fallen 89% since 2010, wind energy costs dropped 70%, and "
        "battery storage costs declined 97% over the past three decades. In 2023, "
        "renewable sources accounted for 30% of global electricity generation, "
        "with solar and wind together surpassing 12% for the first time. China "
        "alone installed more solar capacity in 2023 than the entire world did in "
        "2022. However, electricity represents only about 20% of total energy "
        "consumption; hard-to-abate sectors like heavy industry (steel, cement, "
        "chemicals), long-haul transportation (shipping, aviation), and agriculture "
        "remain heavily dependent on fossil fuels.\n\n"
        "Economic analyses suggest the transition to net-zero emissions by 2050 "
        "would require approximately $4 trillion in annual clean energy investment "
        "by 2030, roughly triple current levels. The International Energy Agency "
        "estimates that every dollar invested in clean energy generates $1.50 in "
        "economic output through manufacturing, installation, and maintenance jobs. "
        "Carbon pricing mechanisms now cover approximately 23% of global emissions, "
        "with the EU Emissions Trading System reaching prices of 90 euros per ton "
        "of CO2 in 2023. However, carbon border adjustment mechanisms and their "
        "impact on international trade remain contentious.\n\n"
        "Emerging technologies offer additional pathways: green hydrogen produced "
        "via electrolysis could decarbonize industrial processes, direct air capture "
        "facilities could remove CO2 at scale (though current costs exceed $400 per "
        "ton), advanced nuclear designs including small modular reactors promise "
        "reliable baseload power, and enhanced geothermal systems could unlock vast "
        "underground heat resources. Meanwhile, nature-based solutions such as "
        "reforestation, wetland restoration, and improved soil management could "
        "sequester 5-10 gigatons of CO2 annually.\n\n"
        "Based on the above analysis, provide a comprehensive evaluation of the "
        "most promising pathways to achieving net-zero emissions by 2050, "
        "considering technological feasibility, economic viability, and political "
        "constraints."
    ),
}

PROMPT_LEVELS = ["short", "medium", "long", "very_long"]

# ============================================================================
# Configuration definitions
# ============================================================================


def _build_configs() -> list[dict[str, Any]]:
    """Return the ordered list of benchmark configurations."""
    configs: list[dict[str, Any]] = []

    # 1. simple-baseline
    configs.append({
        "config_id": "simple-baseline",
        "continuous_batching": False,
        "spec_method": None,
        "draft_model": None,
        "k": None,
    })

    # 2. cb-baseline
    configs.append({
        "config_id": "cb-baseline",
        "continuous_batching": True,
        "spec_method": None,
        "draft_model": None,
        "k": None,
    })

    # 3-7. ngram-k1 through ngram-k5
    for k in range(1, 6):
        configs.append({
            "config_id": f"ngram-k{k}",
            "continuous_batching": True,
            "spec_method": "ngram",
            "draft_model": None,
            "k": k,
        })

    # 8-12. draft-k1 through draft-k5
    for k in range(1, 6):
        configs.append({
            "config_id": f"draft-k{k}",
            "continuous_batching": True,
            "spec_method": "draft_model",
            "draft_model": DRAFT_MODEL,
            "k": k,
        })

    # 13. draft-k8
    configs.append({
        "config_id": "draft-k8",
        "continuous_batching": True,
        "spec_method": "draft_model",
        "draft_model": DRAFT_MODEL,
        "k": 8,
    })

    return configs


ALL_CONFIGS = _build_configs()

# ============================================================================
# Utility helpers
# ============================================================================


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _word_count(text: str) -> int:
    return len(text.split())


def _kill_port(port: int) -> None:
    """Kill any process listening on the given TCP port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        for pid_str in pids:
            pid = int(pid_str)
            _log(f"  Killing existing process {pid} on port {port}")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if pids:
            time.sleep(2)
    except Exception:
        pass


def _http_get(url: str, timeout: float = 10.0) -> dict | None:
    """GET a JSON endpoint. Returns parsed dict or None on failure."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _http_post_stream(url: str, body: dict, timeout: float = REQUEST_TIMEOUT):
    """
    POST JSON and return the raw HTTPResponse for SSE streaming.
    Caller must close the response.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


# ============================================================================
# Server lifecycle
# ============================================================================


def build_server_cmd(config: dict[str, Any], port: int) -> list[str]:
    """Build the CLI command to start the server for a given config."""
    cmd = [PYTHON, "-m", SERVER_MODULE, "--model", MAIN_MODEL, "--port", str(port)]

    if config["continuous_batching"]:
        cmd.append("--continuous-batching")

    if config["spec_method"]:
        cmd.extend(["--speculative-method", config["spec_method"]])
        cmd.extend(["--num-speculative-tokens", str(config["k"])])

    if config["draft_model"]:
        cmd.extend(["--draft-model", config["draft_model"]])

    return cmd


def start_server(config: dict[str, Any], port: int) -> subprocess.Popen | None:
    """Start the vllm-mlx server and return the Popen handle."""
    config_id = config["config_id"]
    log_path = f"/tmp/bench_server_{config_id}.log"

    _kill_port(port)

    cmd = build_server_cmd(config, port)
    _log(f"  Starting server: {' '.join(cmd)}")
    _log(f"  Server log: {log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_file = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.path.expanduser("~"),
            env=env,
            preexec_fn=os.setsid,  # new process group for clean kill
        )
    except Exception as e:
        log_file.close()
        _log(f"  ERROR: Failed to start server: {e}")
        return None

    return proc


def wait_for_health(server_url: str, timeout: float = HEALTH_TIMEOUT) -> bool:
    """Poll /health until it returns 200 or timeout expires."""
    url = f"{server_url}/health"
    deadline = time.monotonic() + timeout
    _log(f"  Waiting for server health ({url}), timeout={timeout}s ...")

    while time.monotonic() < deadline:
        result = _http_get(url, timeout=5.0)
        if result is not None and result.get("status") == "healthy":
            _log("  Server is healthy!")
            return True
        time.sleep(HEALTH_POLL_INTERVAL)

    _log("  ERROR: Server health-check timed out")
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server: SIGTERM, wait, then SIGKILL if needed."""
    if proc.poll() is not None:
        return  # already exited

    _log(f"  Sending SIGTERM to server (pid={proc.pid}) ...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=SIGTERM_GRACE)
        _log("  Server stopped gracefully.")
        return
    except subprocess.TimeoutExpired:
        pass

    _log(f"  Server did not stop after {SIGTERM_GRACE}s, sending SIGKILL ...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)
    except Exception:
        pass
    _log("  Server killed.")


def is_server_alive(proc: subprocess.Popen) -> bool:
    """Check whether the server process is still running."""
    return proc.poll() is None


# ============================================================================
# Benchmark measurement
# ============================================================================


def run_prompt(
    server_url: str,
    prompt: str,
    max_tokens: int,
    level: str,
) -> dict[str, Any]:
    """
    Send a single streaming chat completion and measure metrics.

    Returns a dict matching the per-prompt results schema.
    """
    url = f"{server_url}/v1/chat/completions"
    body = {
        "model": "benchmark",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    prompt_words = _word_count(prompt)
    result: dict[str, Any] = {
        "level": level,
        "prompt_words": prompt_words,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "ttft_s": 0.0,
        "prefill_toks": 0.0,
        "decode_time_s": 0.0,
        "decode_toks": 0.0,
        "total_time_s": 0.0,
        "total_toks": 0.0,
        "status": "ok",
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        resp = _http_post_stream(url, body)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = f"Request failed: {e}"
        return result

    ttft: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    total_completion_tokens = 0

    try:
        # Read line-by-line directly from the HTTP response to get
        # accurate per-chunk timestamps for TTFT measurement.
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break

            chunk_time = time.perf_counter()

            try:
                line = raw_line.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if not line:
                continue

            if line == "data: [DONE]":
                break

            if not line.startswith("data: "):
                continue

            json_str = line[6:]  # strip "data: " prefix
            try:
                event = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            # Check for usage-only chunk (empty choices, has usage)
            choices = event.get("choices", [])
            usage = event.get("usage")

            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                total_completion_tokens = usage.get(
                    "completion_tokens", total_completion_tokens
                )

            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")

            # TTFT: first chunk with non-empty content
            if content and ttft is None:
                ttft = chunk_time - t_start

            if content:
                completion_tokens += 1  # approximate: count chunks as tokens

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = f"Stream error: {e}"
        return result
    finally:
        try:
            resp.close()
        except Exception:
            pass

    t_end = time.perf_counter()
    total_time = t_end - t_start

    # Prefer server-reported token counts
    if total_completion_tokens > 0:
        completion_tokens = total_completion_tokens

    if ttft is None:
        ttft = total_time  # fallback: no tokens streamed

    decode_time = max(total_time - ttft, 0.001)

    result["prompt_tokens"] = prompt_tokens
    result["completion_tokens"] = completion_tokens
    result["ttft_s"] = round(ttft, 4)
    result["prefill_toks"] = (
        round(prompt_tokens / ttft, 1) if ttft > 0 and prompt_tokens > 0 else 0.0
    )
    result["decode_time_s"] = round(decode_time, 4)
    result["decode_toks"] = (
        round(completion_tokens / decode_time, 1)
        if decode_time > 0 and completion_tokens > 0
        else 0.0
    )
    result["total_time_s"] = round(total_time, 4)
    total_tokens = prompt_tokens + completion_tokens
    result["total_toks"] = (
        round(total_tokens / total_time, 1)
        if total_time > 0 and total_tokens > 0
        else 0.0
    )

    return result


def get_spec_decode_stats(server_url: str) -> dict[str, Any] | None:
    """Query /v1/status and extract spec_decode stats."""
    result = _http_get(f"{server_url}/v1/status", timeout=10.0)
    if result is None:
        return None
    return result.get("spec_decode")


# ============================================================================
# Main benchmark loop
# ============================================================================


def run_config(
    config: dict[str, Any],
    config_idx: int,
    total_configs: int,
    server_url: str,
    max_tokens: int,
    warmup: int,
    prompt_levels: list[str],
    manage_server: bool,
    port: int,
) -> dict[str, Any]:
    """Run benchmarks for a single configuration. Returns a result dict."""
    config_id = config["config_id"]
    _log(f"\n{'='*60}")
    _log(f"Config: {config_id} ({config_idx}/{total_configs})")
    _log(f"{'='*60}")

    entry: dict[str, Any] = {
        "config_id": config_id,
        "config": {
            "continuous_batching": config["continuous_batching"],
            "spec_method": config["spec_method"],
            "draft_model": os.path.basename(config["draft_model"]) if config["draft_model"] else None,
            "k": config["k"],
        },
        "status": "ok",
        "spec_decode_stats": None,
        "prompts": [],
    }

    proc: subprocess.Popen | None = None

    if manage_server:
        proc = start_server(config, port)
        if proc is None:
            entry["status"] = "FAILED"
            entry["prompts"] = [
                {"level": lv, "status": "FAILED", "error": "Server failed to start"}
                for lv in prompt_levels
            ]
            return entry

        if not wait_for_health(server_url, timeout=HEALTH_TIMEOUT):
            entry["status"] = "FAILED"
            entry["prompts"] = [
                {"level": lv, "status": "FAILED", "error": "Server health timeout"}
                for lv in prompt_levels
            ]
            stop_server(proc)
            return entry

    # --- Warmup ---
    if warmup > 0:
        _log(f"  Running {warmup} warmup request(s) ...")
        for i in range(warmup):
            warmup_result = run_prompt(
                server_url, PROMPTS["short"], max_tokens, "warmup"
            )
            if warmup_result["status"] != "ok":
                _log(f"  WARNING: warmup request {i+1} failed: {warmup_result.get('error')}")
            else:
                _log(
                    f"  Warmup {i+1}: {warmup_result['completion_tokens']} tokens, "
                    f"TTFT={warmup_result['ttft_s']:.3f}s"
                )

    # --- Benchmark each prompt level ---
    for level in prompt_levels:
        prompt_text = PROMPTS[level]
        _log(f"  Prompt: {level} ({_word_count(prompt_text)} words) ...")

        # Check server is still alive (if managed)
        if manage_server and proc and not is_server_alive(proc):
            _log("  ERROR: Server crashed!")
            entry["status"] = "FAILED"
            # Mark remaining prompts as failed
            remaining = prompt_levels[prompt_levels.index(level):]
            for lv in remaining:
                entry["prompts"].append({
                    "level": lv,
                    "prompt_words": _word_count(PROMPTS[lv]),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "ttft_s": 0,
                    "prefill_toks": 0,
                    "decode_time_s": 0,
                    "decode_toks": 0,
                    "total_time_s": 0,
                    "total_toks": 0,
                    "status": "FAILED",
                    "error": "Server crashed during benchmark",
                })
            break

        result = run_prompt(server_url, prompt_text, max_tokens, level)
        entry["prompts"].append(result)

        if result["status"] == "ok":
            _log(
                f"    TTFT={result['ttft_s']:.3f}s "
                f"prefill={result['prefill_toks']:.1f} t/s | "
                f"decode={result['decode_time_s']:.3f}s "
                f"{result['decode_toks']:.1f} t/s | "
                f"total={result['total_toks']:.1f} t/s "
                f"({result['completion_tokens']} tokens)"
            )
        else:
            _log(f"    FAILED: {result.get('error', 'unknown')}")

    # --- Spec decode stats ---
    spec_stats = get_spec_decode_stats(server_url)
    entry["spec_decode_stats"] = spec_stats

    # --- Print summary ---
    _print_config_summary(config_id, config_idx, total_configs, entry)

    # --- Stop server ---
    if manage_server and proc:
        stop_server(proc)

    return entry


def _print_config_summary(
    config_id: str,
    config_idx: int,
    total_configs: int,
    entry: dict[str, Any],
) -> None:
    """Print a formatted summary for one config."""
    print(f"\n=== Config: {config_id} ({config_idx}/{total_configs}) ===")

    for p in entry["prompts"]:
        level = p.get("level", "?")
        if p.get("status") != "ok":
            print(f"  {level:9s}: FAILED - {p.get('error', 'unknown')}")
            continue
        print(
            f"  {level:9s}: "
            f"prefill {p['ttft_s']:.2f}s ({p['prefill_toks']:.1f} t/s) | "
            f"decode {p['decode_time_s']:.1f}s ({p['decode_toks']:.1f} t/s) | "
            f"total {p['total_toks']:.1f} t/s"
        )

    sd = entry.get("spec_decode_stats")
    if sd:
        ar = sd.get("acceptance_rate")
        if ar is not None:
            print(f"  spec_decode: acceptance_rate={ar:.4f}")
        else:
            print(f"  spec_decode: {sd}")
    print()


def save_results(
    results: list[dict[str, Any]],
    output_path: str,
    machine: str,
) -> None:
    """Save (or update) the results JSON file."""
    doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "model": os.path.basename(MAIN_MODEL.rstrip("/")),
        "draft_model": os.path.basename(DRAFT_MODEL.rstrip("/")),
        "results": results,
    }
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp_path, output_path)


def get_machine_name() -> str:
    """Return the hostname of this machine."""
    return socket.gethostname()


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for all vllm-mlx configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-id",
        type=str,
        default=None,
        help="Run only this config (e.g. 'ngram-k3'). Default: run ALL.",
    )
    parser.add_argument(
        "--prompt-level",
        type=str,
        default=None,
        choices=PROMPT_LEVELS,
        help="Run only this prompt level. Default: run ALL.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help=f"Server base URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per completion (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Path to write JSON results (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup requests before measuring (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--no-server-management",
        action="store_true",
        help="Don't start/stop server (assume it's running externally)",
    )

    args = parser.parse_args()

    # Resolve configs
    if args.config_id:
        configs = [c for c in ALL_CONFIGS if c["config_id"] == args.config_id]
        if not configs:
            valid = ", ".join(c["config_id"] for c in ALL_CONFIGS)
            print(f"ERROR: Unknown config-id '{args.config_id}'", file=sys.stderr)
            print(f"Valid config IDs: {valid}", file=sys.stderr)
            sys.exit(1)
    else:
        configs = ALL_CONFIGS

    # Resolve prompt levels
    if args.prompt_level:
        prompt_levels = [args.prompt_level]
    else:
        prompt_levels = list(PROMPT_LEVELS)

    manage_server = not args.no_server_management

    # Parse port from server URL
    from urllib.parse import urlparse
    parsed = urlparse(args.server_url)
    port = parsed.port or 8000

    # Banner
    machine = get_machine_name()
    _log("=" * 60)
    _log("vllm-mlx Comprehensive Benchmark")
    _log("=" * 60)
    _log(f"  Machine:       {machine}")
    _log(f"  Model:         {MAIN_MODEL}")
    _log(f"  Draft model:   {DRAFT_MODEL}")
    _log(f"  Server URL:    {args.server_url}")
    _log(f"  Max tokens:    {args.max_tokens}")
    _log(f"  Warmup:        {args.warmup}")
    _log(f"  Configs:       {len(configs)}")
    _log(f"  Prompt levels: {prompt_levels}")
    _log(f"  Manage server: {manage_server}")
    _log(f"  Output JSON:   {args.output_json}")
    _log("=" * 60)

    all_results: list[dict[str, Any]] = []
    total = len(configs)

    for idx, config in enumerate(configs, 1):
        entry = run_config(
            config=config,
            config_idx=idx,
            total_configs=total,
            server_url=args.server_url,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
            prompt_levels=prompt_levels,
            manage_server=manage_server,
            port=port,
        )
        all_results.append(entry)

        # Save incrementally
        save_results(all_results, args.output_json, machine)
        _log(f"  Results saved to {args.output_json}")

        # Inter-config delay (if more configs remain)
        if idx < total and manage_server:
            _log(f"  Waiting {INTER_CONFIG_DELAY}s for memory cleanup ...")
            time.sleep(INTER_CONFIG_DELAY)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    # Build a comparison table
    _print_final_table(all_results, prompt_levels)

    print(f"\nResults saved to: {args.output_json}")
    _log("Benchmark complete.")


def _print_final_table(
    results: list[dict[str, Any]],
    prompt_levels: list[str],
) -> None:
    """Print a compact comparison table of all configs."""
    # Header
    header_parts = [f"{'Config':20s}"]
    for lv in prompt_levels:
        header_parts.append(f"{'TTFT':>6s} {'Pf t/s':>7s} {'Dc t/s':>7s}")
    if any(r.get("spec_decode_stats") for r in results):
        header_parts.append(f"{'AccRate':>8s}")
    print("  ".join(header_parts))
    print("-" * len("  ".join(header_parts)))

    for entry in results:
        cid = entry["config_id"]
        if entry["status"] != "ok":
            print(f"{cid:20s}  FAILED")
            continue

        parts = [f"{cid:20s}"]
        prompts_by_level = {p["level"]: p for p in entry.get("prompts", [])}

        for lv in prompt_levels:
            p = prompts_by_level.get(lv)
            if p and p.get("status") == "ok":
                parts.append(
                    f"{p['ttft_s']:6.2f} {p['prefill_toks']:7.1f} {p['decode_toks']:7.1f}"
                )
            else:
                parts.append(f"{'FAIL':>6s} {'FAIL':>7s} {'FAIL':>7s}")

        sd = entry.get("spec_decode_stats")
        if any(r.get("spec_decode_stats") for r in results):
            if sd and sd.get("acceptance_rate") is not None:
                parts.append(f"{sd['acceptance_rate']:8.4f}")
            else:
                parts.append(f"{'N/A':>8s}")

        print("  ".join(parts))


if __name__ == "__main__":
    main()
