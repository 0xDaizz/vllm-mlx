#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for vllm-mlx distributed feature verification tests.

Provides SSH helpers, server lifecycle management, HTTP request helpers,
metric collection, and Markdown report generation. All functions use
only the Python standard library (no pip dependencies).

These test scripts run from a control MacBook and manage servers on
remote Mac Studio nodes via SSH.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYTHON = "/opt/homebrew/bin/python3.14"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
DEFAULT_SERVER_HOST = "hwstudio1"
DEFAULT_SERVER_URL = "http://100.120.177.62:8000"
TP_HOSTS = ["hwstudio1", "hwstudio2"]
LOG_DIR = "/tmp"
RESULTS_DIR = "docs/test-results"

_HOST_IP_MAP = {
    "hwstudio1": "100.120.177.62",
    "hwstudio2": "100.122.158.16",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Result and metrics from a single chat completion request."""

    success: bool
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0  # time to first token (ms)
    total_time_s: float = 0.0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    output_text: str = ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(test_name: str, log_dir: str = "logs") -> logging.Logger:
    """Set up DEBUG-level logging to both console and file.

    Creates ``{log_dir}/{test_name}_{timestamp}.log``.

    Returns:
        The root :class:`logging.Logger`.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{test_name}_{timestamp}.log")

    # Reset root logger handlers
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)

    # File handler: DEBUG level, full format
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler: INFO level, compact format
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(ch)

    root.info("Logging initialised: file=%s (DEBUG), console (INFO)", log_file)
    return root


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------


def ssh_run(
    host: str,
    cmd: str,
    timeout: int = 30,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess:
    """Run *cmd* on *host* via SSH.

    Logs the command and its output at DEBUG level when *logger* is provided.
    """
    log = logger or logging.getLogger(__name__)
    # Validate host to prevent SSH option injection
    if host.startswith("-") or re.search(r'[\s;|&`$(){}]', host):
        raise ValueError(f"Invalid SSH host (contains dangerous characters): {host!r}")
    ssh_cmd = ["ssh"] + SSH_OPTS + [host, cmd]
    log.debug("ssh_run: host=%s cmd=%s timeout=%d", host, cmd, timeout)
    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.debug("ssh_run: TIMEOUT after %ds on %s", timeout, host)
        return subprocess.CompletedProcess(ssh_cmd, -1, "", f"timeout after {timeout}s")

    log.debug(
        "ssh_run: rc=%d stdout=%r stderr=%r",
        result.returncode,
        result.stdout[:500] if result.stdout else "",
        result.stderr[:500] if result.stderr else "",
    )
    return result


def get_host_ip(host: str) -> str:
    """Map a hostname to its Tailscale IP address."""
    return _HOST_IP_MAP.get(host, host)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def stop_all_servers(hosts: list[str]) -> None:
    """Stop any running vllm_mlx processes on all *hosts*.

    Uses SIGTERM first (graceful), waits up to 15s, then SIGKILL as last resort.
    IMPORTANT: --force (SIGKILL) causes Metal wired memory leak requiring reboot.
    """
    log = logging.getLogger(__name__)
    log.info("Stopping all servers on %s ...", hosts)

    # Step 1: SIGTERM on all hosts (graceful shutdown)
    for host in hosts:
        log.debug("Sending SIGTERM on %s", host)
        ssh_run(host, "pkill -15 -f 'vllm_mlx' 2>/dev/null || true", timeout=10, logger=log)

    # Step 2: Wait for graceful shutdown (up to 15s)
    for attempt in range(5):
        time.sleep(3)
        all_stopped = True
        for host in hosts:
            check = ssh_run(host, "pgrep -f vllm_mlx 2>/dev/null", timeout=10, logger=log)
            if check.returncode == 0 and check.stdout.strip():
                all_stopped = False
                break
        if all_stopped:
            log.info("All servers stopped gracefully after %ds", (attempt + 1) * 3)
            return

    # Step 3: SIGKILL as last resort (will leak Metal memory)
    log.warning("Processes still alive after SIGTERM, sending SIGKILL (will leak wired memory)")
    for host in hosts:
        ssh_run(host, "pkill -9 -f 'vllm_mlx' 2>/dev/null || true", timeout=10, logger=log)

    time.sleep(2)
    for host in hosts:
        check = ssh_run(host, "pgrep -f vllm_mlx 2>/dev/null", timeout=10, logger=log)
        if check.returncode == 0 and check.stdout.strip():
            remaining_pids = check.stdout.strip()
            log.warning(
                "vllm_mlx processes still running on %s after SIGKILL: PIDs=%s",
                host, remaining_pids.replace("\n", ", "),
            )
        else:
            log.debug("Confirmed: no vllm_mlx processes on %s", host)

    log.info("Server stop commands sent to all hosts")


def start_tp_server(
    hosts: list[str],
    model: str,
    server_args: list[str],
    log_file: str = "/tmp/tp_server.log",
    timeout: int = 600,
    env_vars: list[str] | None = None,
) -> bool:
    """Start a TP server via ``scripts/tp_start.py`` (local orchestration).

    ``tp_start.py`` performs manual per-rank orchestration: it SSHes into
    each node, sets ``MLX_RANK``, ``MLX_JACCL_COORDINATOR``, and
    ``MLX_IBV_DEVICES`` environment variables, and starts each rank
    individually with ``nohup``.  This avoids the broken
    ``mlx._distributed_utils.launch`` path which overwrites the IBV
    config and exits without properly starting workers.

    Returns:
        ``True`` if the server started and passed the health check.
    """
    log = logging.getLogger(__name__)
    rank0 = hosts[0]

    # Expand ~ in model path and server_args
    model = os.path.expanduser(model)
    server_args = [os.path.expanduser(a) if a.startswith("~") else a for a in server_args]

    # Build tp_start.py command (runs locally on this macbook)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tp_start_script = os.path.join(script_dir, "scripts", "tp_start.py")
    hostfile_path = os.path.join(script_dir, "configs", "hostfile_2node.json")
    log_dir = os.path.dirname(log_file) or "/tmp"

    cmd = [
        PYTHON, tp_start_script,
        "--backend", "jaccl",
        "--hostfile", hostfile_path,
        "--timeout", str(timeout),
        "--log-dir", log_dir,
        "--skip-checks",
    ]

    # Pass env vars via --env flags
    if env_vars:
        for ev in env_vars:
            if re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', ev):
                cmd.extend(["--env", ev])
            else:
                log.warning("Skipping invalid env var: %r", ev)

    # Append server args after "--"
    all_server_args = ["--model", model] + server_args
    cmd.append("--")
    cmd.extend(all_server_args)

    log.info("Starting TP server via tp_start.py with model=%s", model)
    log.debug("tp_start.py command: %s", " ".join(shlex.quote(c) for c in cmd))

    # Run tp_start.py locally -- it handles SSH to nodes internally.
    # tp_start.py includes its own wait_for_startup + health_check,
    # so we give it the full timeout.
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 120,  # extra margin over the internal timeout
        )
    except subprocess.TimeoutExpired:
        log.error("tp_start.py timed out after %ds", timeout + 120)
        return False

    # Log stdout/stderr from tp_start.py
    if result.stdout:
        for line in result.stdout.splitlines():
            log.info("tp_start.py: %s", line)
    if result.stderr:
        for line in result.stderr.splitlines():
            log.debug("tp_start.py stderr: %s", line)

    if result.returncode != 0:
        log.error("tp_start.py failed with rc=%d", result.returncode)
        return False

    log.info("tp_start.py completed successfully, verifying health...")

    # Extract port from server_args
    port = 8000
    for i, arg in enumerate(server_args):
        if arg == "--port" and i + 1 < len(server_args):
            try:
                port = int(server_args[i + 1])
            except ValueError:
                pass

    # tp_start.py already does its own health check, but we do a final
    # verification from this machine to confirm end-to-end connectivity.
    return wait_for_health(rank0, port=port, timeout=60)


def start_single_server(
    host: str,
    model: str,
    server_args: list[str],
    log_file: str = "/tmp/single_server.log",
    timeout: int = 300,
) -> bool:
    """Start a single-node server with DEBUG logging on *host*.

    Similar to :func:`start_tp_server` but launches
    ``vllm_mlx.server`` directly (no distributed launcher).

    Returns:
        ``True`` if the server started and passed the health check.
    """
    log = logging.getLogger(__name__)

    # Expand ~ in model path and server_args before they get shlex-quoted
    model = os.path.expanduser(model)
    server_args = [os.path.expanduser(a) if a.startswith("~") else a for a in server_args]

    # Build the wrapper that enables DEBUG logging
    wrapper_script = (
        'import logging, sys\n'
        'logging.basicConfig(\n'
        '    level=logging.DEBUG,\n'
        '    format="%(asctime)s %(levelname)s %(name)s: %(message)s",\n'
        '    force=True,\n'
        ')\n'
        'from vllm_mlx.server import main\n'
        'main()\n'
    )
    log.debug("Writing single-server debug wrapper to %s", host)
    ssh_run(
        host,
        f"cat > /tmp/debug_single_server.py << 'WRAPPER_EOF'\n{wrapper_script}WRAPPER_EOF",
        timeout=10, logger=log,
    )

    # Build command (use the debug wrapper for DEBUG logging)
    all_args = ["--model", model] + server_args
    args_str = " ".join(shlex.quote(a) for a in all_args)
    remote_cmd = (
        f"cd /Users/hw/vllm-mlx && "
        f"nohup {PYTHON} /tmp/debug_single_server.py {args_str} "
        f"> {shlex.quote(log_file)} 2>&1 &"
    )

    log.info("Starting single-node server on %s with model=%s", host, model)
    log.debug("Remote command: %s", remote_cmd)

    launch_result = ssh_run(host, remote_cmd, timeout=30, logger=log)
    if launch_result.returncode != 0 and launch_result.returncode != -1:
        log.error("Failed to launch single server: rc=%d stderr=%s",
                  launch_result.returncode, launch_result.stderr)
        return False

    log.info("Single server launch command sent, waiting for health check...")
    return wait_for_health(host, port=8000, timeout=timeout)


def wait_for_health(host: str, port: int = 8000, timeout: int = 600) -> bool:
    """Poll ``GET /v1/models`` until the server responds or *timeout* expires.

    Uses the Tailscale IP for *host*.  Prints progress every 30 seconds.

    Returns:
        ``True`` if the server became healthy within the timeout.
    """
    log = logging.getLogger(__name__)
    ip = get_host_ip(host)
    url = f"http://{ip}:{port}/v1/models"
    log.info("Waiting for health: %s (timeout=%ds)", url, timeout)

    start = time.monotonic()
    last_report = start
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            log.error("Health check timed out after %ds", timeout)
            return False

        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read(512).decode("utf-8", errors="replace")
                log.debug("Health check response: status=%d body=%s",
                          resp.status, body[:200])
                if resp.status == 200:
                    log.info("Server is healthy after %.1fs", elapsed)
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            log.debug("Health check attempt failed: %s", exc)

        now = time.monotonic()
        if now - last_report >= 30:
            log.info("Still waiting for server... (%.0fs / %ds)", elapsed, timeout)
            last_report = now

        time.sleep(3)


# ---------------------------------------------------------------------------
# HTTP request helpers
# ---------------------------------------------------------------------------


def send_chat_request(
    server_url: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    model: str = "default",
    timeout: int = 300,
) -> RequestResult:
    """Send a streaming chat completion request and measure metrics.

    Uses only :mod:`urllib` (stdlib).  Parses SSE chunks to measure
    time-to-first-token, decode throughput, and prefill throughput.

    Returns:
        A :class:`RequestResult` with all metrics populated.
    """
    log = logging.getLogger(__name__)
    url = f"{server_url.rstrip('/')}/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")

    log.debug("send_chat_request: url=%s max_tokens=%d temperature=%.2f",
              url, max_tokens, temperature)
    log.debug("send_chat_request: messages=%s", json.dumps(messages)[:500])

    prompt_tokens = 0
    completion_tokens = 0
    content_chunk_count = 0  # fallback counter for completion tokens
    output_parts: list[str] = []
    t_first_token: float | None = None

    t_start = time.perf_counter()

    try:
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line_bytes in resp:
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    log.debug("Skipping non-JSON SSE chunk: %s", data_str[:100])
                    continue

                # Extract usage from final chunk
                usage = data.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                # Detect content delta -> first token
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        output_parts.append(content)
                        content_chunk_count += 1

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        t_end = time.perf_counter()
        error_msg = f"{type(exc).__name__}: {exc}"
        log.error("Request failed: %s", error_msg)
        return RequestResult(
            success=False,
            error=error_msg,
            total_time_s=t_end - t_start,
        )

    t_end = time.perf_counter()
    if t_first_token is None:
        t_first_token = t_end

    total_time = t_end - t_start
    ttft_s = t_first_token - t_start
    ttft_ms = ttft_s * 1000.0
    decode_time = t_end - t_first_token
    output_text = "".join(output_parts)

    # Fallback: if server didn't report completion tokens, approximate
    # from the number of content delta chunks received
    if completion_tokens == 0 and content_chunk_count > 0:
        completion_tokens = content_chunk_count
        log.debug("Using content chunk count (%d) as completion_tokens fallback",
                  content_chunk_count)

    # Estimate prompt tokens if server did not report them
    if prompt_tokens == 0:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = max(1, total_chars // 4)

    # Calculate throughput
    prefill_tok_s = prompt_tokens / ttft_s if ttft_s > 0 else 0.0
    decode_tok_s = completion_tokens / decode_time if decode_time > 0 and completion_tokens > 0 else 0.0

    # Detect empty stream: no tokens generated at all
    if completion_tokens == 0 and not output_text:
        log.warning("Stream returned no content and no completion tokens")
        return RequestResult(
            success=False,
            error="Empty stream: no content or completion tokens returned",
            prompt_tokens=prompt_tokens,
            total_time_s=total_time,
        )

    result = RequestResult(
        success=True,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        total_time_s=total_time,
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        output_text=output_text,
    )

    log.debug(
        "Request complete: prompt_tok=%d completion_tok=%d ttft=%.1fms "
        "prefill=%.1ftok/s decode=%.1ftok/s total=%.2fs output_len=%d",
        result.prompt_tokens, result.completion_tokens, result.ttft_ms,
        result.prefill_tok_s, result.decode_tok_s, result.total_time_s,
        len(result.output_text),
    )

    return result


def send_concurrent_requests(
    server_url: str,
    messages_list: list[list[dict]],
    max_tokens: int = 256,
    concurrency: int = 8,
    temperature: float = 0.7,
    model: str = "default",
) -> list[RequestResult]:
    """Send multiple concurrent chat requests using :class:`~concurrent.futures.ThreadPoolExecutor`.

    Returns:
        A list of :class:`RequestResult` in the same order as *messages_list*.
    """
    log = logging.getLogger(__name__)
    log.info(
        "Sending %d concurrent requests (concurrency=%d, max_tokens=%d)",
        len(messages_list), concurrency, max_tokens,
    )

    results: list[RequestResult | None] = [None] * len(messages_list)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {}
        for idx, messages in enumerate(messages_list):
            fut = executor.submit(
                send_chat_request,
                server_url, messages, max_tokens, temperature, model,
            )
            future_to_idx[fut] = idx

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
                log.info(
                    "Concurrent request %d/%d done: success=%s tokens=%d",
                    idx + 1, len(messages_list),
                    results[idx].success, results[idx].completion_tokens,
                )
            except Exception as exc:
                log.error("Concurrent request %d failed with exception: %s",
                          idx + 1, exc, exc_info=True)
                results[idx] = RequestResult(
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

    # Filter out any None entries (should not happen, but be defensive)
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Server log collection
# ---------------------------------------------------------------------------


def collect_server_logs(
    hosts: list[str],
    remote_log: str = "/tmp/tp_server.log",
    local_dir: str = "logs",
    log_dir: str = "/tmp",
    num_ranks: int = 8,
) -> dict[str, str]:
    """Collect server log files from remote hosts via ``ssh cat``.

    Tries the following log files for each host, in order:
    1. *remote_log* (backward-compatible default: ``/tmp/tp_server.log``)
    2. ``{log_dir}/tp_rank0.log``, ``tp_rank1.log``, ... up to *num_ranks*

    All non-empty files found are concatenated and returned.

    Args:
        hosts: List of hostnames to collect logs from.
        remote_log: Legacy log path to try first (default ``/tmp/tp_server.log``).
        local_dir: Local directory to save copies of the logs.
        log_dir: Remote directory where ``tp_rank{N}.log`` files are written.
        num_ranks: Maximum rank index to probe (0..num_ranks-1).

    Returns:
        A dict mapping ``{host: combined_log_content}``.
    """
    log = logging.getLogger(__name__)
    os.makedirs(local_dir, exist_ok=True)
    collected: dict[str, str] = {}

    for host in hosts:
        parts: list[str] = []

        # 1. Backward-compatible legacy log
        log.info("Collecting legacy log from %s:%s", host, remote_log)
        result = ssh_run(host, f"cat {shlex.quote(remote_log)} 2>/dev/null",
                         timeout=30, logger=log)
        if result.returncode == 0 and result.stdout:
            log.debug("Got legacy log from %s (%d bytes)", host, len(result.stdout))
            parts.append(f"=== {remote_log} ===\n" + result.stdout)
        else:
            log.debug("No legacy log at %s:%s (rc=%d)", host, remote_log, result.returncode)

        # 2. Per-rank logs: tp_rank0.log, tp_rank1.log, ...
        for rank in range(num_ranks):
            rank_log = f"{log_dir}/tp_rank{rank}.log"
            log.info("Collecting rank log from %s:%s", host, rank_log)
            r = ssh_run(host, f"cat {shlex.quote(rank_log)} 2>/dev/null",
                        timeout=30, logger=log)
            if r.returncode == 0 and r.stdout:
                log.debug("Got rank %d log from %s (%d bytes)", rank, host, len(r.stdout))
                parts.append(f"=== {rank_log} ===\n" + r.stdout)
            else:
                log.debug("No rank log at %s:%s (rc=%d)", host, rank_log, r.returncode)
                # Stop probing further ranks once none are found beyond rank 1
                if rank > 0:
                    break

        if parts:
            combined = "\n\n".join(parts)
            collected[host] = combined
            # Save combined log locally
            local_path = os.path.join(local_dir, f"server_{host}.log")
            try:
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(combined)
                log.debug("Saved %s combined log to %s (%d bytes)",
                          host, local_path, len(combined))
            except OSError as exc:
                log.warning("Failed to save log locally: %s", exc)
        else:
            log.warning("No log content from %s", host)
            collected[host] = "(no log available)"

    return collected


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def write_results_md(
    test_name: str,
    config: dict,
    results: list[tuple[str, list[RequestResult]]] | list[RequestResult],
    errors: list[str],
    server_logs: dict[str, str] | None = None,
    extra_sections: str = "",
    output_dir: str = RESULTS_DIR,
    wall_time: float | None = None,
) -> str:
    """Write test results to a Markdown file.

    The *results* parameter accepts either:
    - A list of ``(section_name, [RequestResult, ...])`` tuples (grouped)
    - A flat list of :class:`RequestResult` (auto-wrapped)

    Args:
        wall_time: Optional wall-clock time in seconds for the entire test
            batch (e.g. concurrent requests).  When provided, aggregate
            throughput is computed from ``total_tokens / wall_time`` instead
            of ``total_tokens / sum(per_request_times)`` which double-counts
            overlapping concurrent requests.

    Returns:
        The path to the generated ``.md`` file.
    """
    log = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    # Normalise results to grouped format
    if results and not isinstance(results[0], tuple):
        grouped: list[tuple[str, list[RequestResult]]] = [("Results", results)]
    else:
        grouped = results  # type: ignore[assignment]

    lines: list[str] = []
    lines.append(f"# {test_name} Test Results\n")
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Configuration table
    lines.append("## Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, value in config.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # Results per section
    all_successful: list[RequestResult] = []
    total_requests = 0

    for section_name, section_results in grouped:
        lines.append(f"## {section_name}\n")
        lines.append("| # | Status | Prompt Tokens | Completion Tokens | "
                      "TTFT (ms) | Prefill tok/s | Decode tok/s | Total Time (s) |")
        lines.append("|---|--------|---------------|-------------------|"
                      "-----------|---------------|--------------|----------------|")

        for i, r in enumerate(section_results, 1):
            total_requests += 1
            status = "PASS" if r.success else "FAIL"
            error_note = ""
            if r.error:
                error_note = f" `{r.error[:60]}`"
            lines.append(
                f"| {i} | {status}{error_note} | {r.prompt_tokens} | "
                f"{r.completion_tokens} | {r.ttft_ms:.1f} | "
                f"{r.prefill_tok_s:.1f} | {r.decode_tok_s:.1f} | "
                f"{r.total_time_s:.2f} |"
            )
            if r.success:
                all_successful.append(r)

        lines.append("")

    # Summary
    lines.append("## Summary\n")
    working = len(all_successful) > 0
    lines.append(f"- **Working:** {'YES' if working else 'NO'}")
    lines.append(f"- **Total requests:** {total_requests}")
    lines.append(f"- **Successful:** {len(all_successful)}")
    lines.append(f"- **Failed:** {total_requests - len(all_successful)}")

    if all_successful:
        avg_prefill = sum(r.prefill_tok_s for r in all_successful) / len(all_successful)
        avg_decode = sum(r.decode_tok_s for r in all_successful) / len(all_successful)
        avg_ttft = sum(r.ttft_ms for r in all_successful) / len(all_successful)
        total_tokens = sum(r.completion_tokens for r in all_successful)

        # Use wall_time for aggregate throughput when available (concurrent tests),
        # otherwise fall back to sum of per-request times (sequential tests).
        if wall_time is not None and wall_time > 0:
            throughput = total_tokens / wall_time
            throughput_label = "Aggregate throughput (wall-clock)"
        else:
            total_time = sum(r.total_time_s for r in all_successful)
            throughput = total_tokens / total_time if total_time > 0 else 0.0
            throughput_label = "Avg per-request throughput"

        lines.append(f"- **Avg TTFT:** {avg_ttft:.1f} ms")
        lines.append(f"- **Avg prefill tok/s:** {avg_prefill:.1f}")
        lines.append(f"- **Avg decode tok/s:** {avg_decode:.1f}")
        lines.append(f"- **Total output tokens:** {total_tokens}")
        lines.append(f"- **{throughput_label}:** {throughput:.1f} tok/s")
    lines.append("")

    # Extra sections (e.g., acceptance rate)
    if extra_sections:
        lines.append(extra_sections)
        lines.append("")

    # Errors
    if errors:
        lines.append("## Errors\n")
        for i, err in enumerate(errors, 1):
            lines.append(f"{i}. {err}")
        lines.append("")

    # Server logs excerpt
    if server_logs:
        lines.append("## Server Logs (excerpt)\n")
        for host, log_content in server_logs.items():
            lines.append(f"### {host}\n")
            log_lines = log_content.splitlines()
            if len(log_lines) > 200:
                lines.append(f"*(showing last 200 of {len(log_lines)} lines)*\n")
                excerpt = log_lines[-200:]
            else:
                excerpt = log_lines
            lines.append("```")
            lines.extend(excerpt)
            lines.append("```")
            lines.append("")

    content = "\n".join(lines) + "\n"

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        log.info("Results written to %s", filepath)
    except OSError as exc:
        log.error("Failed to write results: %s", exc)
        # Try fallback location
        fallback = os.path.join("/tmp", filename)
        with open(fallback, "w", encoding="utf-8") as f:
            f.write(content)
        log.info("Results written to fallback: %s", fallback)
        return fallback

    return filepath
