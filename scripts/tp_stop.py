#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Distributed server stop script for vllm-mlx tensor-parallel nodes.

Discovers vllm_mlx processes on remote hosts, sends SIGTERM (or SIGKILL
with --force), waits for graceful shutdown, checks wired memory and port
release, then prints a per-node summary table.

Usage:
    # Stop servers on two hosts
    python scripts/tp_stop.py --hosts hwstudio1 hwstudio2

    # Stop servers using a JACCL hostfile
    python scripts/tp_stop.py --hostfile ~/mlx_hostfile.json

    # Force kill (skip SIGTERM)
    python scripts/tp_stop.py --hosts hwstudio1 hwstudio2 --force

    # Custom timeout
    python scripts/tp_stop.py --hosts hwstudio1 hwstudio2 --timeout 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def bold(t: str) -> str:
    return _c("1", t)


def cyan(t: str) -> str:
    return _c("36", t)


# Labels
STOP = cyan("[STOP]")
OK = green("[OK]")
WARN = yellow("[WARN]")
FAIL = red("[FAIL]")
CHECK = bold("[CHECK]")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIRED_MEMORY_THRESHOLD = 23_000_000  # pages
PORT_HTTP = 8000
PORT_JACCL = 32323
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def parse_hosts(args: argparse.Namespace) -> list[str]:
    """Parse hosts from --hostfile JSON (hosts[].ssh) or --hosts."""
    if args.hostfile:
        with open(args.hostfile) as f:
            data = json.load(f)
        hosts_list = data if isinstance(data, list) else data.get("hosts", [])
        hosts: list[str] = []
        for entry in hosts_list:
            if isinstance(entry, dict):
                hosts.append(entry["ssh"])
            else:
                hosts.append(str(entry))
        if not hosts:
            print(f"{FAIL} No hosts found in {args.hostfile}")
            sys.exit(1)
        return hosts
    if args.hosts:
        return args.hosts
    print(f"{FAIL} Provide --hosts or --hostfile")
    sys.exit(1)


def ssh_run(host: str, cmd: str, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    """Run command on remote host via SSH."""
    ssh_cmd = ["ssh", *SSH_OPTS, host, cmd]
    try:
        return subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=ssh_cmd, returncode=-1, stdout="", stderr="timeout",
        )


def find_vllm_pids(host: str) -> list[int] | None:
    """Find vllm_mlx PIDs on host. Returns None on SSH failure."""
    result = ssh_run(host, "pgrep -f 'vllm_mlx.distributed_launcher'")
    # SSH failure: timeout (-1) or connection refused (255)
    if result.returncode == -1 or result.returncode == 255:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return []
    pids: list[int] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def check_wired_memory(host: str) -> tuple[bool, int, str | None]:
    """Check wired memory pages on host. Returns (ok, pages, error)."""
    result = ssh_run(host, "vm_stat | grep 'Pages wired down'")
    # SSH failure
    if result.returncode == -1 or result.returncode == 255:
        err = result.stderr.strip() or "SSH connection failed"
        return False, 0, err
    if result.returncode != 0:
        return True, 0, None
    # "Pages wired down:  12345."
    parts = result.stdout.strip().rstrip(".").split()
    if not parts:
        return True, 0, None
    try:
        pages = int(parts[-1])
    except ValueError:
        return True, 0, None
    return pages < WIRED_MEMORY_THRESHOLD, pages, None


def check_port(host: str, port: int) -> bool | None:
    """Check if a port is still bound on host. Returns True if free, None on SSH failure."""
    result = ssh_run(host, f"lsof -iTCP:{port} -sTCP:LISTEN -P -n 2>/dev/null")
    # SSH failure
    if result.returncode == -1 or result.returncode == 255:
        return None
    return result.returncode != 0 or not result.stdout.strip()


# ---------------------------------------------------------------------------
# Node status tracking
# ---------------------------------------------------------------------------


class NodeStatus:
    def __init__(self, host: str) -> None:
        self.host = host
        self.pids_found: list[int] = []
        self.ssh_error: str | None = None
        self.sigterm_sent = False
        self.sigkill_sent = False
        self.all_dead = False
        self.wired_ok = True
        self.wired_pages = 0
        self.wired_error: str | None = None
        self.port_http_free = True
        self.port_jaccl_free = True

    @property
    def clean(self) -> bool:
        return (
            self.all_dead
            and self.ssh_error is None
            and self.wired_ok
            and self.wired_error is None
            and self.port_http_free
            and self.port_jaccl_free
        )

    @property
    def status_label(self) -> str:
        if self.ssh_error is not None:
            return red("SSH FAILED")
        if self.clean and not self.pids_found:
            return green("CLEAN (no procs)")
        if self.clean and not self.sigkill_sent:
            return green("CLEAN")
        if self.clean and self.sigkill_sent:
            return yellow("KILLED (clean)")
        return red("ISSUES")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def stop_nodes(
    hosts: list[str],
    force: bool = False,
    timeout: int = 15,
) -> int:
    """Stop vllm_mlx on all hosts. Returns exit code."""
    nodes = [NodeStatus(h) for h in hosts]
    used_sigkill = False

    # Step 1: Process discovery
    print(f"\n{STOP} Discovering processes on {len(hosts)} node(s)...")
    total_pids = 0
    for ns in nodes:
        pids = find_vllm_pids(ns.host)
        if pids is None:
            ns.ssh_error = "SSH connection failed"
            print(f"  {ns.host}: {red('SSH FAILED')} — cannot discover processes")
            continue
        ns.pids_found = pids
        count = len(ns.pids_found)
        total_pids += count
        if count:
            print(f"  {ns.host}: {count} process(es) — PIDs {ns.pids_found}")
        else:
            print(f"  {ns.host}: no processes")
            ns.all_dead = True

    # Step 2: Graceful shutdown (SIGTERM)
    if not force and total_pids > 0:
        print(f"\n{STOP} Sending SIGTERM to all nodes...")
        for ns in nodes:
            if ns.ssh_error or not ns.pids_found:
                continue
            pid_str = " ".join(str(p) for p in ns.pids_found)
            result = ssh_run(ns.host, f"kill -15 {pid_str} 2>/dev/null")
            if result.returncode == 0 or result.returncode == -1:
                # -1 is timeout but signal may have been sent
                pass
            ns.sigterm_sent = True
            print(f"  {ns.host}: SIGTERM -> {pid_str}")

        # Step 3: Wait / poll
        print(f"\n{STOP} Waiting up to {timeout}s for graceful shutdown...")
        elapsed = 0
        while elapsed < timeout:
            all_done = True
            for ns in nodes:
                if ns.all_dead or ns.ssh_error:
                    continue
                remaining = find_vllm_pids(ns.host)
                if remaining is None:
                    # SSH failed mid-poll; skip this node
                    continue
                if not remaining:
                    ns.all_dead = True
                else:
                    all_done = False
            if all_done:
                print(f"  All processes exited after {elapsed}s.")
                break
            time.sleep(2)
            elapsed += 2

    # Step 4: Force kill (SIGKILL) for survivors
    for ns in nodes:
        if ns.all_dead or ns.ssh_error:
            continue
        remaining = find_vllm_pids(ns.host)
        if remaining is None:
            print(f"\n{FAIL} {ns.host}: SSH failed during SIGKILL phase")
            ns.ssh_error = "SSH connection failed"
            continue
        if not remaining:
            ns.all_dead = True
            continue
        # Need SIGKILL
        pid_str = " ".join(str(p) for p in remaining)
        label = "SIGKILL (--force)" if force else "SIGKILL (timeout)"
        print(f"\n{WARN} {ns.host}: processes still alive, sending {label}...")
        ssh_run(ns.host, f"kill -9 {pid_str} 2>/dev/null")
        ns.sigkill_sent = True
        used_sigkill = True
        print(f"  {ns.host}: SIGKILL -> {pid_str}")
        time.sleep(1)
        # Verify
        final = find_vllm_pids(ns.host)
        if final is None:
            print(f"  {FAIL} {ns.host}: SSH failed verifying SIGKILL")
        else:
            ns.all_dead = len(final) == 0
            if not ns.all_dead:
                print(f"  {FAIL} {ns.host}: processes survived SIGKILL: {final}")

    # Step 5: Post-shutdown checks
    print(f"\n{CHECK} Running post-shutdown checks...")
    for ns in nodes:
        if ns.ssh_error:
            print(f"  {ns.host}: {red('skipped')} (SSH unreachable)")
            continue

        # Wired memory
        ns.wired_ok, ns.wired_pages, ns.wired_error = check_wired_memory(ns.host)
        if ns.wired_error:
            print(f"  {ns.host} wired memory: {red('SSH FAILED')} — {ns.wired_error}")
        else:
            pages_m = f"{ns.wired_pages / 1_000_000:.1f}M"
            if ns.wired_ok:
                print(f"  {ns.host} wired memory: {pages_m} pages {green('OK')}")
            else:
                print(
                    f"  {ns.host} wired memory: {red(pages_m)} pages "
                    f"(> {WIRED_MEMORY_THRESHOLD // 1_000_000}M) "
                    f"— {red('REBOOT REQUIRED')}"
                )

        # Ports
        http_result = check_port(ns.host, PORT_HTTP)
        jaccl_result = check_port(ns.host, PORT_JACCL)
        if http_result is None:
            ns.port_http_free = False
            print(f"  {ns.host} port {PORT_HTTP} (HTTP): {red('SSH FAILED')}")
        else:
            ns.port_http_free = http_result
            if not ns.port_http_free:
                print(f"  {ns.host} port {PORT_HTTP} (HTTP): {red('still bound')}")
        if jaccl_result is None:
            ns.port_jaccl_free = False
            print(f"  {ns.host} port {PORT_JACCL} (JACCL): {red('SSH FAILED')}")
        else:
            ns.port_jaccl_free = jaccl_result
            if not ns.port_jaccl_free:
                print(f"  {ns.host} port {PORT_JACCL} (JACCL): {red('still bound')}")

    # Step 6: JACCL cooldown notice
    if used_sigkill:
        print(
            f"\n{WARN} SIGKILL was used — wait 30 seconds before restarting "
            f"to avoid JACCL RDMA EBUSY errors."
        )

    # Summary table
    print(f"\n{bold('Summary')}")
    print("=" * 60)
    header = f"  {'Host':<16} {'Procs':<8} {'Signal':<10} {'Memory':<10} {'Status'}"
    print(header)
    print("  " + "-" * 56)

    exit_code = 0
    for ns in nodes:
        proc_str = str(len(ns.pids_found)) if ns.pids_found else "-"
        if ns.sigkill_sent:
            sig_str = red("SIGKILL")
        elif ns.sigterm_sent:
            sig_str = "SIGTERM"
        else:
            sig_str = "-"
        mem_str = red("HIGH") if not ns.wired_ok else green("OK")
        print(
            f"  {ns.host:<16} {proc_str:<8} {sig_str:<10} {mem_str:<10} "
            f"{ns.status_label}"
        )
        if not ns.clean:
            exit_code = 1

    print()
    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stop vllm-mlx distributed servers across multiple nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --hosts hwstudio1 hwstudio2
  %(prog)s --hostfile ~/mlx_hostfile.json
  %(prog)s --hosts hwstudio1 hwstudio2 --force
  %(prog)s --hosts hwstudio1 hwstudio2 --timeout 30
""",
    )
    host_group = parser.add_mutually_exclusive_group()
    host_group.add_argument(
        "--hostfile",
        metavar="PATH",
        help="Extract hosts from hostfile JSON (hosts[].ssh).",
    )
    host_group.add_argument(
        "--hosts",
        nargs="+",
        metavar="HOST",
        help="Hosts to stop.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip SIGTERM, go straight to SIGKILL.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        metavar="SECS",
        help="Graceful shutdown wait time (default: 15).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hosts = parse_hosts(args)
    return stop_nodes(hosts, force=args.force, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
