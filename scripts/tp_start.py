#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
import urllib.error

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str: return _c("32", text)
def yellow(text: str) -> str: return _c("33", text)
def red(text: str) -> str: return _c("31", text)
def bold(text: str) -> str: return _c("1", text)
def cyan(text: str) -> str: return _c("36", text)


WIRED_MEMORY_THRESHOLD = 23_000_000
PYTHON = "/opt/homebrew/bin/python3.14"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
JACCL_PORT = 32323
DEFAULT_SERVER_PORT = 8000
DEFAULT_LOG_DIR = "/tmp"
DEFAULT_TIMEOUT = 600


def ssh_run(host: str, cmd: str, timeout: int = 10) -> subprocess.CompletedProcess:
    ssh_cmd = ["ssh"] + SSH_OPTS + [host, cmd]
    try:
        return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(ssh_cmd, -1, "", "timeout")


def parse_hosts(args: argparse.Namespace) -> list[str]:
    if args.hostfile:
        try:
            with open(args.hostfile, "r", encoding="utf-8") as f:
                data = json.load(f)
            hosts = []
            for entry in data["hosts"]:
                if isinstance(entry, dict):
                    hosts.append(entry["ssh"])
                else:
                    hosts.append(str(entry))
        except Exception as e:
            print(f"{red('[FAIL]')} Failed to read hostfile {args.hostfile}: {e}")
            sys.exit(1)
        if not hosts:
            print(f"{red('[FAIL]')} No hosts found in hostfile: {args.hostfile}")
            sys.exit(1)
        return hosts
    if args.hosts:
        return args.hosts
    print(f"{red('[FAIL]')} Must provide either --hostfile or --hosts")
    sys.exit(1)


def extract_server_port(server_args: list[str]) -> int:
    for i, arg in enumerate(server_args):
        if arg == "--port" and i + 1 < len(server_args):
            try:
                return int(server_args[i + 1])
            except ValueError:
                break
        if arg.startswith("--port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                break
    return DEFAULT_SERVER_PORT


def check_wired_memory(host: str) -> tuple[bool, int]:
    result = ssh_run(host, "vm_stat | grep 'Pages wired'")
    line = result.stdout.replace(".", "")
    match = re.search(r"(\d+)", line)
    pages = int(match.group(1)) if match else WIRED_MEMORY_THRESHOLD
    return pages < WIRED_MEMORY_THRESHOLD, pages


def find_vllm_pids(host: str) -> list[int]:
    result = ssh_run(host, "pgrep -f 'vllm_mlx.distributed_launcher'")
    if result.returncode != 0:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def run_preflight_checks(hosts: list[str], server_port: int, backend: str) -> bool:
    all_ok = True
    print(bold("[CHECK] Pre-flight checks"))
    for host in hosts:
        print(f"  {bold(host)}:")

        wired_ok, pages = check_wired_memory(host)
        if not wired_ok:
            print(
                f"    {red('[FAIL]')} Wired memory: {pages:,} pages "
                f"(threshold: {WIRED_MEMORY_THRESHOLD:,})"
            )
            all_ok = False
        else:
            print(f"    {green('[OK]')}   Wired memory: {pages:,} pages")

        pids = find_vllm_pids(host)
        if pids:
            print(f"    {red('[FAIL]')} Existing vllm_mlx processes: {pids}")
            all_ok = False
        else:
            print(f"    {green('[OK]')}   No existing vllm_mlx processes")

        port_check = ssh_run(host, f"lsof -i :{server_port} -sTCP:LISTEN -P -n 2>/dev/null")
        if port_check.stdout.strip():
            print(f"    {red('[FAIL]')} Port {server_port} already in use")
            all_ok = False
        else:
            print(f"    {green('[OK]')}   Port {server_port} is available")

        if backend == "jaccl":
            jaccl_check = ssh_run(host, f"lsof -i :{JACCL_PORT} -sTCP:LISTEN -P -n 2>/dev/null")
            if jaccl_check.stdout.strip():
                print(f"    {red('[FAIL]')} Port {JACCL_PORT} already in use")
                all_ok = False
            else:
                print(f"    {green('[OK]')}   Port {JACCL_PORT} is available")

    print()
    return all_ok


def run_rdma_check(hosts: list[str]) -> bool:
    ok = True
    print(bold("[CHECK] RDMA connectivity"))
    for host in hosts:
        print(f"  {bold(host)}:")
        result = ssh_run(host, "test -f /tmp/mlx_ibv_devices.json")
        if result.returncode != 0:
            print(f"    {red('[FAIL]')} Missing /tmp/mlx_ibv_devices.json (run rdma_setup.py)")
            ok = False
        else:
            print(f"    {green('[OK]')}   /tmp/mlx_ibv_devices.json found")
    print()
    return ok


def launch_server(hosts: list[str], args: argparse.Namespace, server_args: list[str]) -> bool:
    cmd_parts = [PYTHON, "-m", "vllm_mlx.distributed_launcher"]
    cmd_parts += ["--backend", args.backend]
    if args.hostfile:
        cmd_parts += ["--hostfile", args.hostfile]
    elif args.hosts:
        cmd_parts += ["--hosts"] + args.hosts
    if args.num_ranks:
        cmd_parts += ["-n", str(args.num_ranks)]
    if args.env:
        cmd_parts += ["--env"] + args.env
    if server_args:
        cmd_parts += ["--"] + server_args

    remote_cmd = shlex.join(cmd_parts)

    if args.foreground:
        print(f"{cyan('[START]')} Launching server in foreground mode...")
        full_cmd = f"cd /Users/hw/vllm-mlx && {remote_cmd}"
        print(f"    Command: {full_cmd}")
        os.execvp("ssh", ["ssh"] + SSH_OPTS + [hosts[0], full_cmd])
        return False

    log_dir = shlex.quote(args.log_dir)
    log_file = f"{log_dir}/tp_server.log"
    print(f"{cyan('[START]')} Launching server in background mode...")
    full_cmd = f"mkdir -p {log_dir} && cd /Users/hw/vllm-mlx && nohup {remote_cmd} > {log_file} 2>&1 &"
    result = ssh_run(hosts[0], full_cmd, timeout=30)
    if result.returncode != 0:
        print(f"    {red('[FAIL]')} Failed to launch server on {hosts[0]}: {result.stderr.strip()}")
        return False
    print(f"    {green('[OK]')}   Server process started on {hosts[0]}")
    print(f"    Log: {args.log_dir}/tp_server.log")
    return True


def wait_for_startup(host: str, log_dir: str, timeout: int) -> bool:
    print(f"{cyan('[WAIT]')} Waiting for server startup (timeout: {timeout}s)...")
    start = time.time()
    last_progress = -30
    while True:
        elapsed = int(time.time() - start)
        print(f"\r    Elapsed: {elapsed}s / {timeout}s", end="")
        sys.stdout.flush()

        result = ssh_run(host, f"grep -c 'Uvicorn running' {log_dir}/tp_server.log 2>/dev/null")
        output = result.stdout.strip()
        if output:
            try:
                if int(output) > 0:
                    print(f"\n    {green('[OK]')}   Server started successfully")
                    return True
            except ValueError:
                pass

        if elapsed >= timeout:
            print(f"\n    {red('[FAIL]')} Server did not start within {timeout}s")
            return False

        if elapsed > 0 and elapsed % 30 == 0 and elapsed != last_progress:
            print(f"\n    [WAIT] Still waiting... ({elapsed}s)")
            last_progress = elapsed

        time.sleep(2)


def health_check(host: str, port: int) -> bool:
    url = f"http://{host}:{port}/v1/models"
    print(f"{cyan('[CHECK]')} Health check: {url}")
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            status = getattr(response, "status", 200)
            body = response.read(300).decode("utf-8", errors="replace")
        snippet = body.strip().replace("\n", " ")[:160]
        if 200 <= status < 300:
            print(f"    {green('[OK]')}   HTTP {status}: {snippet}")
            return True
        print(f"    {red('[FAIL]')} HTTP {status}: {snippet}")
        return False
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"    {red('[FAIL]')} {e}")
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start distributed TP server for vllm-mlx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Launch with explicit hosts
  %(prog)s --backend jaccl --hosts hwstudio1 hwstudio2 -- \\
      --model /Users/hw/models/Kimi-K2.5 --host 0.0.0.0 --port 8000 \\
      --continuous-batching

  # Launch with hostfile
  %(prog)s --backend jaccl --hostfile ~/mlx_hostfile.json -- \\
      --model /Users/hw/models/Kimi-K2.5 --continuous-batching
""",
    )

    parser.add_argument("--backend", default="jaccl", choices=["jaccl", "ring", "any"])
    host_group = parser.add_mutually_exclusive_group()
    host_group.add_argument("--hostfile", type=str, default=None)
    host_group.add_argument("--hosts", nargs="+", default=None)
    parser.add_argument("-n", "--num-ranks", type=int, default=None)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")

    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)

    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    server_args = args.server_args
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]

    hosts = parse_hosts(args)
    server_port = extract_server_port(server_args)

    if not args.skip_checks:
        if not run_preflight_checks(hosts, server_port, args.backend):
            return 1
        if args.backend == "jaccl" and not run_rdma_check(hosts):
            return 1

    if not launch_server(hosts, args, server_args):
        return 1

    if not args.foreground:
        if not wait_for_startup(hosts[0], args.log_dir, args.timeout):
            return 1
        if not health_check(hosts[0], server_port):
            print(f"    {yellow('[WARN]')} Health check failed, but not treating as fatal")

    print(f"\n{bold(green('[OK]'))} Server is running on {hosts[0]}:{server_port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
