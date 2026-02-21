#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Real-time cluster monitoring dashboard for Mac Studio nodes.

Displays CPU (with per-core activity), GPU, RAM, Wired Memory, Swap,
and RDMA metrics side-by-side with btop-like visual elements (progress
bars, sparkline history graphs, per-core grids).

Polls remote nodes via SSH and renders using the Rich library.

Usage:
    # Basic two-node monitoring
    python scripts/cluster_monitor.py \\
      --node hwstudio1 \\
      --node hwstudio2

    # With custom refresh interval
    python scripts/cluster_monitor.py \\
      --node hwstudio1 \\
      --node hwstudio2 \\
      --interval 3

    # With SSH proxy (hwstudio2 reached via hwstudio1)
    python scripts/cluster_monitor.py \\
      --node hwstudio1 \\
      --node hwstudio2 \\
      --proxy hwstudio1:hwstudio2
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
BAR_FILL = "█"
BAR_EMPTY = "░"
BAR_WIDTH = 20
HISTORY_LENGTH = 30
SSH_TIMEOUT = 8

# Color thresholds
COLOR_GREEN_MAX = 50.0
COLOR_YELLOW_MAX = 75.0

# Wired memory thresholds (GB)
WIRED_NORMAL_MAX = 10.0
WIRED_WARNING_MAX = 50.0

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NodeMetrics:
    """Metrics collected from a single node."""

    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    core_percents: list[float] = field(default_factory=list)
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    wired_gb: float = 0.0
    swap_gb: float = 0.0
    ncpu: int = 0
    uptime_seconds: int = 0
    rdma_enabled: bool | None = None
    rdma_ip: str = ""
    server_running: bool = False
    online: bool = False
    error: str = ""


class MetricsHistory:
    """Rolling buffer of metric values for sparkline rendering."""

    def __init__(self, maxlen: int = HISTORY_LENGTH) -> None:
        self.cpu: deque[float] = deque(maxlen=maxlen)
        self.gpu: deque[float] = deque(maxlen=maxlen)
        self.ram: deque[float] = deque(maxlen=maxlen)
        self.wired: deque[float] = deque(maxlen=maxlen)

    def record(self, metrics: NodeMetrics) -> None:
        """Record a data point (only when node is online)."""
        if metrics.online:
            self.cpu.append(metrics.cpu_percent)
            self.gpu.append(metrics.gpu_percent)
            self.ram.append(metrics.ram_used_gb)
            self.wired.append(metrics.wired_gb)


# ---------------------------------------------------------------------------
# SSH execution
# ---------------------------------------------------------------------------

# The single combined command to collect all metrics from a macOS node.
COLLECT_CMD = (
    "sysctl -n hw.memsize hw.ncpu kern.boottime vm.swapusage; "
    "vm_stat; "
    "top -l 1 -n 0 -s 0; "
    "rdma_ctl status 2>/dev/null; "
    "ifconfig en5 2>/dev/null | grep 'inet '; "
    "cat /tmp/mlx_ibv_devices.json 2>/dev/null; "
    "echo __VLLM_PROC__; pgrep -f 'vllm_mlx\\.distributed_launcher|vllm_mlx\\.server|vllm_mlx\\.cli|uvicorn.*vllm' 2>/dev/null | head -1; "
    "sudo powermetrics --samplers gpu_power,cpu_power -n 1 -i 500 2>/dev/null"
)


def ssh_run(
    host: str,
    command: str,
    *,
    proxy: str | None = None,
    timeout: int = SSH_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Execute a command on a remote host via SSH.

    If *proxy* is set, the command is tunneled through the proxy host:
        ssh proxy "ssh host 'command'"
    """
    escaped_cmd = command.replace("'", "'\"'\"'")

    if proxy:
        inner = f"ssh -o ConnectTimeout=3 {host} '{escaped_cmd}'"
        ssh_cmd = ["ssh", "-o", "ConnectTimeout=3", proxy, inner]
    else:
        ssh_cmd = ["ssh", "-o", "ConnectTimeout=3", host, command]

    return subprocess.run(
        ssh_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------


def parse_metrics(output: str) -> NodeMetrics:
    """Parse the combined SSH output into a NodeMetrics object."""
    metrics = NodeMetrics(online=True)
    lines = output.splitlines()

    # --- sysctl values (first few lines) ---
    # Line 0: hw.memsize (bytes)
    # Line 1: hw.ncpu
    # Line 2: kern.boottime (e.g., "{ sec = 1739000000, usec = 0 } ...")
    # Line 3: vm.swapusage (e.g., "total = 0.00M  used = 0.00M  free = 0.00M")
    try:
        metrics.ram_total_gb = int(lines[0].strip()) / (1024**3)
    except (IndexError, ValueError):
        pass

    try:
        metrics.ncpu = int(lines[1].strip())
    except (IndexError, ValueError):
        pass

    # kern.boottime
    try:
        m = re.search(r"sec\s*=\s*(\d+)", lines[2])
        if m:
            boot_epoch = int(m.group(1))
            metrics.uptime_seconds = int(time.time()) - boot_epoch
    except (IndexError, ValueError):
        pass

    # vm.swapusage
    try:
        m = re.search(r"used\s*=\s*([\d.]+)([MG])", lines[3])
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            if unit == "M":
                metrics.swap_gb = val / 1024.0
            elif unit == "G":
                metrics.swap_gb = val
    except (IndexError, ValueError):
        pass

    # --- vm_stat parsing ---
    page_size = 16384  # default for Apple Silicon
    vm_stat_data: dict[str, int] = {}

    for line in lines:
        # Parse page size from header
        ps_match = re.match(
            r"Mach Virtual Memory Statistics:\s*\(page size of (\d+) bytes\)",
            line,
        )
        if ps_match:
            page_size = int(ps_match.group(1))
            continue

        # Parse vm_stat value lines:
        #   "Pages wired down: 462952."
        #   "Anonymous pages: 1261387."
        #   "Pages occupied by compressor: 0."
        pg_match = re.match(r'^"?Pages\s+(.+?):\s+([\d]+)\.?$', line.strip())
        if pg_match:
            key = pg_match.group(1).strip().lower()
            vm_stat_data[key] = int(pg_match.group(2))
            continue
        anon_match = re.match(r'^(\w[\w-]*)\s+pages:\s+([\d]+)\.?$', line.strip())
        if anon_match:
            key = anon_match.group(1).strip().lower()
            vm_stat_data[key] = int(anon_match.group(2))

    pages_wired = vm_stat_data.get("wired down", 0)
    pages_anonymous = vm_stat_data.get("anonymous", 0)
    pages_compressor = vm_stat_data.get("occupied by compressor", 0)

    # Real memory = anonymous (app heap/stack) + wired (Metal/GPU + kernel)
    # + compressor (compressed pages). Excludes file-backed cache which macOS
    # releases on demand and inflates usage during rsync/large file I/O.
    if metrics.ram_total_gb > 0:
        real_used_pages = pages_anonymous + pages_wired + pages_compressor
        metrics.ram_used_gb = real_used_pages * page_size / (1024**3)
        metrics.ram_used_gb = max(0.0, min(metrics.ram_used_gb, metrics.ram_total_gb))

    metrics.wired_gb = pages_wired * page_size / (1024**3)

    # --- top output: CPU usage ---
    for line in lines:
        cpu_match = re.search(
            r"CPU usage:\s*([\d.]+)%\s*user,\s*([\d.]+)%\s*sys",
            line,
        )
        if cpu_match:
            user = float(cpu_match.group(1))
            sys_ = float(cpu_match.group(2))
            metrics.cpu_percent = user + sys_
            break

    # --- GPU Active Residency (from powermetrics) ---
    # Format: "GPU HW active residency:   1.72% (...)"
    try:
        gpu_match = re.search(
            r"GPU HW active residency:\s*([\d.]+)%", output
        )
        if gpu_match:
            metrics.gpu_percent = float(gpu_match.group(1))
    except (ValueError, AttributeError):
        pass

    # --- Per-core CPU active residency (from powermetrics) ---
    # Format: "CPU 0 active residency:  16.48% (...)"
    try:
        core_matches = re.findall(
            r"CPU\s+(\d+)\s+active residency:\s*([\d.]+)%", output
        )
        if core_matches:
            # Sort by CPU index and extract percentages
            sorted_cores = sorted(core_matches, key=lambda x: int(x[0]))
            metrics.core_percents = [float(v) for _, v in sorted_cores]
    except (ValueError, AttributeError):
        pass

    # --- RDMA status ---
    # rdma_ctl status outputs just "enabled" or "disabled" (no "rdma" prefix)
    for line in lines:
        stripped = line.strip().lower()
        if stripped == "enabled":
            metrics.rdma_enabled = True
            break
        elif stripped == "disabled":
            metrics.rdma_enabled = False
            break

    # --- RDMA IP (ifconfig en5 inet) ---
    for line in lines:
        inet_match = re.search(r"inet\s+([\d.]+)", line)
        if inet_match:
            ip = inet_match.group(1)
            # Only capture RDMA-range IPs (10.254.x.x) or similar
            if ip.startswith("10.") or ip.startswith("192.168.2."):
                metrics.rdma_ip = ip
                break

    # --- Server process detection ---
    # Look for PID after our __VLLM_PROC__ marker
    marker_found = False
    for line in lines:
        if "__VLLM_PROC__" in line:
            marker_found = True
            continue
        if marker_found and line.strip().isdigit():
            metrics.server_running = True
            break

    return metrics


# ---------------------------------------------------------------------------
# Node poller
# ---------------------------------------------------------------------------


class NodePoller:
    """Polls a single node via SSH and returns parsed metrics."""

    def __init__(self, host: str, proxy: str | None = None) -> None:
        self.host = host
        self.proxy = proxy

    def poll(self) -> NodeMetrics:
        """Execute the collection command via SSH and parse results."""
        try:
            result = ssh_run(
                self.host,
                COLLECT_CMD,
                proxy=self.proxy,
                timeout=SSH_TIMEOUT,
            )
            if result.returncode != 0:
                return NodeMetrics(
                    online=False,
                    error=result.stderr.strip()[:80] if result.stderr else "SSH failed",
                )
            return parse_metrics(result.stdout)
        except subprocess.TimeoutExpired:
            return NodeMetrics(online=False, error="SSH timeout")
        except Exception as exc:
            return NodeMetrics(online=False, error=str(exc)[:80])


# ---------------------------------------------------------------------------
# Visual rendering helpers
# ---------------------------------------------------------------------------


def _threshold_color(percent: float) -> str:
    """Return a Rich color name based on percentage thresholds."""
    if percent <= COLOR_GREEN_MAX:
        return "green"
    elif percent <= COLOR_YELLOW_MAX:
        return "yellow"
    return "red"


def _render_bar(label: str, percent: float, suffix: str, width: int = BAR_WIDTH) -> Text:
    """Render a labeled progress bar with color gradient.

    Example: CPU  [████████░░░░░░░░░░░░]  34.2%
    """
    percent = max(0.0, min(100.0, percent))
    filled = int(round(percent / 100.0 * width))
    empty = width - filled

    color = _threshold_color(percent)

    text = Text()
    text.append(f"  {label:<6}", style="bold")
    text.append("[", style="dim")
    text.append(BAR_FILL * filled, style=color)
    text.append(BAR_EMPTY * empty, style="dim")
    text.append("]", style="dim")
    text.append(f"  {suffix:>8}", style=f"bold {color}")
    return text


def _render_sparkline(values: deque[float], max_val: float = 100.0) -> Text:
    """Render a sparkline graph from historical values."""
    if not values:
        return Text("         (no history)", style="dim")

    text = Text("         ")  # indent to align under bar
    max_idx = len(SPARKLINE_CHARS) - 1

    for val in values:
        normalized = max(0.0, min(1.0, val / max_val)) if max_val > 0 else 0.0
        idx = int(normalized * max_idx)
        char = SPARKLINE_CHARS[idx]
        color = _threshold_color(val if max_val == 100.0 else (val / max_val * 100.0))
        text.append(char, style=color)

    return text


def _render_core_block(values: list[float], max_idx: int) -> Text:
    """Render a row of core blocks as colored unicode chars."""
    t = Text()
    for val in values:
        normalized = max(0.0, min(1.0, val / 100.0))
        idx = int(normalized * max_idx)
        char = SPARKLINE_CHARS[idx]
        color = _threshold_color(val)
        t.append(char, style=color)
    return t


# Apple Silicon cluster layouts (core index ranges)
# M3/M4 Ultra 32 cores: E0(0-3) P0(4-9) P1(10-15) | E1(16-19) P2(20-25) P3(26-31)
# M3/M4 Max  16 cores: E(0-3) P0(4-9) P1(10-15)

_ULTRA_DIE0 = [(0, 4, "E"), (4, 10, "P"), (10, 16, "P")]  # die 0: 16 cores
_ULTRA_DIE1 = [(16, 20, "E"), (20, 26, "P"), (26, 32, "P")]  # die 1: 16 cores
_MAX_CLUSTERS = [(0, 4, "E"), (4, 10, "P"), (10, 16, "P")]


def _render_core_grid(core_percents: list[float]) -> Text | None:
    """Render per-core activity as a two-row grid with cluster labels.

    Returns None if *core_percents* is empty (powermetrics unavailable).
    """
    if not core_percents:
        return None

    max_idx = len(SPARKLINE_CHARS) - 1
    n = len(core_percents)
    pad = "         "  # 9-char indent to align with bars

    text = Text()

    if n >= 32:
        # Ultra: two dies, one row each
        for die_idx, clusters in enumerate([_ULTRA_DIE0, _ULTRA_DIE1]):
            # Block row
            text.append(pad)
            for start, end, _ in clusters:
                vals = core_percents[start:min(end, n)]
                text.append_text(_render_core_block(vals, max_idx))
                text.append(" ", style="dim")
            text.append("\n")
            # Legend row
            text.append(pad)
            for start, end, label in clusters:
                count = min(end, n) - start
                text.append(label + "-" * (count - 1) + " ", style="dim")
            text.append(f"  die{die_idx}\n", style="dim")
    elif n >= 16:
        # Max: single die
        text.append(pad)
        for start, end, _ in _MAX_CLUSTERS:
            vals = core_percents[start:min(end, n)]
            text.append_text(_render_core_block(vals, max_idx))
            text.append(" ", style="dim")
        text.append("\n")
        text.append(pad)
        for start, end, label in _MAX_CLUSTERS:
            count = min(end, n) - start
            text.append(label + "-" * (count - 1) + " ", style="dim")
        text.append("\n")
    else:
        # Small chips: single row
        text.append(pad)
        text.append_text(_render_core_block(core_percents, max_idx))
        text.append("\n")

    text.append(f"         ({n} cores)\n", style="dim")
    return text


def _format_uptime(seconds: int) -> str:
    """Format uptime as 'Xd Yh Zm'."""
    if seconds <= 0:
        return "N/A"
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    return f"{days}d {hours}h {minutes}m"


def _render_node_panel(
    host: str,
    metrics: NodeMetrics,
    history: MetricsHistory,
    blink_on: bool,
) -> Panel:
    """Render a complete panel for a single node."""
    content = Text()

    if not metrics.online:
        # --- OFFLINE panel ---
        content.append("\n")
        content.append("  Node is unreachable\n", style="dim")
        if metrics.error:
            content.append(f"  {metrics.error}\n", style="dim red")
        content.append("\n")
        content.append("  Waiting for node to come back online...\n", style="dim")
        content.append("\n")

        # Still show last sparkline history if available
        if history.cpu:
            content.append("  Last CPU history:\n", style="dim")
            content.append(_render_sparkline(history.cpu))
            content.append("\n")

        status_text = Text()
        status_text.append(f" {host} ", style="bold")
        if blink_on:
            status_text.append("● OFFLINE", style="bold red blink")
        else:
            status_text.append("● OFFLINE", style="bold red")

        return Panel(
            content,
            title=status_text,
            border_style="dim red",
            padding=(0, 1),
        )

    # --- ONLINE panel ---
    content.append("\n")

    # --- CPU ---
    cpu_bar = _render_bar("CPU", metrics.cpu_percent, f"{metrics.cpu_percent:.1f}%")
    content.append_text(cpu_bar)
    content.append("\n")
    content.append_text(_render_sparkline(history.cpu))
    content.append("\n")

    # Per-core activity grid
    core_grid = _render_core_grid(metrics.core_percents)
    if core_grid is not None:
        content.append_text(core_grid)
    else:
        content.append(f"         {metrics.ncpu} cores\n", style="dim")
    content.append("\n")

    # --- GPU ---
    gpu_bar = _render_bar("GPU", metrics.gpu_percent, f"{metrics.gpu_percent:.1f}%")
    content.append_text(gpu_bar)
    content.append("\n\n")
    content.append_text(_render_sparkline(history.gpu))
    content.append("\n\n")

    # RAM bar + detail + sparkline
    ram_percent = (
        (metrics.ram_used_gb / metrics.ram_total_gb * 100.0)
        if metrics.ram_total_gb > 0
        else 0.0
    )
    ram_bar = _render_bar("RAM", ram_percent, f"{ram_percent:.1f}%")
    content.append_text(ram_bar)
    content.append("\n")
    content.append(
        f"          {metrics.ram_used_gb:.1f} / {metrics.ram_total_gb:.1f} GB\n",
        style="dim",
    )
    ram_max = max(metrics.ram_total_gb, 1.0)
    content.append_text(_render_sparkline(history.ram, max_val=ram_max))
    content.append("\n\n")

    # Wired memory bar + sparkline + warning
    wired_percent = (
        (metrics.wired_gb / metrics.ram_total_gb * 100.0)
        if metrics.ram_total_gb > 0
        else 0.0
    )
    wired_bar = _render_bar("Wired", wired_percent, f"{metrics.wired_gb:.1f}G")
    content.append_text(wired_bar)
    content.append("\n")

    # Wired sparkline (scale to ram_total for percentage display)
    wired_max = max(metrics.ram_total_gb, 1.0)
    content.append_text(_render_sparkline(history.wired, max_val=wired_max))
    content.append("\n")

    # Wired warning — context-aware based on server process status
    if metrics.server_running:
        # Server is running: high wired is expected (model loaded in Metal)
        content.append(f"  Serving (process active)\n", style="bold cyan")
    elif metrics.wired_gb < WIRED_NORMAL_MAX:
        content.append("  Normal (< 10 GB)\n", style="green")
    elif metrics.wired_gb < WIRED_WARNING_MAX:
        # No server but wired 10-50 GB — suspicious
        content.append("  ⚠ No server but wired elevated\n", style="bold yellow")
    else:
        # No server but wired > 50 GB — definite leak
        if blink_on:
            content.append(
                "  LEAK DETECTED — REBOOT NEEDED\n",
                style="bold red blink",
            )
        else:
            content.append(
                "  LEAK DETECTED — REBOOT NEEDED\n",
                style="bold red",
            )
    content.append("\n")

    # Swap
    if metrics.swap_gb > 0.001:
        content.append(f"  Swap   {metrics.swap_gb:.1f} GB\n", style="bold red")
    else:
        content.append(f"  Swap   {metrics.swap_gb:.1f} GB\n", style="dim")
    content.append("\n")

    # RDMA status
    if metrics.rdma_enabled is True:
        content.append("  RDMA   ", style="dim")
        content.append("✓ enabled\n", style="bold green")
    elif metrics.rdma_enabled is False:
        content.append("  RDMA   ", style="dim")
        content.append("✗ disabled\n", style="bold red")
    else:
        content.append("  RDMA   ", style="dim")
        content.append("? unknown\n", style="dim yellow")

    # RDMA IP
    if metrics.rdma_ip:
        content.append(f"  TB5    en5  {metrics.rdma_ip}\n", style="dim")
    else:
        content.append("  TB5    en5  N/A\n", style="dim")

    # Uptime
    uptime_str = _format_uptime(metrics.uptime_seconds)
    content.append(f"  Uptime {uptime_str}\n", style="dim")

    # Panel title with status
    status_text = Text()
    status_text.append(f" {host} ", style="bold")
    status_text.append("● ONLINE", style="bold green")

    return Panel(
        content,
        title=status_text,
        border_style="green",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class Dashboard:
    """Main dashboard that polls nodes and renders the Rich Live display."""

    def __init__(
        self,
        pollers: list[tuple[str, NodePoller]],
        interval: float = 1.0,
    ) -> None:
        self.pollers = pollers
        self.interval = interval
        self.histories: dict[str, MetricsHistory] = {
            host: MetricsHistory() for host, _ in pollers
        }
        self.latest_metrics: dict[str, NodeMetrics] = {
            host: NodeMetrics() for host, _ in pollers
        }
        self._blink_counter = 0

    def _poll_all(self) -> dict[str, NodeMetrics]:
        """Poll all nodes in parallel and return results."""
        results: dict[str, NodeMetrics] = {}
        with ThreadPoolExecutor(max_workers=len(self.pollers)) as executor:
            futures = {
                executor.submit(poller.poll): host
                for host, poller in self.pollers
            }
            for future in as_completed(futures):
                host = futures[future]
                try:
                    results[host] = future.result()
                except Exception:
                    results[host] = NodeMetrics(online=False, error="Poll failed")
        return results

    def _build_layout(self) -> Layout:
        """Build the full dashboard layout."""
        self._blink_counter += 1
        blink_on = (self._blink_counter % 2) == 0

        # Create node panels
        panels = []
        for host, _ in self.pollers:
            metrics = self.latest_metrics.get(host, NodeMetrics())
            history = self.histories[host]
            panel = _render_node_panel(host, metrics, history, blink_on)
            panels.append(panel)

        # Build layout
        layout = Layout()

        if len(panels) == 1:
            main_layout = Layout(panels[0], name="main")
        else:
            node_layouts = []
            for i, panel in enumerate(panels):
                node_layouts.append(Layout(panel, name=f"node{i}"))
            main_layout = Layout(name="main")
            main_layout.split_row(*node_layouts)

        # Footer
        now = datetime.now().strftime("%H:%M:%S")
        footer_text = Text()
        footer_text.append(f"  Last updated: {now}", style="dim")
        footer_text.append(f"  |  Refresh: {self.interval:.0f}s", style="dim")
        footer_text.append("  |  Press Ctrl+C to exit", style="dim")
        footer_layout = Layout(footer_text, name="footer", size=1)

        root = Layout()
        root.split_column(main_layout, footer_layout)

        return root

    def run(self) -> None:
        """Run the dashboard in a Rich Live display loop."""
        console = Console()

        # Initial poll
        results = self._poll_all()
        for host, metrics in results.items():
            self.latest_metrics[host] = metrics
            self.histories[host].record(metrics)

        layout = self._build_layout()

        try:
            with Live(
                layout,
                console=console,
                refresh_per_second=2,
                screen=True,
            ) as live:
                while True:
                    time.sleep(self.interval)

                    # Poll all nodes
                    results = self._poll_all()
                    for host, metrics in results.items():
                        self.latest_metrics[host] = metrics
                        self.histories[host].record(metrics)

                    # Rebuild and update layout
                    layout = self._build_layout()
                    live.update(layout)

        except KeyboardInterrupt:
            pass

        console.print("\n[dim]Dashboard stopped.[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time cluster monitoring dashboard for Mac Studio nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --node hwstudio1 --node hwstudio2
  %(prog)s --node hwstudio1 --node hwstudio2 --interval 5
  %(prog)s --node hwstudio1 --node hwstudio2 --proxy hwstudio1:hwstudio2
""",
    )
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        metavar="HOST",
        help="SSH hostname to monitor (can be repeated).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Polling interval in seconds (default: 1).",
    )
    parser.add_argument(
        "--proxy",
        action="append",
        default=[],
        metavar="PROXY_HOST:TARGET_HOST",
        help=(
            "SSH proxy routing (can be repeated). "
            "Format: proxy_host:target_host  "
            "Means: reach target_host by SSHing through proxy_host first."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Parse proxy mappings: target_host -> proxy_host
    proxy_map: dict[str, str] = {}
    for spec in args.proxy:
        parts = spec.split(":")
        if len(parts) != 2:
            print(
                f"Error: Invalid --proxy format: {spec!r}  "
                f"(expected proxy_host:target_host)",
                file=sys.stderr,
            )
            return 1
        proxy_host, target_host = parts
        proxy_map[target_host] = proxy_host

    # Create pollers
    pollers: list[tuple[str, NodePoller]] = []
    for host in args.node:
        proxy = proxy_map.get(host)
        pollers.append((host, NodePoller(host, proxy=proxy)))

    # Run dashboard
    dashboard = Dashboard(pollers, interval=args.interval)
    dashboard.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
