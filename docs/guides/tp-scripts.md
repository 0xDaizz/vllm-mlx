<!-- SPDX-License-Identifier: Apache-2.0 -->

# Distributed Server Management Scripts

Three scripts manage the full lifecycle of a tensor-parallel (TP) distributed server across Mac Studio nodes connected via Thunderbolt 5 RDMA.

| Script | Purpose | When to run |
|--------|---------|-------------|
| `scripts/rdma_setup.py` | Configure RDMA networking | Once after each reboot |
| `scripts/tp_start.py` | Launch distributed server | Each time you start serving |
| `scripts/tp_stop.py` | Shut down distributed server | When you're done serving |

### Operational Flow

```
 Reboot
   │
   ▼
 rdma_setup.py          Configure IPs, IBV config, verify connectivity
   │
   ▼
 tp_start.py            Pre-flight checks → launch rank 0 → launch rank 1 → health check
   │
   ▼
 (server running)       Accepting requests on hwstudio1:8000
   │
   ▼
 tp_stop.py             SIGTERM → wait → SIGKILL if needed → verify cleanup
```

---

## Prerequisites

- **Python 3.10+** at `/opt/homebrew/bin/python3.14` on all nodes
- **SSH key-based authentication** to all nodes (no password prompts)
- **RDMA setup completed** — see [RDMA Setup Guide](rdma-setup.md)

---

## rdma_setup.py

Configures static IPs, writes the JACCL-compatible IBV device matrix, and verifies RDMA connectivity. Run once after every reboot.

```bash
python scripts/rdma_setup.py \
    --node hwstudio1:en5:10.254.0.5 \
    --node hwstudio2:en5:10.254.0.6 \
    --netmask 30
```

For full details, flags, and troubleshooting, see [RDMA Setup Guide](rdma-setup.md).

---

## tp_start.py

Launches the distributed vllm-mlx server across multiple nodes. Handles rank ordering, environment setup, health checks, and log management.

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--backend` | choice | `jaccl` | Distributed backend: `jaccl`, `ring`, `any` |
| `--hostfile` | path | — | Path to hostfile JSON (alternative to `--hosts`) |
| `--hosts` | list | — | Space-separated list of SSH hostnames |
| `-n` / `--num-ranks` | int | — | Number of ranks (inferred from hosts if omitted) |
| `--env` | KEY=VALUE | — | Extra environment variable passed to each rank (repeat flag for multiple vars) |
| `--timeout` | int | `600` | Startup timeout in seconds |
| `--skip-checks` | flag | — | Skip all pre-flight checks |
| `--foreground` | flag | — | Run in foreground (no nohup, logs to stdout) |
| `--log-dir` | path | `/tmp` | Directory for log files |
| `-- args...` | — | — | Everything after `--` is passed to `vllm-mlx serve` |

### Usage Examples

**Basic two-node start with `--hosts`:**

```bash
python scripts/tp_start.py --backend jaccl --hosts hwstudio1 hwstudio2 \
    -- --model ~/models/Kimi-K2.5-4bit --port 8000 --continuous-batching
```

**Using a hostfile:**

```json
{
    "backend": "jaccl",
    "envs": ["MLX_METAL_FAST_SYNCH=1"],
    "hosts": [
        {"ssh": "hwstudio1", "ips": ["10.254.0.5"], "rdma": [null, "rdma_en5"]},
        {"ssh": "hwstudio2", "ips": ["10.254.0.6"], "rdma": ["rdma_en5", null]}
    ]
}
```

Each host entry has:
- `ssh` — hostname or IP used for SSH connections
- `ips` — list of IPs for this host (used by JACCL for data transport)
- `rdma` — NxN matrix row: RDMA device name for each peer, `null` for self

```bash
python scripts/tp_start.py --hostfile hostfile.json \
    -- --model ~/models/Kimi-K2.5-4bit --port 8000
```

**Foreground mode (for debugging):**

```bash
python scripts/tp_start.py --backend jaccl --hosts hwstudio1 hwstudio2 \
    --foreground \
    -- --model ~/models/Moonlight-16B-A3B --port 8000
```

Output streams directly to the terminal instead of log files, making it easier to spot errors during development.

**Passing extra environment variables:**

```bash
python scripts/tp_start.py --backend jaccl --hosts hwstudio1 hwstudio2 \
    --env MLX_METAL_FAST_SYNCH=1 \
    --env PYTHONUNBUFFERED=1 \
    -- --model ~/models/Kimi-K2.5-4bit --port 8000
```

### Pre-flight Checks

Before launching ranks, `tp_start.py` runs these checks on every node (skip with `--skip-checks`):

| Check | Failure condition | Action |
|-------|-------------------|--------|
| Wired memory | > 23M pages (~350GB) | Abort — reboot required (Metal memory leak) |
| Existing processes | `pgrep -f 'vllm_mlx.distributed_launcher'` finds running processes | Abort — run `tp_stop.py` first |
| Port 8000 | Already bound | Abort — another server is running |
| Port 32323 | Already bound (jaccl backend only) | Abort — JACCL coordinator conflict |
| RDMA config file | `/tmp/mlx_ibv_devices.json` missing on any node (jaccl backend only) | Abort — run `rdma_setup.py` first |

### Log Files

Logs are written to `--log-dir` (default: `/tmp`):

| File | Contents |
|------|----------|
| `tp_server.log` | Combined server output from all ranks |

In foreground mode, no log files are created; all output goes to stdout/stderr.

### Troubleshooting

**Startup timeout:**
1. Check the log file: `tail -f /tmp/tp_server.log`
2. Verify RDMA connectivity: `ping -c 1 10.254.0.6` from hwstudio1
3. Ensure `/tmp/mlx_ibv_devices.json` exists on both nodes

**"Wired memory too high":**
- Metal memory was not released from a previous session (likely after SIGKILL).
- Reboot the affected node. There is no other fix.

**"Port already in use":**
- Run `tp_stop.py` to clean up the previous session.
- If that fails, manually kill processes: `ssh hwstudio1 "pkill -f vllm"`

**JACCL coordinator error (`Recv failed with errno=2`):**
- A stale coordinator or test connection is occupying port 32323.
- Never pre-test port 32323 with `nc`, `curl`, or similar tools — the coordinator misinterprets test connections as real rank connections.

---

## tp_stop.py

Gracefully shuts down the distributed server across all nodes. Uses SIGTERM first, escalates to SIGKILL only if necessary.

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hostfile` | path | — | Path to hostfile JSON (alternative to `--hosts`) |
| `--hosts` | list | — | Space-separated list of SSH hostnames |
| `--force` | flag | — | Skip SIGTERM, go straight to SIGKILL |
| `--timeout` | int | `15` | Seconds to wait for graceful shutdown before SIGKILL |

### Usage Examples

**Graceful shutdown:**

```bash
python scripts/tp_stop.py --hosts hwstudio1 hwstudio2
```

Sends SIGTERM to server processes on both nodes simultaneously, waits up to 15 seconds, then verifies processes are gone.

**Force kill (last resort):**

```bash
python scripts/tp_stop.py --hosts hwstudio1 hwstudio2 --force
```

Immediately sends SIGKILL. Use only when SIGTERM fails.

### Post-shutdown Checks

After stopping, the script automatically checks:

1. **Wired memory** — if pages exceed the safe threshold, it prints a reboot warning.
2. **JACCL cooldown** — if SIGKILL was used, it advises waiting 30 seconds before restarting to allow RDMA resources to be reclaimed by the kernel.

---

## Complete Workflow

A typical session from reboot to shutdown:

```bash
# 1. Configure RDMA (once after reboot)
python scripts/rdma_setup.py \
    --node hwstudio1:en5:10.254.0.5 \
    --node hwstudio2:en5:10.254.0.6 \
    --netmask 30

# 2. Start the distributed server
python scripts/tp_start.py --backend jaccl --hosts hwstudio1 hwstudio2 \
    -- --model ~/models/Kimi-K2.5-4bit --port 8000 --continuous-batching

# 3. Verify the server is running
curl http://hwstudio1:8000/v1/models

# 4. Use the server (from any machine that can reach hwstudio1)
curl http://hwstudio1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Kimi-K2.5-4bit", "messages": [{"role": "user", "content": "Hello"}]}'

# 5. Stop the server
python scripts/tp_stop.py --hosts hwstudio1 hwstudio2
```

---

## Important Notes

> **SIGKILL causes cascading failures.** Sending SIGKILL during active RDMA operations can trigger a Metal GPU hang, which may escalate to a kernel panic. Always use SIGTERM first (`tp_stop.py` does this by default). Only use `--force` as a last resort.

> **Check wired memory after every SIGKILL.** Run `vm_stat | grep wired` on each node. Normal is ~340,000 pages (~5GB). If you see 23,000,000+ pages, the Metal memory was not released — reboot before starting a new session. Running a server on top of leaked memory will cause immediate OOM.

> **Do not repeatedly down/up the RDMA interface.** Cycling `ifconfig en5 down && ifconfig en5 up` corrupts the kernel GID table. RDMA connections will fail with errno 96 or 22 until the node is rebooted. If RDMA stops working, reboot — do not try to fix it with interface cycling.

> **Wait 30 seconds after SIGKILL before restarting.** JACCL holds RDMA resources (QPs, CQs, memory registrations) that the kernel reclaims asynchronously. Starting a new session too quickly can hit EBUSY errors.

> **The server runs on hwstudio1:8000, not on the control machine.** All `tp_start.py` and `tp_stop.py` commands are run from the control machine (your MacBook), but the actual server process listens on hwstudio1.
