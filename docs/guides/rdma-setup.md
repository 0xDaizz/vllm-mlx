# RDMA Network Setup Tool

A CLI tool for configuring Thunderbolt 5 RDMA networking on remote Mac Studio nodes via SSH. Use this after a macOS reboot or reinitialization to restore static IP addresses, IBV device configuration, and verify RDMA connectivity.

---

## Quick Start

The most common scenario: two Mac Studios connected via a single Thunderbolt 5 cable.

```bash
python scripts/rdma_setup.py \
    --node hwstudio1:en5:10.254.0.5 \
    --node hwstudio2:en5:10.254.0.6 \
    --netmask 30
```

Expected output:

```
RDMA Network Setup
==================

[1/2] Configuring hwstudio1 (en5 -> 10.254.0.5/30)
  ✓ en5 not in any bridge
  ✓ Static IP set: 10.254.0.5/30
  ✓ IBV config written: /tmp/mlx_ibv_devices.json
  ✓ RDMA: enabled
  ✓ IBV devices: rdma_en5

[2/2] Configuring hwstudio2 (en5 -> 10.254.0.6/30)
  ✓ en5 not in any bridge
  ✓ Static IP set: 10.254.0.6/30
  ✓ IBV config written: /tmp/mlx_ibv_devices.json
  ✓ RDMA: enabled
  ✓ IBV devices: rdma_en5

Connectivity Test
=================
  hwstudio1 -> hwstudio2: ✓ 0.12ms
  hwstudio2 -> hwstudio1: ✓ 0.11ms

Summary
=======
  hwstudio1      en5   10.254.0.5/30      RDMA:enabled   ✓
  hwstudio2      en5   10.254.0.6/30      RDMA:enabled   ✓
```

---

## CLI Reference

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--node SSH_HOST:IFACE:RDMA_IP` | Yes | -- | Node specification. Can be repeated for each machine. |
| `--proxy PROXY_HOST:TARGET_HOST` | No | -- | SSH proxy routing. Can be repeated. |
| `--netmask BITS` | No | `30` | CIDR prefix length for the subnet mask. |
| `--dry-run` | No | `false` | Print commands without executing them. |
| `-v` / `--verbose` | No | `false` | Enable debug logging. |

### Format Specifications

**`--node`** format: `ssh_host:interface:rdma_ip`

| Field | Description | Example |
|-------|-------------|---------|
| `ssh_host` | SSH-reachable hostname or IP | `hwstudio1` |
| `interface` | macOS network interface for TB5 | `en5`, `en3` |
| `rdma_ip` | Static IP to assign | `10.254.0.5` |

**`--proxy`** format: `proxy_host:target_host`

| Field | Description | Example |
|-------|-------------|---------|
| `proxy_host` | Intermediate SSH host to tunnel through | `hwstudio1` |
| `target_host` | Final destination host | `hwstudio2` |

When a proxy is configured, SSH commands to `target_host` are tunneled:
`ssh proxy_host "ssh target_host '<command>'"`.

---

## What the Script Does

The script runs a six-phase configuration sequence on each node, then verifies connectivity between all pairs.

### Phase 1: Bridge Removal

After macOS reinitialization (Recovery Mode), Thunderbolt interfaces are sometimes placed into `bridge0`. The script scans `bridge0` through `bridge9` and removes the target interface from any bridge it finds.

```bash
# What runs on the remote host
ifconfig bridge0                        # Check if bridge exists
sudo ifconfig bridge0 deletem en5       # Remove interface from bridge
```

### Phase 2: Static IP Assignment

Assigns the specified RDMA IP address with the given netmask to the interface.

```bash
sudo ifconfig en5 10.254.0.5 netmask 255.255.255.252
```

> **Note:** A `/30` netmask (`255.255.255.252`) provides exactly 2 usable host addresses per subnet, which is ideal for point-to-point Thunderbolt links.

### Phase 3: IBV Config Creation

Writes `/tmp/mlx_ibv_devices.json` on each node in JACCL-compatible NxN matrix format. Each entry `matrix[i][j]` is the RDMA device that node `i` uses to reach node `j`, or `null` when `i == j`. The device name is derived from the interface name (e.g., `en5` becomes `rdma_en5`).

```json
[[null, "rdma_en5"], ["rdma_en5", null]]
```

### Phase 4: RDMA Status Check

Runs `rdma_ctl status` to verify RDMA is enabled at the OS level.

### Phase 5: IBV Devices Check

Runs `ibv_devices` to list available InfiniBand Verbs devices. The target device (e.g., `rdma_en5`) should appear in the output.

### Phase 6: Connectivity Test

Pings each node's RDMA IP from every other node. Reports round-trip latency on success. Typical TB5 latency is under 0.2ms.

---

## Examples

### Two-Node Setup (Basic)

Direct SSH access to both machines from the host running the script:

```bash
python scripts/rdma_setup.py \
    --node hwstudio1:en5:10.254.0.5 \
    --node hwstudio2:en5:10.254.0.6 \
    --netmask 30
```

### Two-Node with SSH Proxy

When `hwstudio2` is not directly reachable (e.g., only accessible through `hwstudio1`):

```bash
python scripts/rdma_setup.py \
    --node hwstudio1:en5:10.254.0.5 \
    --node hwstudio2:en5:10.254.0.6 \
    --netmask 30 \
    --proxy hwstudio1:hwstudio2
```

### Using an Alternative Interface

If `en5` is damaged, switch to `en3`:

```bash
python scripts/rdma_setup.py \
    --node hwstudio1:en3:10.254.0.1 \
    --node hwstudio2:en3:10.254.0.2 \
    --netmask 30
```

### Three-Node Setup

For a three-node mesh, each node needs a separate TB5 interface per peer. Each point-to-point link gets its own `/30` subnet:

```bash
python scripts/rdma_setup.py \
    --node machine-a:en3:10.254.0.1 \
    --node machine-b:en3:10.254.0.2 \
    --node machine-a:en5:10.254.0.5 \
    --node machine-c:en5:10.254.0.6 \
    --node machine-b:en5:10.254.0.9 \
    --node machine-c:en3:10.254.0.10 \
    --netmask 30
```

> **Tip:** For 3+ nodes, plan your subnets carefully. Each point-to-point link needs its own `/30` subnet (4 addresses, 2 usable).

### Dry-Run Mode

Preview all SSH commands without executing them:

```bash
python scripts/rdma_setup.py \
    --node hwstudio1:en5:10.254.0.5 \
    --node hwstudio2:en5:10.254.0.6 \
    --netmask 30 \
    --dry-run
```

Output shows each command prefixed with `[dry-run]`:

```
[1/2] Configuring hwstudio1 (en5 -> 10.254.0.5/30)
    [dry-run] ssh hwstudio1 ifconfig bridge0 2>/dev/null
    [dry-run] ssh hwstudio1 sudo ifconfig en5 10.254.0.5 netmask 255.255.255.252
    [dry-run] ssh hwstudio1 printf '%s' '[[null, "rdma_en5"], ["rdma_en5", null]]' > /tmp/mlx_ibv_devices.json
    ...
```

---

## Post-Reboot Checklist

### After a Normal Reboot

Static IPs and `/tmp/mlx_ibv_devices.json` are lost on reboot. RDMA remains enabled.

1. Run the setup script:
   ```bash
   python scripts/rdma_setup.py \
       --node hwstudio1:en5:10.254.0.5 \
       --node hwstudio2:en5:10.254.0.6 \
       --netmask 30
   ```
2. Verify the summary shows `RDMA:enabled` and ping latency for all pairs.

### After macOS Reinitialization (Recovery Mode)

A full OS reinitialization disables RDMA and may place TB5 interfaces into a bridge.

1. Boot each machine into **Recovery Mode** (hold power button until startup options appear).
2. Open **Terminal** from the Utilities menu.
3. Run:
   ```bash
   rdma_ctl enable
   ```
4. Reboot normally.
5. Run the setup script (it will handle bridge removal automatically):
   ```bash
   python scripts/rdma_setup.py \
       --node hwstudio1:en5:10.254.0.5 \
       --node hwstudio2:en5:10.254.0.6 \
       --netmask 30
   ```

### What Is Lost on Reboot

| Item | Lost on reboot? | Restored by script? |
|------|:---------------:|:-------------------:|
| Static IP addresses | Yes | Yes |
| `/tmp/mlx_ibv_devices.json` | Yes | Yes |
| RDMA enabled status | No | N/A (persists) |
| Bridge membership | Only after reinit | Yes (removes) |

---

## Troubleshooting

### "Bridge not found"

```
  ✓ en5 not in any bridge
```

This is normal. It means the interface was already clean and not a member of any bridge. No action needed.

### "RDMA: disabled"

```
  ⚠ RDMA: disabled (enable via Recovery Mode)
```

RDMA is not enabled at the OS level. You must boot into Recovery Mode and run `rdma_ctl enable`. See [Post-Reboot Checklist](#after-macos-reinitialization-recovery-mode).

### "IBV devices: none"

```
  ✗ IBV devices: none (RDMA disabled)
```

No InfiniBand Verbs devices are visible. Causes:

1. RDMA is disabled (see above)
2. Thunderbolt cable is not connected
3. The interface name is wrong (try `en3` instead of `en5` or vice versa)

### "Ping failed"

```
  hwstudio1 -> hwstudio2: ✗ no response
```

Possible causes:

| Cause | Fix |
|-------|-----|
| Wrong interface | Verify with `ifconfig` on each node |
| Cable disconnected | Re-seat the Thunderbolt 5 cable |
| IP not set | Check for errors in the static IP step above |
| Wrong netmask | Ensure both nodes share the same `/30` subnet |

### SSH Proxy Errors

```
ERROR: Invalid --proxy format: 'hwstudio1-hwstudio2'  (expected proxy_host:target_host)
```

Use a colon `:` as the separator, not a hyphen: `--proxy hwstudio1:hwstudio2`.

If proxy commands time out, verify that the proxy host can reach the target host via SSH:

```bash
ssh hwstudio1 "ssh hwstudio2 hostname"
```

---

## Important Notes

> **Warning:** IP addresses must NOT conflict with your regular network. Use `10.254.0.x` for RDMA, NOT `192.168.x.x`. Conflicting subnets cause the OS to route internet traffic over the TB5 cable, breaking Wi-Fi and Tailscale connectivity.

> **Note:** A `/30` netmask provides exactly 2 usable addresses per subnet (e.g., `10.254.0.5` and `10.254.0.6` in the `10.254.0.4/30` subnet). This is ideal for point-to-point Thunderbolt links.

> **Note:** The script uses `sudo` for `ifconfig` commands on the remote hosts. Either configure passwordless sudo on each machine, or be prepared to enter the password interactively during setup.

> **Tip:** The script is idempotent -- it is safe to run multiple times. Re-running will re-apply the same configuration and re-verify connectivity.
