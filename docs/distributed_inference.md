# Distributed Tensor-Parallel Inference

Run large language models across multiple Mac Studios using Thunderbolt 5 RDMA for high-bandwidth, low-latency tensor parallelism.

## Overview

Distributed tensor-parallel (TP) inference splits a model's weight matrices across multiple machines. Each machine holds a shard of every layer and participates in every forward pass, communicating partial results via `all_sum` operations over RDMA.

### Architecture

```
┌─────────────────────────────────────┐    Thunderbolt 5    ┌──────────────────────────┐
│           Rank 0 (machine-a)        │◄══════ RDMA ══════►│   Rank 1 (machine-b)     │
│                                     │     ~120 Gbps       │                          │
│  ┌─────────────────────────────┐    │                     │  ┌────────────────────┐  │
│  │   FastAPI Server (OpenAI)   │    │                     │  │   Worker Loop       │  │
│  │   Scheduler + Engine Core   │    │                     │  │   (headless)        │  │
│  │   BatchGenerator (sharded)  │    │                     │  │   BatchGenerator    │  │
│  └─────────────────────────────┘    │                     │  │   (sharded)         │  │
│                                     │                     │  └────────────────────┘  │
│  Model shard: shard 0 (layers split)│                     │  Model shard: shard 1 (layers split)│
└─────────────────────────────────────┘                     └──────────────────────────┘
```

**Rank 0** runs the full API server (FastAPI + Scheduler + EngineCore) and serves HTTP requests. **Rank 1+** are headless workers that only execute model forward passes.

### StepPlan Protocol

Every decode step follows this synchronized protocol:

1. **Rank 0** builds a `StepPlan` (inserts, removes, sampling seeds, batch fingerprint) and broadcasts it to all workers.
2. **All ranks** execute the model forward pass simultaneously. The sharded `all_sum` operations inside the model layers synchronize tensor results across RDMA.
3. **Rank 0** samples tokens from the output logits and broadcasts the sampled token IDs to all workers.
4. **All ranks** update their local batch state with the new tokens and loop back to step 1.

This ensures synchronized generation across all ranks, with Rank 0 as the authoritative source of sampled tokens.

---

## Prerequisites

### Hardware

| Requirement | Details |
|-------------|---------|
| Machines | 2 (or more) Apple Silicon Macs with Thunderbolt 5 (M4 Ultra, M3 Ultra, or any Apple Silicon Mac with Thunderbolt 5) |
| Memory | Sufficient unified memory on each machine to hold its model shard (e.g., 192GB per machine for a 300B+ parameter model) |
| Cable | Thunderbolt 5 cable connecting the two machines directly |

> **Note:** Mac Studio with M4 Ultra (512GB) or M3 Ultra (192GB) are the highest-capacity configurations.

### Software

| Requirement | Version |
|-------------|---------|
| macOS | 26.2 (Tahoe) or later -- required for RDMA over Thunderbolt |
| Python | 3.11+ |
| MLX | 0.30+ |
| mlx-lm | Latest (with `sharded_load` support) |

### Passwordless SSH

Both machines must have passwordless SSH access to each other. This is required for `mlx.launch` to spawn processes remotely.

```bash
# On machine-a, generate key if needed
ssh-keygen -t ed25519

# Copy to machine-b
ssh-copy-id <your-user>@machine-b

# Test (should not prompt for password)
ssh machine-b hostname
```

Repeat in reverse (from machine-b to machine-a) if using `mlx.launch` from either machine.

### Version Matching

All machines **must** have identical versions of:

| Package | Check command |
|---------|--------------|
| Python | `python3 --version` |
| MLX | `python3 -c "import mlx; print(mlx.__version__)"` |
| mlx-lm | `python3 -c "import mlx_lm; print(mlx_lm.__version__)"` |
| vllm-mlx | `python3 -c "import vllm_mlx; print(vllm_mlx.__version__)"` |

Mismatched versions can cause silent failures or crashes during `sharded_load()`.

### Enable RDMA (One-Time Setup Per Machine)

RDMA over Thunderbolt must be explicitly enabled from Recovery Mode on **each machine**.

1. Shut down the Mac.
2. Press and hold the power button until "Loading startup options..." appears.
3. Select **Options** to boot into Recovery Mode.
4. Open **Terminal** from the Utilities menu.
5. Run:
   ```bash
   rdma_ctl enable
   ```
6. Reboot normally.

> **Warning:** This step requires booting into Recovery Mode. It only needs to be done once per machine; the setting persists across normal reboots.

---

## Installation

Perform these steps on **both** machines.

### 1. Clone the Repository

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

> **Note:** This installs vllm-mlx in editable/development mode. For production use, `pip install vllm-mlx` is also available.

> **Tip:** It is recommended to use a Python virtual environment (`python3 -m venv .venv && source .venv/bin/activate`) to avoid package conflicts. Create the same virtual environment on both machines before installing.

> **Note:** Both machines **must** have the repository (and model weights) at the **same absolute path**. The distributed launcher and `sharded_load()` expect identical paths on all ranks.

### 2. Ensure Model Weights Are Available

Both machines must have the model weights at the **same path**. You can either:

**Option A: Download on both machines (if both have internet)**

```bash
# On both machine-a and machine-b
huggingface-cli download <org>/<model-name> --local-dir <your-model-path>
```

**Option B: Download once, transfer to the other machine**

```bash
# On machine-a (has internet)
huggingface-cli download <org>/<model-name> --local-dir <your-model-path>

# Transfer to machine-b
scp -r <your-model-path> <your-user>@machine-b:<your-model-path>
```

### 3. Transfer Wheels (If One Machine Has No Internet)

If `machine-b` has no internet access, you can transfer the Python packages:

```bash
# On machine-a: download wheels
pip download vllm-mlx -d /tmp/wheels
pip download mlx mlx-lm -d /tmp/wheels

# Transfer to machine-b
scp -r /tmp/wheels <your-user>@machine-b:/tmp/wheels

# On machine-b: install from local wheels
pip install --no-index --find-links /tmp/wheels vllm-mlx mlx mlx-lm
```

---

## RDMA Setup

### Verify RDMA Is Enabled

On each machine, check that RDMA is active:

```bash
rdma_ctl status
```

Expected output:

```
RDMA is enabled
```

### Verify RDMA Devices

```bash
ibv_devices
```

Expected output (device names may vary):

```
    device          node GUID
    ------          ---------
    rdma_en2        xxxx:xxxx:xxxx:xxxx
```

> **Note:** RDMA interfaces (`rdma_en*`) do **not** appear in `ifconfig`. They are InfiniBand-class devices visible only via `ibv_devices`. This is expected.

### Run Auto-Setup

Use the MLX distributed configuration tool to auto-discover RDMA devices and generate the hostfile:

```bash
mlx.distributed_config \
    --auto-setup \
    --over thunderbolt \
    --backend jaccl \
    --hosts machine-a,machine-b \
    --output-hostfile ~/mlx_hostfile.json
```

> **Note:** Replace `machine-a` and `machine-b` with the actual hostnames or IP addresses of your two Mac Studios.

### Hostfile Format

The generated hostfile (`~/mlx_hostfile.json`) will look like this:

```json
{
  "backend": "jaccl",
  "envs": ["MLX_METAL_FAST_SYNCH=1"],
  "hosts": [
    {
      "ssh": "machine-a",
      "ips": ["10.254.0.1"],
      "rdma": [null, "rdma_en2"]
    },
    {
      "ssh": "machine-b",
      "ips": ["10.254.0.2"],
      "rdma": ["rdma_en2", null]
    }
  ]
}
```

Key fields:

| Field | Description |
|-------|-------------|
| `backend` | `"jaccl"` for RDMA over Thunderbolt |
| `envs` | Environment variables set on all ranks (`MLX_METAL_FAST_SYNCH=1` is required) |
| `hosts[].ssh` | SSH-reachable hostname for `mlx.launch` to spawn the process |
| `hosts[].ips` | IP addresses used for communication |
| `hosts[].rdma` | RDMA device mapping -- `null` for self, device name for each peer |

The `rdma` array has one entry per host. Entry `i` in host `j`'s `rdma` array is the RDMA device on host `j` that connects to host `i`. A `null` entry means "self" (no RDMA device needed for local communication).

> **WARNING: Avoid common subnets for TB5 RDMA interfaces.**
>
> Thunderbolt 5 point-to-point links need static IP addresses. Do NOT use
> `192.168.0.x` or `192.168.1.x` — these conflict with most home/office Wi-Fi
> routers (default gateway `192.168.0.1` or `192.168.1.1`). When a TB5
> interface shares a subnet with the default gateway, the OS routes internet
> traffic over the TB5 cable instead of Wi-Fi, breaking internet connectivity
> and Tailscale on the receiving machine.
>
> **Recommended subnets** for TB5 RDMA (point-to-point, /30):
> - `10.254.0.0/30` → machine-a: `10.254.0.1`, machine-b: `10.254.0.2`
> - `172.30.0.0/30` → alternative if 10.x is already in use
>
> To configure on macOS:
> ```bash
> # On machine-a (find the TB5 interface with `ifconfig | grep -B2 rdma`)
> sudo ifconfig en3 inet 10.254.0.1 netmask 255.255.255.252
>
> # On machine-b
> sudo ifconfig en3 inet 10.254.0.2 netmask 255.255.255.252
> ```
>
> **Note:** These settings are lost on reboot. To persist, configure them in
> System Settings → Network → Thunderbolt Bridge, or add the commands to a
> login script.

---

## Verify Connectivity

Before launching the full server, verify that the two machines can communicate over RDMA.

### Create a Test Script

Save the following as `test_distributed.py` on both machines:

```python
"""Simple distributed connectivity test using MLX all_sum."""
import mlx.core as mx

group = mx.distributed.init(backend="jaccl", strict=True)
rank = group.rank()
world_size = group.size()

# Each rank contributes its rank number; the sum should equal 0+1 = 1
result = mx.distributed.all_sum(mx.array(rank), group=group)
mx.eval(result)

print(f"[Rank {rank}] all_sum result = {result.item()} (expected {world_size * (world_size - 1) // 2})")

if rank == 0:
    print(f"SUCCESS: {world_size} ranks communicating over RDMA")
```

### Run the Test

```bash
mlx.launch \
    --hostfile ~/mlx_hostfile.json \
    --backend jaccl \
    test_distributed.py
```

Expected output:

```
[Rank 0] all_sum result = 1 (expected 1)
[Rank 1] all_sum result = 1 (expected 1)
SUCCESS: 2 ranks communicating over RDMA
```

If this test succeeds, your RDMA link is working correctly.

---

## Preflight Checklist

Before launching, verify all prerequisites on **both** machines:

```bash
# 1. RDMA enabled
rdma_ctl status                    # → "RDMA is enabled"

# 2. RDMA device visible
ibv_devices                        # → shows rdma_en* device

# 3. SSH connectivity (from machine-a)
ssh machine-b hostname             # → prints hostname without password prompt

# 4. Python and packages
python3 --version                  # → same version on both
python3 -c "import mlx; print(mlx.__version__)"      # → same on both
python3 -c "import mlx_lm; print(mlx_lm.__version__)" # → same on both

# 5. Model path exists
ls <your-model-path>/config.json   # → file exists on both machines

# 6. Hostfile exists
cat ~/mlx_hostfile.json            # → valid JSON with correct IPs and RDMA devices
```

If any check fails, refer to the relevant section above before proceeding.

---

## Launch Distributed Server

### Environment Variables

The distributed launcher sets several environment variables automatically. Understanding them helps with debugging:

| Variable | Description | Example |
|----------|-------------|---------|
| `MLX_RANK` | This process's rank (set by `mlx.launch`) | `0` or `1` |
| `MLX_JACCL_COORDINATOR` | Coordinator address for JACCL group formation | `machine-a:port` |
| `MLX_IBV_DEVICES` | Path to RDMA device mapping JSON | `/tmp/mlx_ibv_devices.json` |
| `MLX_METAL_FAST_SYNCH` | Enable fast GPU synchronization (required for JACCL) | `1` |
| `MLX_DISTRIBUTED_BACKEND` | Distributed backend type | `jaccl` |
| `VLLM_MLX_DISTRIBUTED` | Signals to vllm-mlx internals that distributed mode is active | `1` |
| `VLLM_MLX_WORLD_SIZE` | Total number of ranks | `2` |

### Launch Command

The distributed launcher handles spawning processes on both machines via SSH:

```bash
python -m vllm_mlx.distributed_launcher \
    --backend jaccl \
    --hostfile ~/mlx_hostfile.json \
    -- \
    --model <your-model-path> \
    --continuous-batching \
    --host 0.0.0.0 \
    --port 8000
```

Everything after `--` is passed to the vllm-mlx server on Rank 0.

> **Note:** `--continuous-batching` is **required** for distributed mode. Without `--continuous-batching`, the StepPlan broadcast protocol will not function.

### What Happens at Launch

1. `mlx.launch` SSHes into each host listed in the hostfile and starts a Python process.
2. Each process calls `distributed_main()`, which initializes `mx.distributed` with the JACCL backend.
3. **All ranks** collectively load the model via `sharded_load()` -- this is a distributed operation that uses `all_sum` to distribute weight shards. Expect ~60-120 seconds for large models.
4. All ranks hit a barrier to confirm model loading is complete.
5. **Rank 0** starts the FastAPI server and begins accepting HTTP requests.
6. **Rank 1+** enter the `worker_loop()`, waiting for `StepPlan` broadcasts.

### Monitor Startup Logs

Watch for these key log messages:

```
[Rank 0] Loading sharded model: <your-model-path>
[Rank 1] Loading sharded model: <your-model-path>
...
[Rank 0] Sharded model loaded successfully
[Rank 1] Sharded model loaded successfully
[Rank 0] Post-load barrier passed
[Rank 1] Post-load barrier passed
[Rank 1] BatchGenerator created, entering StepPlan loop
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Once you see the Uvicorn message, the server is ready.

### Manual Launch (Alternative)

If `mlx.launch` SSH is not set up, you can start each rank manually. Start Rank 0 first, then Rank 1 after approximately 5 seconds:

**On machine-a (Rank 0):**

```bash
export MLX_RANK=0
export MLX_JACCL_COORDINATOR="machine-a:12345"
export MLX_IBV_DEVICES="/tmp/mlx_ibv_devices.json"
export MLX_METAL_FAST_SYNCH=1
export PYTHONUNBUFFERED=1

python -m vllm_mlx.distributed_launcher \
    --model <your-model-path> \
    --continuous-batching \
    --host 0.0.0.0 \
    --port 8000
```

**On machine-b (Rank 1), ~5 seconds later:**

```bash
export MLX_RANK=1
export MLX_JACCL_COORDINATOR="machine-a:12345"
export MLX_IBV_DEVICES="/tmp/mlx_ibv_devices.json"
export MLX_METAL_FAST_SYNCH=1
export PYTHONUNBUFFERED=1

python -m vllm_mlx.distributed_launcher \
    --model <your-model-path> \
    --continuous-batching
```

> **Warning:** Rank 0 (the coordinator) **must** start first. If Rank 1 starts before Rank 0, the JACCL group formation will fail.

---

## Send Requests

The distributed server exposes the standard OpenAI-compatible API on **Rank 0's port only**. Rank 1 does not serve HTTP.

### Chat Completions

```bash
curl http://machine-a:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Explain tensor parallelism in 3 sentences."}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": true
  }'
```

### Using the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://machine-a:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello from distributed inference!"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### Health Check

```bash
curl http://machine-a:8000/health
```

The response includes `"model_type"` and confirms the server is operational.

---

## Performance Expectations

Benchmarks from a real deployment with **Kimi K2.5** (612GB, MoE, 4-bit quantized) on **2x Mac Studio M4 Ultra (512GB unified memory each)** connected via Thunderbolt 5 RDMA:

### Throughput

| Metric | Value |
|--------|-------|
| Prefill speed | ~13 tok/s (~2.3s for a 30-token prompt) |
| Decode speed (single request) | ~20.9 tok/s |
| Decode step latency | 47.8 ms total |
| GPU compute per step | 44.4 ms |
| Distributed overhead per step | 3.4 ms (~7% of step time) |
| Decode speed (4 concurrent requests) | ~27 tok/s aggregate |

### Latency Characteristics

| Metric | Value |
|--------|-------|
| First request after idle (RDMA warmup) | ~1.9 s added latency |
| Steady-state inter-token latency | ~47.8 ms |
| StepPlan broadcast overhead | < 1 ms |
| Token broadcast overhead | < 1 ms |
| `all_sum` per layer (RDMA) | ~0.1 ms |

> **Note:** The first request after the server has been idle for some time incurs an approximately 1.9-second RDMA warmup penalty. Subsequent requests do not pay this cost.

### Scaling Behavior

Distributed TP is most beneficial for models that **do not fit in a single machine's memory**. For models that already fit on one machine, single-node inference will be faster due to zero communication overhead.

The ~7% distributed overhead means you retain ~93% of the theoretical 2x speedup from splitting the model across two machines. The actual benefit is being able to serve models that are too large for any single Apple Silicon machine.

---

## Troubleshooting

### RDMA Interfaces Not Showing

**Symptom:** `ibv_devices` returns no devices.

**Fix:**
1. Verify RDMA is enabled: `rdma_ctl status`
2. If it shows "disabled", re-enable from Recovery Mode (see [Prerequisites](#enable-rdma-one-time-setup-per-machine))
3. Ensure macOS 26.2 (Tahoe) or later is installed
4. Check that the Thunderbolt cable is plugged in

### JACCL EBUSY Errors

**Symptom:** Error message containing `EBUSY` when starting a distributed process.

**Fix:**
1. Kill the distributed launcher processes on both machines:
   ```bash
   pkill -f "vllm_mlx.distributed_launcher"
   ```
   This targets only the vllm-mlx distributed launcher processes, avoiding disruption to other Python programs running on the machine.
2. Wait 30 seconds for RDMA resources to be released
3. Retry the launch

### Connection Timeout

**Symptom:** Rank 1 hangs at `mx.distributed.init()` or times out.

**Fix:**
1. Verify the Thunderbolt cable is connected and both machines are powered on
2. Verify the hostfile IPs are correct: the IPs should be reachable from each machine
3. Ensure Rank 0 was started first
4. Check that no firewall rules are blocking the JACCL coordinator port

### Rank Ordering Issues

**Symptom:** Processes start but never synchronize; hangs at barrier or `sharded_load`.

**Fix:** Rank 0 (the coordinator) **must** start before all other ranks. In manual launch mode, start Rank 0 first and wait ~5 seconds before starting Rank 1.

### Thunderbolt Port Issues (Mac Studio)

**Symptom:** Intermittent connectivity or RDMA device not detected.

**Fix:** On Mac Studio, avoid using the Thunderbolt port immediately adjacent to the Ethernet port. Use one of the other Thunderbolt ports for the inter-machine cable. Once a working port is found, use it consistently.

### TB5 IP Conflict with Router

**Symptom:** One machine loses internet connectivity or Tailscale goes offline
after configuring TB5 RDMA.

**Cause:** The TB5 interface IP (e.g., `192.168.0.1/30`) conflicts with the
Wi-Fi router's gateway IP (`192.168.0.1`). The OS resolves ARP for the gateway
to the TB5 interface MAC instead of the router.

**Fix:**
```bash
# Remove conflicting IP
sudo ifconfig en3 inet 192.168.0.1 delete

# Set non-conflicting IP
sudo ifconfig en3 inet 10.254.0.1 netmask 255.255.255.252

# Flush ARP cache
sudo arp -d -a

# Verify gateway is reachable
ping -c 1 192.168.0.1  # should get router response
```

### Model Loading Fails on One Rank

**Symptom:** `sharded_load` crashes or hangs on one machine.

**Fix:**
1. Verify the model path is identical on both machines
2. Ensure both machines have the same model files (check file sizes)
3. Ensure both machines have the same version of `mlx-lm` installed

### After Reboot

After rebooting either machine:

1. Verify RDMA is still enabled: `rdma_ctl status`
2. Re-run the auto-setup to regenerate device mappings:
   ```bash
   mlx.distributed_config \
       --auto-setup \
       --over thunderbolt \
       --backend jaccl \
       --hosts machine-a,machine-b \
       --output-hostfile ~/mlx_hostfile.json
   ```
3. If using manual launch mode, ensure `/tmp/mlx_ibv_devices.json` is recreated (it is in `/tmp` and may be cleared on reboot)

### Batch Desync Errors

**Symptom:** Log messages containing `BATCH DESYNC DETECTED` or `Fingerprint mismatch`.

**Fix:** This indicates the worker's batch state has diverged from Rank 0. This is a bug -- please file an issue with the full logs from both ranks. As a workaround, restart the entire distributed server.

---

## Current Limitations

| Limitation | Details |
|------------|---------|
| **Prefix cache disabled** | Prefix caching is automatically disabled in distributed mode. Workers cannot share KV cache state across the RDMA link. Each request starts with a full prefill. |
| **Speculative decoding** | Speculative decoding in distributed mode is experimental. The n-gram proposer is supported; EAGLE/draft-model proposers are not yet available in TP mode. |
| **Tensor parallelism only** | Only tensor parallelism (TP) is supported. Pipeline parallelism (PP) is not implemented. |
| **Fully connected mesh** | JACCL requires every node to be directly cabled to every other node. For 2 nodes, this means 1 cable. For 3+ nodes, the cabling requirement grows quadratically. |
| **Maximum cluster size** | Practical limit of ~4 nodes due to Thunderbolt port count on Mac Studio (3 usable TB5 ports; each node needs N-1 ports for N nodes in a full mesh). |
| **No hot-add/remove** | Nodes cannot be added or removed while the server is running. The cluster topology is fixed at startup. |
| **Text models only** | Distributed inference currently supports text LLM models. Vision-language (mlx-vlm) and audio models are not yet supported in TP mode. |

---

## Reference: Distributed Module APIs

### `vllm_mlx.distributed_launcher`

The main entry point for distributed inference.

```python
# Programmatic launch
import os
from vllm_mlx.distributed_launcher import launch_distributed

launch_distributed(
    script_or_module="vllm_mlx.distributed_launcher",
    args=["--model", "<your-model-path>", "--continuous-batching"],
    backend="jaccl",
    hostfile=os.path.expanduser("~/mlx_hostfile.json"),
)
```

### `vllm_mlx.distributed.MLXCommunicator`

The communication layer used internally by the scheduler and workers.

| Method | Description |
|--------|-------------|
| `barrier()` | Synchronize all ranks |
| `broadcast_int(value, src=0)` | Broadcast a single integer from source rank |
| `broadcast_tensor(tensor, src=0)` | Broadcast an MLX array from source rank |
| `broadcast_object(obj, src=0)` | Broadcast any picklable Python object |
| `broadcast_step_plan(plan)` | Broadcast a `StepPlan` from Rank 0 |
| `receive_step_plan()` | Receive a `StepPlan` on a worker rank |
| `broadcast_tokens(token_ids, src=0)` | Broadcast sampled token IDs |

All broadcast operations use `all_sum` internally: the source rank contributes the real value while all other ranks contribute zeros, so the sum equals the original value on every rank.

### `vllm_mlx.distributed.StepPlan`

The plan broadcast from Rank 0 to workers each decode step.

| Field | Type | Description |
|-------|------|-------------|
| `step_id` | `int` | Monotonically increasing step counter |
| `inserts` | `list[InsertOp]` | Requests to add to the batch |
| `removes` | `list[str]` | Request IDs to remove from the batch |
| `sampling_seeds` | `dict[str, int]` | Per-request RNG seeds |
| `fingerprint` | `str` | MD5 hash of batch composition for sync verification |
| `spec_decode_plan` | `SpecDecodePlan \| None` | Speculative decoding plan (if enabled) |
