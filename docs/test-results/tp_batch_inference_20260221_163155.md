# tp_batch_inference Test Results

**Date:** 2026-02-21 07:31:55 UTC

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Kimi K2.5 (612GB MoE, 4-bit) |
| TP | 2 (hwstudio1 + hwstudio2) |
| Backend | JACCL (TB5 RDMA) |
| Continuous Batching | Enabled |
| KV Cache Quantization | FP8 |
| Status | SERVER STARTUP FAILED |

## Summary

- **Working:** NO
- **Total requests:** 0
- **Successful:** 0
- **Failed:** 0

## Errors

1. Failed to start TP server

## Server Logs (excerpt)

### hwstudio1

```
2026-02-21 16:20:12,196 WARNING vllm_mlx.distributed_launcher: RDMA device auto-detection is a placeholder. For production use, specify rdma_devices explicitly or run 'mlx.distributed_config' to discover devices.
2026-02-21 16:20:12,197 INFO vllm_mlx.distributed_launcher: Hostfile written to /var/folders/7v/ymqc8bkx6dj85w1_ny0bdtqh0000gn/T/vllm_mlx_hostfile_k8u5qno4.json
2026-02-21 16:20:12,197 INFO vllm_mlx.distributed_launcher: Launching distributed processes: /opt/homebrew/opt/python@3.14/bin/python3.14 -m mlx._distributed_utils.launch --backend jaccl --hostfile /var/folders/7v/ymqc8bkx6dj85w1_ny0bdtqh0000gn/T/vllm_mlx_hostfile_k8u5qno4.json /Users/hw/vllm-mlx/vllm_mlx/distributed_launcher.py --model ~/models/Kimi-K2.5 --continuous-batching --kv-cache-quantization --kv-cache-quantization-bits 8 --port 8000
2026-02-21 16:20:12,268 INFO vllm_mlx.distributed_launcher: Distributed launch completed successfully
```

### hwstudio2

```
(no log available: rc=1)
```

