# tp_batch_inference Test Results

**Date:** 2026-02-21 07:36:50 UTC

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
2026-02-21 16:33:46,322 INFO vllm_mlx.distributed_launcher: Launching distributed processes: /opt/homebrew/opt/python@3.14/bin/python3.14 -m mlx._distributed_utils.launch --backend jaccl --hostfile /tmp/vllm_mlx_test_hostfile.json /Users/hw/vllm-mlx/vllm_mlx/distributed_launcher.py --model ~/models/Kimi-K2.5 --continuous-batching --kv-cache-quantization --kv-cache-quantization-bits 8 --port 8000
2026-02-21 16:33:46,387 INFO vllm_mlx.distributed_launcher: Distributed launch completed successfully
```

### hwstudio2

```
(no log available: rc=1)
```

