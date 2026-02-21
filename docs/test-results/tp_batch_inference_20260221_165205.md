# tp_batch_inference Test Results

**Date:** 2026-02-21 07:52:05 UTC

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
(no log available: rc=0)
```

### hwstudio2

```
(no log available: rc=1)
```

