# tp_batch_inference Test Results

**Date:** 2026-02-21 08:37:01 UTC

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Kimi K2.5 (612GB MoE, 4-bit) |
| TP | 2 (hwstudio1 + hwstudio2) |
| Backend | JACCL (TB5 RDMA) |
| Continuous Batching | Enabled |
| KV Cache Quantization | FP8 |

## Single Request

| # | Status | Prompt Tokens | Completion Tokens | TTFT (ms) | Prefill tok/s | Decode tok/s | Total Time (s) |
|---|--------|---------------|-------------------|-----------|---------------|--------------|----------------|
| 1 | PASS | 34 | 256 | 1101.0 | 30.9 | 9.9 | 26.86 |

## 4 Concurrent Requests

| # | Status | Prompt Tokens | Completion Tokens | TTFT (ms) | Prefill tok/s | Decode tok/s | Total Time (s) |
|---|--------|---------------|-------------------|-----------|---------------|--------------|----------------|
| 1 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 341.62 |
| 2 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 341.65 |
| 3 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 341.45 |
| 4 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 340.87 |

## 8 Concurrent Requests

| # | Status | Prompt Tokens | Completion Tokens | TTFT (ms) | Prefill tok/s | Decode tok/s | Total Time (s) |
|---|--------|---------------|-------------------|-----------|---------------|--------------|----------------|
| 1 | FAIL `ConnectionResetError: [Errno 54] Connection reset by peer` | 0 | 0 | 0.0 | 0.0 | 0.0 | 120.35 |
| 2 | FAIL `ConnectionResetError: [Errno 54] Connection reset by peer` | 0 | 0 | 0.0 | 0.0 | 0.0 | 120.35 |
| 3 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 300.07 |
| 4 | FAIL `ConnectionResetError: [Errno 54] Connection reset by peer` | 0 | 0 | 0.0 | 0.0 | 0.0 | 120.35 |
| 5 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 300.32 |
| 6 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 300.19 |
| 7 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 300.32 |
| 8 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 300.07 |

## Summary

- **Working:** YES
- **Total requests:** 13
- **Successful:** 1
- **Failed:** 12
- **Avg TTFT:** 1101.0 ms
- **Avg prefill tok/s:** 30.9
- **Avg decode tok/s:** 9.9
- **Total output tokens:** 256
- **Avg per-request throughput:** 9.5 tok/s

## Errors

1. Test 2 (4 concurrent): request failed - TimeoutError: timed out
2. Test 2 (4 concurrent): request failed - TimeoutError: timed out
3. Test 2 (4 concurrent): request failed - TimeoutError: timed out
4. Test 2 (4 concurrent): request failed - TimeoutError: timed out
5. Test 3 (8 concurrent): request failed - ConnectionResetError: [Errno 54] Connection reset by peer
6. Test 3 (8 concurrent): request failed - ConnectionResetError: [Errno 54] Connection reset by peer
7. Test 3 (8 concurrent): request failed - TimeoutError: timed out
8. Test 3 (8 concurrent): request failed - ConnectionResetError: [Errno 54] Connection reset by peer
9. Test 3 (8 concurrent): request failed - TimeoutError: timed out
10. Test 3 (8 concurrent): request failed - TimeoutError: timed out
11. Test 3 (8 concurrent): request failed - TimeoutError: timed out
12. Test 3 (8 concurrent): request failed - TimeoutError: timed out

## Server Logs (excerpt)

### hwstudio1

```
(no log available: rc=0)
```

### hwstudio2

```
(no log available: rc=1)
```

