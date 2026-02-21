# tp_batch_inference Test Results

**Date:** 2026-02-21 08:31:59 UTC

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
| 1 | PASS | 34 | 256 | 2981.0 | 11.4 | 12.6 | 23.30 |

## 4 Concurrent Requests

| # | Status | Prompt Tokens | Completion Tokens | TTFT (ms) | Prefill tok/s | Decode tok/s | Total Time (s) |
|---|--------|---------------|-------------------|-----------|---------------|--------------|----------------|
| 1 | PASS | 38 | 256 | 1274.2 | 29.8 | 7.0 | 37.61 |
| 2 | PASS | 40 | 256 | 2624.9 | 15.2 | 7.4 | 37.34 |
| 3 | PASS | 42 | 256 | 2013.5 | 20.9 | 7.2 | 37.52 |
| 4 | PASS | 47 | 256 | 1274.8 | 36.9 | 7.1 | 37.48 |

## 8 Concurrent Requests

| # | Status | Prompt Tokens | Completion Tokens | TTFT (ms) | Prefill tok/s | Decode tok/s | Total Time (s) |
|---|--------|---------------|-------------------|-----------|---------------|--------------|----------------|
| 1 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 314.60 |
| 2 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 314.96 |
| 3 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 314.62 |
| 4 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 314.61 |
| 5 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 314.57 |
| 6 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 315.13 |
| 7 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 315.09 |
| 8 | FAIL `TimeoutError: timed out` | 0 | 0 | 0.0 | 0.0 | 0.0 | 314.57 |

## Summary

- **Working:** YES
- **Total requests:** 13
- **Successful:** 5
- **Failed:** 8
- **Avg TTFT:** 2033.7 ms
- **Avg prefill tok/s:** 22.8
- **Avg decode tok/s:** 8.3
- **Total output tokens:** 1280
- **Avg per-request throughput:** 7.4 tok/s

## Errors

1. Test 3 (8 concurrent): request failed - TimeoutError: timed out
2. Test 3 (8 concurrent): request failed - TimeoutError: timed out
3. Test 3 (8 concurrent): request failed - TimeoutError: timed out
4. Test 3 (8 concurrent): request failed - TimeoutError: timed out
5. Test 3 (8 concurrent): request failed - TimeoutError: timed out
6. Test 3 (8 concurrent): request failed - TimeoutError: timed out
7. Test 3 (8 concurrent): request failed - TimeoutError: timed out
8. Test 3 (8 concurrent): request failed - TimeoutError: timed out

## Server Logs (excerpt)

### hwstudio1

```
(no log available: rc=0)
```

### hwstudio2

```
(no log available: rc=1)
```

