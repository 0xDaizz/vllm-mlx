# GLM-5-4bit Benchmark — Draft Model Speculative Decoding

**Date**: 2026-02-17
**Server**: hwstudio1 — Apple M4 Ultra 512GB Unified Memory
**vllm-mlx**: v0.2.6 (develop, commit `34d5e18`)
**Client**: macbook via Tailscale (~1ms RTT)

## Configuration

| Setting | Value |
|---------|-------|
| Target Model | GLM-5-4bit (~390GB, glm_moe_dsa / deepseek_v32) |
| Draft Model | GLM-4.7-Flash-4bit (~16GB) |
| Speculative Method | draft_model, k=3 |
| KV Cache Quantization | 8-bit (group_size=64) |
| Prefix Cache | Memory-aware (hash-based, 20% RAM ~51.7GB) |
| Continuous Batching | Enabled (batch_size=16) |
| Auto-Disable Threshold | 0.0 (never disable) |
| Max Output Tokens | 256 |

## Phase 1: Sequential (concurrency=1)

**Aggregate**: 10 requests, 154.5s total, **8.3 tok/s** throughput, 0.06 RPS

| Metric | avg | p50 | p95 | p99 |
|--------|-----|-----|-----|-----|
| TTFT (ms) | 906 | 990 | 1,184 | 1,194 |
| Overall Tok/s | 8.0 | 8.2 | 9.9 | 10.8 |
| **Prefill (tok/s)** | **23.1** | **24.0** | **33.1** | **34.0** |
| **Decode (tok/s)** | **10.6** | **9.0** | **18.0** | **22.3** |

### Per-Request Detail

| # | TTFT(ms) | Prefill | Decode | Tokens | Time(s) | Prompt |
|---|----------|---------|--------|--------|---------|--------|
| 0 | 555 | 30.6 | 10.9 | 8 | 1.29 | France capital |
| 1 | 660 | 28.8 | 8.8 | 156 | 18.37 | TCP vs UDP |
| 2 | 1,081 | 15.7 | 11.6 | 256 | 23.22 | Python prime function |
| 3 | 1,168 | 22.3 | 9.3 | 105 | 12.50 | 한국 수도/인구 |
| 4 | 1,013 | 15.8 | 8.8 | 256 | 30.15 | OOP principles |
| 5 | 1,160 | 12.9 | 6.9 | 256 | 38.12 | Python vs Rust |
| 6 | 967 | 13.4 | 8.5 | 19 | 3.21 | AI haiku |
| 7 | 1,197 | 34.3 | 9.3 | 66 | 8.27 | 한국어→영어 번역 |
| 8 | 663 | 25.6 | 23.3 | 8 | 1.01 | France capital (prefix hit) |
| 9 | 600 | 31.6 | 8.8 | 156 | 18.35 | TCP vs UDP (prefix hit) |

## Phase 2: Concurrent (concurrency=4)

**Aggregate**: 10 requests, 144.1s total, **9.0 tok/s** throughput, 0.07 RPS

| Metric | avg | p50 | p95 | p99 |
|--------|-----|-----|-----|-----|
| TTFT (ms) | 3,607 | 3,899 | 6,559 | 7,271 |
| Overall Tok/s | 2.2 | 2.2 | 4.2 | 4.6 |
| **Prefill (tok/s)** | **14.6** | **7.0** | **53.0** | **72.7** |
| **Decode (tok/s)** | **2.6** | **2.3** | **4.6** | **5.3** |

### Per-Request Detail

| # | TTFT(ms) | Prefill | Decode | Tokens | Time(s) | Prompt |
|---|----------|---------|--------|--------|---------|--------|
| 0 | 740 | 23.0 | 1.6 | 8 | 5.69 | France capital |
| 1 | 1,791 | 10.6 | 2.7 | 256 | 96.71 | TCP vs UDP |
| 2 | 1,805 | 9.4 | 3.5 | 256 | 74.79 | Python prime |
| 3 | 335 | 77.6 | 2.7 | 36 | 13.45 | 한국 수도/인구 |
| 4 | 5,470 | 2.9 | 2.2 | 256 | 120.54 | OOP principles |
| 5 | 5,420 | 2.8 | 2.4 | 256 | 112.69 | Python vs Rust |
| 6 | 7,449 | 1.7 | 1.5 | 19 | 20.10 | AI haiku |
| 7 | 4,012 | 10.2 | 1.9 | 17 | 12.97 | 한국어→영어 번역 |
| 8 | 3,785 | 4.5 | 1.5 | 8 | 9.24 | France capital |
| 9 | 5,260 | 3.6 | 5.5 | 182 | 38.20 | TCP vs UDP |

## Phase 3: Prefix Cache Test

| Prompt | 1st TTFT(ms) | 2nd TTFT(ms) | Speedup |
|--------|-------------|-------------|---------|
| France capital | 674 | 683 | 0.99x |
| TCP vs UDP | 744 | 615 | **1.21x** |
| Python prime | 723 | 604 | **1.20x** |

## Key Observations

1. **Decode throughput**: avg 10.6 tok/s sequential (peak 23.3), drops to 2.6 tok/s at c=4 — memory bandwidth saturation
2. **Prefill throughput**: avg 23.1 tok/s sequential — reasonable for 390GB MoE model
3. **TTFT**: avg 906ms sequential, 3.6s at c=4
4. **Aggregate throughput**: 8.3 tok/s (seq) → 9.0 tok/s (c=4) — 1.08x gain limited by bandwidth
5. **Prefix cache**: 1.2x TTFT speedup on longer prompts, minimal on short prompts
6. **Engine error fixed**: `mx` variable shadowing bug (commit `34d5e18`) eliminated `mx.clear_cache()` failures, improving stability and preventing Metal memory leaks

## Bug Fix Applied During Benchmark

**Commit 34d5e18**: `engine_core.py` had a conditional `import mlx.core as mx` inside `_engine_loop()` that made Python treat `mx` as a local variable. In non-distributed mode, the import never executed, causing `mx.clear_cache()` to fail every engine step with "cannot access local variable 'mx'". This was non-fatal (caught by try/except) but:
- Prevented Metal cache cleanup after finished requests
- Generated hundreds of ERROR log lines per session
- May have contributed to performance degradation over time

## Environment Notes

- CacheList compatibility fixes applied (commit `35e648f`) for deepseek_v32/glm_moe_dsa models
- GLM-5 uses CacheList(KVCache(), KVCache()) per layer — required _inner_cache() helper
- 8-bit KV cache quantization reduces memory usage ~50%
- Model occupies ~390GB + draft ~16GB, leaving ~100GB for KV cache on 512GB unified memory
