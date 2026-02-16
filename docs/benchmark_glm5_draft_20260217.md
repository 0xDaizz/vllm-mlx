# GLM-5 Draft Model Speculative Decoding Benchmark

**Date**: 2026-02-17
**Server**: Apple M4 Ultra (512GB Unified Memory)
**Client**: Remote client (network latency ~1ms)

## Configuration

| Setting | Value |
|---------|-------|
| Target Model | GLM-5-4bit (~390GB) |
| Draft Model | GLM-4.7-Flash-4bit (~16GB) |
| Speculative Method | draft_model |
| Num Speculative Tokens | 3 |
| KV Cache Quantization | 8-bit (group_size=64) |
| Prefix Cache | Memory-aware (hash-based, 20% RAM) |
| Continuous Batching | Enabled (batch_size=16) |
| Auto-Disable Threshold | 0.0 (never disable) |
| Max Tokens | 256 |

## Phase 1: Sequential (concurrency=1)

| Metric | avg | p50 | p95 | p99 |
|--------|-----|-----|-----|-----|
| TTFT (ms) | 1088.9 | 997.1 | 1807.9 | 2060.7 |
| Tok/s (overall) | 8.0 | 8.2 | 9.9 | 10.9 |
| Prefill (tok/s) | 20.2 | 20.5 | 32.6 | 38.3 |
| Decode (tok/s) | 12.0 | 9.2 | 23.6 | 23.7 |

**Aggregate**: 10 requests, 154.6s total, 8.3 tok/s throughput, 0.06 RPS

### Per-Request Detail

| # | TTFT(ms) | Prefill | Decode | Tokens | Time(s) | Prompt |
|---|----------|---------|--------|--------|---------|--------|
| 0 | 749.7 | 22.7 | 23.6 | 8 | 1.09 | What is the capital of France? |
| 1 | 796.5 | 23.9 | 8.8 | 156 | 18.49 | TCP vs UDP |
| 2 | 962.3 | 17.7 | 11.6 | 256 | 23.09 | Python prime function |
| 3 | 1185.4 | 21.9 | 9.4 | 105 | 12.36 | 한국 수도/인구 |
| 4 | 843.6 | 19.0 | 8.4 | 256 | 31.20 | OOP principles |
| 5 | 2123.9 | 7.1 | 7.3 | 256 | 37.15 | Python vs Rust |
| 6 | 1421.6 | 9.1 | 9.2 | 18 | 3.39 | AI haiku |
| 7 | 1031.9 | 39.7 | 9.3 | 66 | 8.12 | 한국어→영어 번역 |
| 8 | 735.8 | 23.1 | 23.7 | 8 | 1.07 | France capital (cached) |
| 9 | 1038.0 | 18.3 | 8.8 | 156 | 18.68 | TCP vs UDP (cached) |

## Phase 2: Concurrent (concurrency=4)

| Metric | avg | p50 | p95 | p99 |
|--------|-----|-----|-----|-----|
| TTFT (ms) | 3355.4 | 3171.3 | 6610.7 | 7182.5 |
| Tok/s (overall) | 2.4 | 2.2 | 3.4 | 3.5 |
| Prefill (tok/s) | 10.6 | 6.9 | 22.7 | 23.3 |
| Decode (tok/s) | 3.3 | 3.3 | 5.0 | 5.1 |

**Aggregate**: 10 requests, 157.9s total, 8.1 tok/s throughput, 0.06 RPS

### Per-Request Detail

| # | TTFT(ms) | Prefill | Decode | Tokens | Time(s) | Prompt |
|---|----------|---------|--------|--------|---------|--------|
| 0 | 724.0 | 23.5 | 4.8 | 8 | 2.41 | France capital |
| 1 | 877.9 | 21.6 | 2.2 | 150 | 69.78 | TCP vs UDP |
| 2 | 884.0 | 19.2 | 3.6 | 256 | 72.89 | Python prime |
| 3 | 1728.6 | 15.0 | 2.4 | 131 | 57.08 | 한국 수도/인구 |
| 4 | 5087.2 | 3.1 | 2.1 | 256 | 126.48 | OOP principles |
| 5 | 7325.5 | 2.0 | 3.0 | 256 | 91.72 | Python vs Rust |
| 6 | 3527.1 | 3.7 | 3.5 | 19 | 8.90 | AI haiku |
| 7 | 5737.1 | 7.1 | 4.0 | 17 | 9.99 | 한국어→영어 번역 |
| 8 | 4847.4 | 3.5 | 5.2 | 8 | 6.39 | France capital |
| 9 | 2815.4 | 6.7 | 2.5 | 183 | 75.03 | TCP vs UDP |

## Phase 3: Prefix Cache Test

| Prompt | 1st TTFT(ms) | 2nd TTFT(ms) | Speedup |
|--------|-------------|-------------|---------|
| France capital | 761.2 | 717.3 | 1.06x |
| TCP vs UDP | 631.6 | 2423.5 | 0.26x |
| Python prime | 3405.1 | 2236.2 | 1.52x |

Prefix cache shows inconsistent speedup — the 2nd prompt actually slowed down, likely due to the concurrent `mx` variable error causing engine restarts during the test.

## Key Observations

1. **Decode throughput**: ~9 tok/s sequential, drops to ~3 tok/s at concurrency=4 due to memory bandwidth saturation
2. **Prefill throughput**: ~20 tok/s sequential — draft model spec decode adds overhead during prefill
3. **TTFT**: ~1s sequential, ~3.4s at concurrency=4
4. **Aggregate throughput**: Nearly identical between seq (8.3 tok/s) and concurrent (8.1 tok/s) — memory bandwidth bound
5. **Intermittent engine error**: `cannot access local variable 'mx'` occurs sporadically during spec decode steps — non-fatal, engine recovers. Root cause TBD.

## Environment Notes

- CacheList compatibility fixes applied (commit `35e648f`)
- GLM-5 uses deepseek_v32 architecture with `CacheList(KVCache(), KVCache())` per layer
- 8-bit KV cache quantization significantly reduces memory (~50% savings)
- Model occupies ~390GB, draft ~16GB, leaving ~100GB for KV cache + overhead on 512GB unified memory
