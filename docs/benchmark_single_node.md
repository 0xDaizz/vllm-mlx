# Single-Node Benchmark: Moonlight-16B-A3B + Kimi-K2 Draft

> 실행일: 2026-02-16
> 환경: Mac Studio M4 Ultra 512GB (hwstudio1)
> 주 모델: Moonlight-16B-A3B-Instruct-4-bit (~16B params, 3B active, int4)
> Draft 모델: Kimi-K2-Instruct-DRAFT-0.6B-MLX
> max_tokens: 256, temperature: 0.0 (greedy)
> 상태: **완료**

---

## 핵심 결론

1. **cb-baseline이 가장 빠름**: decode ~130 tok/s, prefill ~1800 tok/s (very_long)
2. **N-gram spec decode는 역효과**: acceptance rate가 너무 낮아 (19-47%) verify 오버헤드가 이득보다 큼. 다양한 자연어 텍스트에는 n-gram 패턴 매칭이 비효율적
3. **Draft model spec decode는 심각한 성능 저하**: Kimi-K2 0.6B ↔ Moonlight-16B는 모델 패밀리가 달라 acceptance rate가 낮음 (17-60%). 게다가 draft model forward 자체의 오버헤드가 커서 decode 속도가 1/3~1/5로 감소
4. **Simple vs CB**: prefill은 simple이 빠르지만 (TTFT 0.045s vs 0.146s for short), decode는 거의 동일. CB는 concurrent request 처리가 가능하므로 production에서 CB 사용 권장
5. **최적 k**: draft model의 경우 k=1이 가장 높은 acceptance rate (60.1%)이지만 여전히 순손실. 같은 모델 패밀리의 작은 모델을 draft로 사용해야 효과적

---

## 테스트 구성

| # | Config ID | Mode | Spec Method | Draft Model | k | 상태 |
|---|-----------|------|-------------|-------------|---|------|
| 1 | simple-baseline | simple | - | - | - | ✅ 완료 |
| 2 | cb-baseline | CB | - | - | - | ✅ 완료 |
| 3 | ngram-k1 | CB | ngram | - | 1 | ✅ 완료 |
| 4 | ngram-k2 | CB | ngram | - | 2 | ✅ 완료 |
| 5 | ngram-k3 | CB | ngram | - | 3 | ✅ 완료 |
| 6 | ngram-k4 | CB | ngram | - | 4 | ✅ 완료 |
| 7 | ngram-k5 | CB | ngram | - | 5 | ✅ 완료 |
| 8 | draft-k1 | CB | draft_model | Kimi-K2 0.6B | 1 | ✅ 완료 |
| 9 | draft-k2 | CB | draft_model | Kimi-K2 0.6B | 2 | ✅ 완료 |
| 10 | draft-k3 | CB | draft_model | Kimi-K2 0.6B | 3 | ✅ 완료 |
| 11 | draft-k4 | CB | draft_model | Kimi-K2 0.6B | 4 | ✅ 완료 |
| 12 | draft-k5 | CB | draft_model | Kimi-K2 0.6B | 5 | ✅ 완료 |
| 13 | draft-k8 | CB | draft_model | Kimi-K2 0.6B | 8 | ✅ 완료 |

## 프롬프트

| Level | 단어 수 | 설명 |
|-------|---------|------|
| short | ~7 | "Write a haiku about the ocean." |
| medium | ~40 | 일반상대성이론 설명 요청 |
| long | ~140 | 양자 컴퓨팅 하드웨어 비교 요청 |
| very_long | ~370 | 기후변화/재생에너지 분석 요청 |

---

## Decode 속도 비교 (tok/s)

| Config | short | medium | long | very_long | Acceptance Rate |
|--------|-------|--------|------|-----------|----------------|
| **simple-baseline** | **140.7** | **130.2** | **123.0** | **128.1** | - |
| **cb-baseline** | **141.7** | **131.4** | **123.6** | **129.2** | - |
| ngram-k1 | 112.1 | 105.4 | 98.4 | 98.4 | 47.5% |
| ngram-k2 | 113.8 | 105.5 | 98.0 | 97.7 | 33.7% |
| ngram-k3 | 113.1 | 106.0 | 98.8 | 96.8 | 27.1% |
| ngram-k4 | 113.0 | 105.7 | 98.0 | 96.7 | 22.8% |
| ngram-k5 | 108.7 | 105.0 | 97.3 | 96.3 | 19.1% |
| draft-k1 | 54.1 | 46.1 | 36.5 | 25.1 | 60.1% |
| draft-k2 | 43.0 | 47.9 | 38.2 | 27.1 | 45.9% |
| draft-k3 | 39.8 | 45.6 | 39.4 | 27.9 | 38.5% |
| draft-k4 | 35.0 | 42.8 | 38.6 | 26.9 | 31.8% |
| draft-k5 | 28.1 | 42.1 | 35.9 | 26.1 | 26.9% |
| draft-k8 | 22.4 | 33.7 | 26.6 | 22.0 | 17.2% |

> **Best performer: cb-baseline** (no speculative decoding)

---

## Prefill 속도 비교

| Config | short TTFT (s) | medium TTFT (s) | long TTFT (s) | very_long TTFT (s) | long prefill (t/s) | very_long prefill (t/s) |
|--------|---------------|-----------------|---------------|---------------------|--------------------|----|
| simple-baseline | 0.045 | 0.063 | 0.090 | 0.310 | N/A* | N/A* |
| cb-baseline | 0.146 | 0.181 | 0.214 | 0.292 | 887.0 | 1783.5 |
| ngram-k1 | 0.147 | 0.156 | 0.152 | 0.140 | 1251.1 | 3720.2 |
| ngram-k3 | 0.148 | 0.139 | 0.152 | 0.146 | 1247.7 | 3554.8 |
| draft-k1 | 0.170 | 0.162 | 0.185 | 0.221 | 1029.0 | 2357.0 |
| draft-k3 | 0.186 | 0.179 | 0.191 | 0.214 | 992.7 | 2430.4 |
| draft-k8 | 0.231 | 0.224 | 0.245 | 0.264 | 774.2 | 1970.5 |

*Simple 모드는 prompt_tokens를 보고하지 않아 정확한 prefill tok/s 계산 불가*

---

## Acceptance Rate vs k

| k | N-gram | Draft Model |
|---|--------|-------------|
| 1 | 47.5% | 60.1% |
| 2 | 33.7% | 45.9% |
| 3 | 27.1% | 38.5% |
| 4 | 22.8% | 31.8% |
| 5 | 19.1% | 26.9% |
| 8 | - | 17.2% |

---

## 분석

### N-gram Speculative Decoding

- **결론: 비효과적** (이 모델/프롬프트 조합에서)
- 원인: 일반 자연어 텍스트에서는 n-gram 패턴 반복이 드물어 acceptance rate가 낮음
- k가 증가할수록 acceptance rate 감소하고, verify 오버헤드만 증가
- decode 속도가 baseline 대비 ~20-25% 감소
- **적합한 사용 사례**: JSON, XML, 코드 보일러플레이트 등 반복적/구조적 출력
- 오버헤드: verify forward에서 k+1 토큰을 한번에 처리하지만, rejection으로 인한 rollback과 재시도 비용이 이득을 상회

### Draft Model Speculative Decoding

- **결론: 심각한 성능 저하** (이 모델 조합에서)
- 원인 1: Kimi-K2 0.6B와 Moonlight 16B는 **모델 패밀리가 완전히 다름** → 토큰 분포가 크게 달라 acceptance rate가 낮음
- 원인 2: draft model forward 자체의 계산 비용이 target model 단독 decode보다 오히려 큰 오버헤드 추가
- 원인 3: 프롬프트가 길어질수록 draft model의 KV cache 관리 오버헤드가 증가하여 very_long에서 특히 심각한 성능 저하 (25 tok/s vs 130 tok/s baseline)
- **적합한 사용 사례**: 같은 모델 패밀리의 작은 버전을 draft로 사용 (예: Moonlight-2B → Moonlight-16B)

### Continuous Batching

- decode 속도는 simple mode와 거의 동일 (~130 tok/s)
- TTFT는 simple mode가 더 빠름 (0.045s vs 0.146s for short) — simple mode는 scheduling 오버헤드가 없음
- CB의 장점: concurrent request 처리 가능, speculative decoding 활성화 가능
- **Production 권장**: continuous batching 모드 사용

---

## 발견된 이슈

### 1. Moonlight 토크나이저 `add_special_tokens=True` 크래시

**문제**: Moonlight 모델의 커스텀 토크나이저 (`tokenization_moonshot.py`)가 `tokenizer.encode(prompt, add_special_tokens=True)` 호출 시 `ValueError: type of None unknown` 에러 발생.

**원인**: mlx_lm의 `stream_generate()` 내부에서 프롬프트가 BOS 토큰으로 시작하지 않으면 자동으로 `add_special_tokens=True`를 사용. Moonlight의 커스텀 토크나이저는 이를 처리하지 못함.

**수정**: `vllm_mlx/models/llm.py`에서 프롬프트를 `mx.array`로 사전 인코딩하여 mlx_lm에 전달. mlx_lm은 `mx.array` 입력 시 내부 `tokenizer.encode()` 호출을 건너뜀.

### 2. Simple 엔진 prompt_tokens 미보고

**문제**: simple 엔진 모드에서 응답의 `prompt_tokens`가 항상 0으로 보고됨.

**원인**: `LLMModel.stream_generate()` → `mlx_lm.stream_generate()`가 `prompt_tokens` 정보를 streaming output에 포함하지 않음.

**영향**: prefill tok/s 계산 불가. 벤치마크에서는 TTFT만 사용.

---

## 스크립트 및 재현

벤치마크 스크립트: `scripts/benchmark_all_configs.py`

```bash
# 전체 실행
python3.14 scripts/benchmark_all_configs.py

# 특정 구성만 실행
python3.14 scripts/benchmark_all_configs.py --config-id cb-baseline

# 특정 프롬프트만 실행
python3.14 scripts/benchmark_all_configs.py --prompt-level short

# 외부 서버 사용 (자동 시작/중지 안함)
python3.14 scripts/benchmark_all_configs.py --no-server-management
```

JSON 결과: `/tmp/benchmark_results.json`
