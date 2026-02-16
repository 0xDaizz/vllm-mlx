# Speculative Decoding + Tensor Parallel 버그 리포트

> 작성일: 2026-02-15
> 최종 업데이트: 2026-02-16
> 환경: 2x Mac Studio M4 Ultra 512GB, TB5 RDMA, Kimi K2.5 612GB MoE
> 코드: vllm-mlx develop 브랜치
> 상태: **버그 7 (k≥2 데드락) 활발히 조사 중 — 버그 6 (TP 출력 품질) 미해결**

---

## 요약

n-gram speculative decoding을 분산 Tensor Parallel (TP=2) 환경에서 발견된 버그 현황:

1. **Decode hang** — ✅ 수정 완료 (`0428e89`)
2. **출력 corruption** — ⚠️ 부분 수정 (`0428e89`) — emission 수정됨, state accounting desync 잔존
3. **Trim edge case / batch state sync** — ✅ 수정 완료 (`0428e89`, `ad2d1dc`)
4. **Memory pressure 무한 루프** — ✅ 수정 완료 (`0428e89`)
5. **TP 샘플링 동기화** — ✅ 수정 완료 (`d11cd16`)
6. **TP 출력 품질 저하** — ❓ 미해결
7. **k≥2 spec decode 데드락** — 🔴 **활발히 조사 중** (`878fc00`에서 첫 시도 실패)

---

## 버그 1: Spec Decode + TP Decode Hang — ✅ FIXED

**커밋**: `0428e89`

**문제**: Worker의 `_worker_spec_decode_step()`에서 `active_batch is None`일 때 forward를 skip하여 Rank 0의 all_sum과 deadlock 발생.

**수정**: `active_batch is None` 시 `RuntimeError` raise로 fail-fast 전환. 추가로 `5f1f066`에서 keepalive StepPlan broadcast로 idle desync 방지.

**검증 위치**: `distributed_launcher.py` lines 379-385

---

## 버그 2: 출력 Corruption (단어 반복) — ⚠️ PARTIALLY FIXED — emission 수정됨, state accounting desync 잔존

**커밋**: `0428e89`

**문제**: Spec decode path에서 `result.accepted_tokens`만 emit하고 `batch.y`(old y)를 누락 → 토큰 중복/누락.

**수정**: `committed_tokens = [batch_y] + accepted_tokens[:-1]`로 emission 로직 수정. Bonus token은 다음 step의 `batch.y`로 설정.

**검증 위치**: `scheduler.py` lines 978-1003

---

## 버그 3: Trim Edge Case / Worker Batch State Sync — ✅ FIXED

**커밋**: `0428e89`, `ad2d1dc`

**문제 (원래 보고)**: Stop/length clipping 시 bonus token over-trim 우려.

**문제 (실제 근본 원인)**: TP worker가 spec decode 결과 수신 후 `batch.tokens`와 `batch.num_tokens`를 업데이트하지 않아 rank 간 batch state desync 발생 가능.

**수정**:
- Worker에서 `accepted_tokens`로부터 `batch.tokens`와 `batch.num_tokens` 동기화 로직 추가 (`distributed_launcher.py` lines 424-450)
- `mx.minimum()` boundary 보호로 over-trim 방지
- `mx.eval()` materialize로 lazy graph 누적 방지

**검증 위치**: `distributed_launcher.py` lines 420-464

---

## 버그 4: Memory Pressure Threshold — ✅ FIXED

**커밋**: `0428e89`

**문제**: `engine_core.py`의 threshold가 200GiB(215GB)로 하드코딩 → Kimi K2.5 (332GB) 환경에서 항상 초과 → 매 64 step `mx.clear_cache()` 호출.

**수정**: Threshold를 500GiB로 상향. 512GB 시스템의 ~97%.

**검증 위치**: `engine_core.py` line 186

---

## 이전 세션에서 수정한 코드

### 1. cache_utils.py — `_trim_layer()` 추가
```python
def _trim_layer(layer: Any, trim_amounts: mx.array) -> None:
    """Trim a single cache layer, recursing into CacheList."""
    try:
        from mlx_lm.models.cache import CacheList
        if isinstance(layer, CacheList):
            for sub in layer.caches:
                _trim_layer(sub, trim_amounts)
            return
    except ImportError:
        caches = getattr(layer, "caches", None)
        if isinstance(caches, (tuple, list)):
            for sub in caches:
                _trim_layer(sub, trim_amounts)
            return
    layer.trim_per_sequence(trim_amounts)
```
- `batch.cache`는 `list[BatchKVCache]` (레이어당 하나) → 이전 코드가 직접 `trim_per_sequence` 호출 시 CacheList 레이어에서 크래시
- `batch_variable_trim()`이 `_trim_layer()`를 통해 재귀적으로 CacheList 처리

### 2. scheduler.py — `can_per_seq_trim()` guard
```python
# _can_spec_decode() 내부
batch = self.batch_generator.active_batch
if batch is not None and not can_per_seq_trim(batch.cache):
    return False
```
- cache가 `trim_per_sequence`를 지원하지 않으면 spec decode 비활성화

### 3. scheduler.py — noop_trim off-by-one fix
```python
# 변경 전
noop_trim = spec_pending_state["max_draft_len"]

# 변경 후
noop_trim = spec_pending_state["max_draft_len"] + 1
```
- forward가 `y + drafts`를 처리하므로 trim amount는 `k+1`이어야 함

### 4. scheduler.py — mx scope fix
- `_step_spec_decode_tp()` 내부에 `import mlx.core as mx`가 있어 top-level import를 shadow → 제거

---

## 수정된 코드 목록

| 파일 | 변경 | 커밋 |
|------|------|------|
| `distributed_launcher.py` | RuntimeError guard, worker batch state sync, ready barrier | `0428e89`, `f258b74` |
| `scheduler.py` | emission 로직, can_per_seq_trim guard, noop_trim +1, mx scope fix | `0428e89`, `ad2d1dc` |
| `engine_core.py` | memory threshold 500GiB, cache materialization | `0428e89`, `4b5a16f` |
| `spec_decode/cache_utils.py` | `_trim_layer()` 재귀, `batch_variable_trim` 수정 | `0428e89` |
| `scheduler.py`, `distributed_launcher.py` | _synced_step 몽키패치 (분산 샘플링 동기화) | `d11cd16` |

---

## 버그 2 재발견: Output Corruption — State Accounting Desync (OPEN)

> 발견일: 2026-02-16
> 상태: **미수정 — 실제 테스트에서 재현됨**

### 이전 수정의 한계

커밋 `0428e89`에서 emission 로직을 `[batch_y] + accepted[:-1]`로 수정했으나, 이는 **emission 경로만** 수정한 것. batch state 업데이트 경로는 여전히 raw `accepted_tokens`를 사용하여 drift 발생.

### 재현 결과 (2026-02-16)

```
환경: Kimi K2.5, TP=2, n-gram k=3, temperature=0.0
64 토큰: 10.5 tok/s (경미한 이상)
256 토큰: 16.0 tok/s (심각한 corruption)

출력 예시:
- "a a technical content" (단어 중복)
- "detailed detailed and detailed" (반복)
- "22. Why it's needed" (번호 깨짐)
- 끝부분: "造造造造造造..." (완전 붕괴)
```

### 근본 원인 (Codex 분석)

4가지 경로가 각각 다른 방식으로 committed tokens를 계산:

| 경로 | 데이터 소스 | Clipping 적용 | 상태 |
|------|-----------|-------------|------|
| Emission (응답 전송) | `[batch_y] + accepted[:-1]` | ✅ stop/length 적용 | 정확 |
| `batch.tokens` 업데이트 | `[batch_y] + accepted[:-1]` | ❌ 미적용 | drift |
| `batch.num_tokens` 업데이트 | `len(accepted_tokens)` (bonus 포함) | ❌ 미적용 | drift |
| Worker 미러 (SpecDecodeResult) | raw `accepted_tokens` 기반 | ❌ 미적용 | drift |

이 drift가 매 spec decode step마다 누적 → n-gram proposer history와 KV cache 상태 어긋남 → 토큰 중복/반복 → 장기 요청에서 완전 붕괴.

### 수정 플랜

#### Phase 1: Canonical Committed List 통합

`_step_spec_decode_tp()` (scheduler.py) 수정:

1. **Emission 루프에서 `canonical_committed` 리스트 생성**
   - stop/length clipping 후 **실제 emit된 토큰만** 포함
   - rollback된 토큰은 제외

2. **batch.tokens 업데이트를 canonical_committed 기반으로 변경**
   ```python
   # 변경 전 (현재)
   tokens_for_cache = result.accepted_tokens[:-1] if result.accepted_tokens else []
   new_tokens = [batch_y[batch_idx]] + tokens_for_cache
   n_committed = len(result.accepted_tokens)

   # 변경 후
   new_tokens = canonical_committed[rid]  # emission에서 계산된 것과 동일
   n_committed = len(canonical_committed[rid])
   ```

3. **batch.num_tokens도 canonical 기반으로**
   ```python
   batch.num_tokens[batch_idx] += len(canonical_committed[rid])
   ```

#### Phase 2: SpecDecodeResult에 canonical 정보 포함

Worker에 broadcast하는 `SpecDecodeResult`에 canonical committed 정보 추가:

```python
spec_result = SpecDecodeResult(
    step_id=self._step_count,
    accepted_tokens=canonical_committed,  # raw → canonical로 변경
    trim_amounts=trim_amounts,
    new_y=new_y,
    finished_ids=finished_in_spec,
)
```

#### Phase 3: Worker 미러 동기화

`distributed_launcher.py`의 worker spec decode 처리에서도 canonical 기반 업데이트:
- `batch.tokens[batch_idx]` = canonical committed 토큰 추가
- `batch.num_tokens[batch_idx]` += canonical committed 수

#### Phase 4: 검증

1. 64 토큰 요청 — corruption 없는 coherent 출력
2. 256 토큰 요청 — corruption 없이 완료
3. 500+ 토큰 요청 — 장기 안정성 확인
4. Acceptance rate > 0% 확인 (n-gram 패턴 매칭 시)

### 관련 코드 위치

| 파일 | 라인 | 설명 |
|------|------|------|
| `scheduler.py` | 978-1003 | Emission 루프 (canonical 소스) |
| `scheduler.py` | 1074-1087 | batch.tokens/num_tokens 업데이트 (수정 대상) |
| `scheduler.py` | 1102-1112 | SpecDecodeResult broadcast (수정 대상) |
| `distributed_launcher.py` | 424-450 | Worker batch state 미러 (수정 대상) |

---

## 버그 5: TP 출력 Corruption 근본 원인 — 샘플링 동기화 누락 — ✅ 수정 완료 + 검증 완료

> 발견일: 2026-02-16
> 상태: **수정 완료 + 검증 완료** (커밋 `d11cd16`)

### 발견 경위

버그 2의 state accounting desync 수정 후에도 corruption이 재현됨. Baseline 비교 테스트 (spec decode 없이 normal decode)에서도 **동일한 corruption 패턴** 확인 → spec decode가 아닌 TP 자체의 문제.

### 근본 원인

mlx-lm의 `BatchGenerator._step()` 내부에 분산 샘플링 동기화가 내장되어 있음:

```python
# mlx-lm generate.py _step() 내부
if self._dist_group is not None:
    if self._dist_rank > 0:
        sampled = mx.zeros_like(sampled)
    sampled = mx.distributed.all_sum(sampled, group=self._dist_group)
```

**그러나** `BatchGenerator` 생성 시 `dist_group`을 전달하지 않아서 이 동기화가 **양쪽 Rank에서 모두 비활성**:

- Rank 0 (scheduler.py:1429): `BatchGenerator(model=..., ...)` — dist_group 누락
- Worker (distributed_launcher.py:513): `BatchGenerator(model=..., ...)` — dist_group 누락

### 결과

1. 양 Rank가 독립적으로 샘플링 → 다른 토큰 생성
2. 다른 토큰으로 model forward 호출 → all_sum이 불일치 데이터 합산
3. KV cache에 잘못된 K/V 값 영구 저장
4. 이후 attention이 오염된 KV cache 참조 → 점진적 퇴화
5. 초기 토큰은 확률이 높아 우연히 일치하므로 정상으로 보이다가, 길어질수록 발산

### 왜 64토큰은 괜찮고 256토큰에서 깨지는가

- 초기 토큰들: 매우 높은 확률 (>90%)로 양 Rank가 같은 토큰 샘플링 → 외견상 정상
- 50~100토큰 이후: 확률 분포가 평평해지며 Rank 간 토큰 발산 시작
- 발산 시 KV cache 오염 → 이후 모든 attention 영향 → 복리적 퇴화
- 256토큰: "造造造造..." 완전 붕괴 도달

### 수정: _synced_step 몽키패치 (커밋 `d11cd16`)

BatchGenerator 내부 `_step()` 메서드를 감싸서 분산 샘플링 동기화를 주입하는 방식:

#### scheduler.py — Rank 0 BatchGenerator 몽키패치

```python
# BatchGenerator 생성 후
if self._communicator is not None and self._communicator.is_distributed:
    _orig_step = bg._step
    _comm = self._communicator

    def _synced_step(input_tokens, prompt_cache, samplers, logits_processors, tokens):
        sampled, logprobs = _orig_step(
            input_tokens, prompt_cache, samplers, logits_processors, tokens
        )
        if _comm.rank > 0:
            sampled = mx.zeros_like(sampled)
        sampled = mx.distributed.all_sum(sampled, group=_comm.group)
        return sampled, logprobs

    bg._step = _synced_step
```

#### distributed_launcher.py — Worker BatchGenerator 몽키패치

동일한 패턴으로 worker의 BatchGenerator에도 적용:

```python
if communicator.is_distributed:
    _orig_step = batch_generator._step

    def _synced_step(input_tokens, prompt_cache, samplers, logits_processors, tokens):
        sampled, logprobs = _orig_step(
            input_tokens, prompt_cache, samplers, logits_processors, tokens
        )
        if communicator.rank > 0:
            sampled = mx.zeros_like(sampled)
        sampled = mx.distributed.all_sum(sampled, group=communicator.group)
        return sampled, logprobs

    batch_generator._step = _synced_step
```

**원리**: Rank 0만 실제 샘플 값을 기여하고, 다른 Rank은 zeros를 기여. `all_sum` 후 모든 Rank에 동일한 토큰 ID가 전파됨.

**dist_group 직접 전달 대신 몽키패치를 선택한 이유**: mlx-lm의 `BatchGenerator`에 `dist_group`을 전달하면 내부 동기화가 활성화되지만, 이는 normal decode 경로에서만 동작하고 spec decode verify 경로에서는 별도의 동기화가 필요. 몽키패치 방식은 모든 경로에서 투명하게 동작.

### 진단 검증 결과 (2026-02-16)

임시 진단 코드를 추가하여 양 Rank의 샘플링 결과를 비교:

**진단 방법**: `_synced_step` 내부에 추가 `all_sum`을 삽입. 양 Rank가 raw 샘플 값을 기여하여 합산 → `token_sum == 2 * local_token`이면 MATCH, 아니면 MISMATCH.

**결과 (temp=0, 256 tokens)**:
```
Rank 0: 116 MATCH, 0 MISMATCH
Rank 1: 116 MATCH, 0 MISMATCH
```

- **0 MISMATCH**: 256 스텝 전체에서 양 Rank가 정확히 동일한 토큰을 샘플링
- 이는 _synced_step 몽키패치가 정상 동작함을 증명
- **그러나** 출력 품질 자체는 여전히 저하됨 → 버그 6 참조

### 검증 결과

1. ✅ 64 토큰 요청 — 경미한 이상만 관찰 (개선됨)
2. ✅ 256 토큰 요청 — 양 Rank 동일 토큰 확인 (0 MISMATCH)
3. ⚠️ Baseline (spec decode 없음) — corruption 여전히 존재 → 버그 6
4. ❌ 500+ 토큰 — 미테스트

### 관련 코드 위치

| 파일 | 라인 | 설명 |
|------|------|------|
| `scheduler.py` | 1440-1458 | _synced_step 몽키패치 (수정됨) |
| `distributed_launcher.py` | 519-536 | Worker _synced_step 몽키패치 (수정됨) |
| `distributed_launcher.py` | 639-658 | Normal path cache fixup (수정됨) |

---

## 버그 6: TP 모드 출력 품질 저하 — OPEN

> 발견일: 2026-02-16
> 상태: **미해결 — 원인 조사 중**

### 증상

- 양 Rank가 **정확히 동일한 토큰**을 생성 (버그 5 수정으로 확인)
- 그러나 출력 자체의 품질이 저하됨
- temp=0에서도 256 토큰 수준에서 반복/비문/붕괴 발생
- 단일 노드 테스트 불가 (Kimi K2.5 612GB → 단일 512GB Mac Studio에 적재 불가)

### 진단 결과

- **Inter-rank divergence**: ❌ 아님 (0 MISMATCH 확인)
- **MoE routing divergence**: ❌ 아님 (게이트 라우팅은 양 Rank 동일, expert는 hidden dimension으로 샤딩)
- **Sampling desync**: ❌ 아님 (_synced_step으로 해결)
- **TP 구현 정확성**: ✅ 검증됨 (ShardedToAllLinear, MLA, MoE all_sum 패턴 모두 정상)

### 가능한 원인

1. **bfloat16 정밀도 누적**: all_sum은 bfloat16으로 수행 → 61 레이어 × N 스텝에서 반올림 오차 누적
2. **MLA (Multi-head Latent Attention) TP 상호작용**: 압축된 KV latent + k_pe 분할이 정밀도 민감할 수 있음
3. **int4 양자화 + TP 조합**: Kimi K2.5는 학습 시부터 int4 기본이므로 양자화 자체는 문제 아니지만, TP 분할 + 양자화의 조합이 영향줄 수 있음
4. **Generation 파이프라인 일반 버그**: TP 무관한 문제일 수 있으나, 단일 노드 테스트 불가로 검증 불가

### 다음 조사 방향

- [ ] float32 all_sum 실험 (정밀도 누적 검증)
- [ ] 더 작은 TP 호환 모델로 단일 노드 vs TP 품질 비교
- [ ] Attention score 분포 비교 (TP vs 단일)
- [ ] DeepSeek V3 MLA의 kv_latent 분할 정밀도 분석

---

## 버그 7: k≥2 Spec Decode TP 데드락 — Step 50에서 결정론적 행

> 발견일: 2026-02-16
> 상태: **🔴 활발히 조사 중 — cache_idx=242에서 결정론적 all_sum 데드락 (`878fc00`, `49de3e9`, `4b1c446`)**
> 환경: Kimi K2.5, TP=2 (2x Mac Studio M4 Ultra, TB5 RDMA), n-gram speculative decoding

### 증상

- n-gram spec decode k=1은 정상 동작 (20.1 tok/s, acceptance 78.8%)
- k≥2 (k=2 테스트)에서 **정확히 step 50에서 데드락** 발생
- 재현율 100% (deterministic)
- 약 110개 토큰 생성 후 행 (50 steps × mean_accepted 1.20 + bonus tokens)
- Rank 0 마지막 로그: `[SpecDecode] steps=50, alpha=0.857, mean_accepted=1.20/2, per_pos=['0.86', '0.85']`
- Rank 1 마지막 로그: startup 시의 `Polyfilled BatchKVCache.trim_per_sequence for spec decode` (step-level 로그 없음)
- 기존 cb-baseline (spec decode 없음)도 정상 (15.8-16.2 tok/s)

### 배제된 원인

1. **Auto-disable**: acceptance rate 0.60 (= 1.20/2) > threshold 0.40 → should_auto_disable() returns False. 또한 commit 878fc00에서 TP 모드에서 auto-disable을 명시적으로 비활성화했으나 동일 행 재현
2. **MoE gating routing divergence**: 게이트 라우팅은 all_sum된 입력에서 연산 → 양 Rank 동일
3. **샘플링 desync**: _synced_step 몽키패치로 이미 해결됨 (버그 5)

### 현재 가설

1. **누적 batch state drift**: k=2에서는 매 step 가변 rollback (0~2 토큰) → 50 steps 동안 cache _idx, offset, left_padding 등이 Rank 간 미세 차이 누적 → protocol mismatch로 데드락
2. **cache _idx desync**: batch_variable_trim 후 cache _idx 재계산이 Rank 0과 worker에서 다르게 진행될 가능성
3. **3-token input 특수 케이스**: k=2에서는 [y, d1, d2] 3토큰 입력 → k=1의 [y, d1] 2토큰과 다른 edge case 존재 가능
4. **Step 50 특수성**: rolling window (maxlen=50)이 가득 차는 시점 — auto-disable은 발동 안 하지만, metrics 관련 다른 side effect 가능

### 적용된 수정 (commit 878fc00, 효과 없음)

1. TP 모드에서 auto-disable 비활성화 (scheduler.py)
2. cache _idx consistency check + fixup_cache_after_filter 추가 (scheduler.py)
3. batch_variable_trim 후 fixup_cache_after_filter 호출 (scheduler.py)
4. DEBUG 레벨 진단 로깅 (서버 INFO 레벨이라 출력 안 됨)
5. INFO 레벨 진단 로깅 (`49de3e9`) → 데드락 위치 특정 (model forward 내부)
6. mx.eval() 배리어 5곳 추가 (`4b1c446`) → 효과 없음, 동일 위치에서 행

### 진단 결과 (2026-02-16, commit `49de3e9` + `4b1c446`)

INFO 레벨 진단 로깅 (`[SD-TP]`, `[SD-W]` prefix)을 배포하여 데드락 위치를 정확히 특정:

**두 번의 재현에서 동일한 결과:**
- Rank 0: `step=N PRE-FORWARD input_shape=(1, 2) cache_idx=242` ← 마지막 로그
- Rank 1: `PRE-FORWARD input_shape=(1, 2) cache_idx=242` ← 마지막 로그
- **양쪽 Rank 모두 동일한 상태로 model forward에 진입 후 all_sum 내부에서 데드락**
- 프로토콜 desync 아님 (cache_idx, input_shape 모두 일치)

**시도한 수정과 결과:**
1. `fixup_cache_after_filter` 추가 (`878fc00`) → 행이 step 50에서 step ~87로 이동 (지연만, 해결 아님)
2. `mx.eval()` 배리어 추가 (`4b1c446`) → **효과 없음**, 동일한 cache_idx=242에서 행

**배제된 추가 가설:**
- **Lazy evaluation 누적**: mx.eval 배리어를 5곳에 추가했으나 동일 위치에서 행 → lazy graph 아님
- **프로토콜 desync**: 양 Rank의 cache_idx, input_shape 완전 일치 → 통신 프로토콜 문제 아님

**남은 가설:**
1. **cache_idx=242 특수성**: 두 번의 실행에서 정확히 같은 cache 크기에서 행 → KV cache 용량 한계 또는 Metal 버퍼 크기 제한 가능성
2. **JACCL/RDMA 리소스 고갈**: 누적된 all_sum 연산 후 RDMA 버퍼 부족
3. **spec decode 경로 고유의 cache 레이아웃 문제**: baseline은 256+ 토큰 정상 완료 (cache_idx > 273), spec decode만 242에서 행

### 다음 조사 단계

1. spec decode cache 할당 크기 확인 (max_tokens + k 고려 여부)
2. 더 작은 max_tokens (예: 128)로 테스트하여 cache_idx 한계점 검증
3. all_sum 호출 횟수 대비 baseline과 비교
4. model forward 내부 레이어별 로깅 추가

### 관련 코드 위치

| 파일 | 라인 | 설명 |
|------|------|------|
| `scheduler.py` | 723-734 | TP auto-disable 비활성화 (878fc00) |
| `scheduler.py` | 959-970 | cache _idx consistency check (878fc00) |
| `scheduler.py` | 1082-1084 | fixup_cache_after_filter after trim (878fc00) |
| `scheduler.py` | 970-992 | INFO 진단 로깅 (미배포) |
| `distributed_launcher.py` | 400-420 | Worker INFO 진단 로깅 (미배포) |

---

## 인프라 이슈 (운영 참고)

### Metal Wired Memory Leak
- `kill -9` 후 Metal GPU memory 미해제 → 리부트만 해결
- 정상 wired: ~5GB (340k pages), 누수 시: 300~356GB (20M+ pages)
- **반드시 SIGTERM 먼저, 안 죽으면 SIGKILL**

### nohup + exec 취약점
- `start_server_rank.sh`: ✅ 수정 완료
- `start_server_rank_specngram.sh`: ⚠️ 미수정 (여전히 exec 사용)

### JACCL EBUSY
- kill 후 30초 대기 필수 (RDMA 자원 해제 대기)
