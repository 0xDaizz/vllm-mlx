# 버그 8: TP=2 동시 요청 Batch Hang

> 작성일: 2026-02-21
> 환경: 2x Mac Studio M3 Ultra 512GB, TB5 RDMA (JACCL), Kimi K2.5 612GB MoE
> 코드: vllm-mlx develop 브랜치
> 상태: **조사 중** — stop_tokens 버그 수정 + 상세 로깅 추가, 재현 테스트 필요

---

## 증상

- **TP=1**: 동시 요청 (concurrent requests) 정상 동작
- **TP=2**: 동시 요청 2개 이상 시 서버 hang (무한 대기)
- 단일 요청은 TP=2에서도 정상 동작
- 벤치마크: `bench_concurrent.py` concurrency=2에서 600초 타임아웃

## 재현 조건

```
모델: Kimi K2.5 (MoE, ~306GB/rank)
백엔드: JACCL RDMA over Thunderbolt 5
서버: python -m vllm_mlx.distributed_launcher --backend jaccl
테스트: 동시 2개 요청, max_tokens=30
```

## 로그 분석

### 타임라인 (로그 파일: hwstudio1:/tmp/tp_rank0.log, hwstudio2:/tmp/tp_rank1.log)

1. **Step 24363**: 첫 번째 단일 요청 (acd696d6) 삽입 → 정상 완료
2. **Step 54530**: 첫 번째 동시 요청 (307a0560) 삽입 → batch_size=1
3. **Step 54531**: 두 번째 동시 요청 (5925abdd) 삽입 → batch_size=2
4. **Steps 54532-54558**: batch_size=2 정상 디코딩 (Rank 0: ~28ms/step, Rank 1: ~18ms/step)
5. **Step 54559**: 요청 307a0560 완료 (finished), 내부 filter로 batch_size 2→1 전환
   - Rank 0: 0.048s (평소 0.028s 대비 느림)
   - Rank 1: 0.037s (평소 0.018s 대비 느림)
6. **Step 54560**: removes=[307a0560], batch_size=1, fingerprint 일치
   - 양쪽 Rank: `BEFORE batch_generator.next()` 로그 후 **HANG** (마지막 로그 라인)

### 핵심 관찰

- Step 54560에서 양쪽 Rank의 상태가 일치 (batch_size=1, fingerprint 동일, token 값 동일)
- Hang은 `batch_generator.next()` 내부, 즉 `_synced_step` 내부에서 발생
- `_synced_step`은 `_orig_step()` (모델 forward) + `sampled.item()` (평가 강제) + `all_sum` (토큰 동기화) 3단계
- `sampled.item()`이 전체 모델 forward 계산 그래프를 평가하며, 여기에는 모든 transformer 레이어의 all_sum이 포함됨

## 확인된 버그: stop_tokens 불일치

### 원인

- **Rank 0** (scheduler.py): `stop_tokens = self._get_stop_tokens()` → 모델 EOS 토큰 포함 (예: `{151329, 151336}`)
- **Rank 1** (distributed_launcher.py): `BatchGenerator(model=model, sampler=sampler)` → `stop_tokens` 미전달 → 기본값 `set()` (빈 집합)

### 영향

`BatchGenerator._next()` (mlx_lm/generate.py 내부):
```python
# Line 1273: stop_tokens 체크
if t in self.stop_tokens:  # Rank 0: True, Rank 1: False (빈 집합)
    finish_reason = "stop"
elif num_tok >= max_tok:    # 둘 다 여기서 종료 가능
    finish_reason = "length"

# Line 1287-1289: 내부 filter
if len(end_idx):
    batch.filter(keep_idx)  # batch_size 변경!
```

EOS 토큰이 max_tokens보다 먼저 도달하면:
1. Rank 0: stop_tokens에서 매치 → 해당 요청 내부 filter → batch_size 줄어듦
2. Rank 1: stop_tokens 빈 집합이라 매치 안 됨 → batch_size 유지
3. 다음 `_step()` 호출: input_tokens 크기 불일치 → all_sum shape mismatch → **DEADLOCK**

### 수정

```python
# distributed_launcher.py line 527-532
batch_generator = BatchGenerator(
    model=model,
    max_tokens=4096,
    stop_tokens=set(tokenizer.eos_token_ids) if hasattr(tokenizer, 'eos_token_ids') else set(),
    sampler=sampler,
)
```

### 이 테스트 케이스와의 관련성

이 특정 테스트에서는 max_tokens=30이 EOS보다 먼저 도달하여 양쪽 모두 `finish_reason="length"`로 종료. 따라서 stop_tokens 버그가 **이 특정 hang의 직접 원인은 아닐 수 있음**. 하지만 다른 시나리오에서 hang을 유발할 수 있는 실제 버그.

## 추가 의심 지점

### 1. fixup_cache_after_filter 누락 (Worker remove 경로)

Worker의 `batch_generator.remove()` 호출 후 (distributed_launcher.py line 658):
```python
batch_generator.remove([uid])  # 내부적으로 batch.filter() 호출
# fixup_cache_after_filter 미호출!
```

반면 Rank 0의 `_process_batch_responses`에서는 (scheduler.py):
```python
bg.remove(finished_uids)
fixup_cache_after_filter(bg.active_batch.cache)  # 호출됨
```

`batch.filter()`는 `_idx`를 재계산하지 않음 → stale `_idx` → 다음 forward에서 서로 다른 크기의 KV 텐서 → all_sum 불일치

### 2. 내부 filter vs 외부 remove 타이밍

`batch_generator.next()` 내부에서 완료된 요청을 filter한 후, 바깥에서 다시 remove를 시도하면:
- Rank 0: next() 내부에서 이미 filter됨 → 외부 remove는 이미 없는 uid를 시도
- Rank 1: next() 내부에서 filter 안 됨 (stop_tokens 문제) → 외부 remove에서 처리

이 불일치가 batch.uids 목록 차이를 만들어 fingerprint mismatch 또는 all_sum shape 불일치를 유발할 수 있음.

### 3. lazy evaluation 내의 all_sum 누적

Step 54559에서 내부 filter가 발생한 후, Step 54560의 `_next()` 내부:
1. `_step()` 호출 → 모델 forward → 각 transformer 레이어에서 all_sum (TP sharding)
2. 이 all_sum들은 lazy하게 그래프에 누적
3. `sampled.item()` 또는 `mx.async_eval`에서 전체 평가
4. 만약 이전 step의 filter로 인해 cache 상태가 rank 간 다르면, forward 자체의 텐서 크기가 달라져 all_sum에서 hang

## 아키텍처 문제: `_dist_group` 내장 지원 vs `_synced_step` monkey-patch

### 발견

mlx-lm의 `BatchGenerator._step()`에는 이미 `_dist_group`을 통한 분산 토큰 동기화가 **내장**되어 있음 (`generate.py:1191-1194`):

```python
# mlx-lm 내장 코드 (현재 비활성 — dist_group=None)
if self._dist_group is not None:
    if self._dist_rank > 0:
        sampled = mx.zeros_like(sampled)
    sampled = mx.distributed.all_sum(sampled, group=self._dist_group)
```

그러나 vllm-mlx는 `BatchGenerator` 생성 시 `dist_group` 파라미터를 전달하지 않고, 대신 `_synced_step` monkey-patch로 **동일한 로직을 재구현**함.

### 핵심 차이: forced evaluation

| | mlx-lm 내장 | _synced_step monkey-patch |
|---|---|---|
| 평가 모드 | **lazy** (mx.eval 없음, graph에 누적) | **forced** (`sampled.item()` + `mx.eval()`) |
| 동기 지점 | `_next()` 내 `mx.async_eval(batch.y, batch.logprobs)` | _step 내부에서 즉시 |
| 부수 효과 | 없음 | `sampled.item()`이 MoE all_sum 포함 전체 graph 평가 강제 |

monkey-patch의 `sampled.item()`이 **모델 forward의 전체 compute graph를 동기적으로 평가**하는 것이 중요. MoE 레이어의 TP all_sum이 여기서 실행되며, **cache 상태가 rank 간 다르면 이 지점에서 deadlock**.

### 권장 수정 (Option 1 — dist_group 활용)

```python
# distributed_launcher.py — 수정 전
batch_generator = BatchGenerator(model=model, max_tokens=4096, sampler=sampler)
# + _synced_step monkey-patch (40줄)

# distributed_launcher.py — 수정 후
batch_generator = BatchGenerator(
    model=model, max_tokens=4096, sampler=sampler,
    dist_group=communicator.group,  # 내장 지원 활용
)
# monkey-patch 제거
```

```python
# scheduler.py — 수정 후
bg = BatchGenerator(
    model=self.model, ...,
    dist_group=self._communicator.group if self._communicator else None,
)
# monkey-patch 제거
```

## Codex 분석 결과

### 가장 유력한 원인: 내부 filter + 외부 remove 비대칭 (#3/#4)

- `_next()` 내부에서 이미 filter된 요청을 외부에서 다시 `remove()` 시도
- `remove()`는 **idempotent가 아님** — 이미 없는 uid에 대해서도 `batch.filter(keep_idx=all)` 실행 가능
- Rank 0과 Worker가 remove/fixup을 **다른 시점**에 수행 → cache metadata drift

### 강력한 기여 요인: Worker fixup 누락 (#2)

- `_idx` drift가 직접 all_sum shape을 바꾸진 않지만
- **attention context length를 변경** → 토큰/종료 조건 divergence → 이후 collective mismatch

### 원인 순위 (이 케이스)

1. **#3/#4 timing + double-remove 비대칭** (가장 유력한 트리거)
2. **#2 Worker remove 후 fixup 누락** (강력한 기여 요인)
3. **#1 stop_tokens 불일치** (실제 버그, 하지만 이 max_tokens=30 테스트에서는 직접 원인 아닐 수 있음)

### Codex 권장 경화 조치

1. Worker에서 `remove()` 후 **항상** `fixup_cache_after_filter` 호출
2. uid가 `active_batch.uids`에 없으면 **filter 건너뛰기** (guard)
3. 내부+외부 이중 제거 race 방지 (single-source-of-truth)
4. fingerprint에 cache metadata 포함 (`_idx`, `keys.shape[2]`, `max(offset+left_padding)`)

## 추가된 디버깅 로그

`_synced_step`에 7단계 로깅 추가 (양쪽 Rank):
1. ENTER: input_shape
2. BEFORE _orig_step
3. AFTER _orig_step: sampled_shape, 소요 시간
4. BEFORE sampled.item()
5. AFTER sampled.item(): pre_sync 값
6. BEFORE all_sum
7. BEFORE mx.eval(sampled)
8. EXIT: pre/post 값, 총 소요 시간

이 로그로 hang이 _orig_step (모델 forward) 내부인지, sampled.item() (평가 강제)인지, all_sum 동기화인지 정확히 파악 가능.

## 다음 단계

1. [x] stop_tokens 버그 수정
2. [x] 상세 _synced_step 로깅 추가
3. [ ] 코드를 양쪽 studio에 배포 (rsync)
4. [ ] 동일 조건으로 hang 재현 → 새 로그에서 정확한 hang 지점 파악
5. [ ] fixup_cache_after_filter 누락 수정 (Worker remove 경로)
6. [ ] Codex 분석 결과 반영

## 관련 파일

| 파일 | 역할 |
|------|------|
| `vllm_mlx/distributed_launcher.py` | Worker 루프 (Rank 1+) |
| `vllm_mlx/scheduler.py` | Rank 0 스케줄러 |
| `mlx_lm/generate.py` (외부) | BatchGenerator._next(), _step(), filter() |
| `mlx_lm/models/cache.py` (외부) | BatchKVCache.filter(), update_and_fetch() |
| `vllm_mlx/spec_decode/cache_utils.py` | fixup_cache_after_filter |
| `vllm_mlx/distributed.py` | MLXCommunicator (all_sum, broadcast_object) |
