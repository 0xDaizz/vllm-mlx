# vllm-mlx 분산 추론 시스템 구현 계획

> TB5 + JACCL + RDMA 기반 Tensor Parallelism + Speculative Decoding
>
> 생성일: 2026-02-14
> 상태: Phase 0-3 (TP) + Phase 4a-4b (Spec Decode) 구현 완료, Phase 5 대기

---

## 핵심 아키텍처 결정사항

| 결정 | 선택 | 근거 |
|------|------|------|
| 개발 트랙 | TP와 Spec Decode를 병렬 트랙으로 개발, Phase 5에서 머지 | 독립적 기능, 합류점 명확 |
| BatchGenerator | TP는 투명 래핑 (sharded model 전달), Spec Decode는 새 런타임 추상화 필요 | BG.next()가 1-token-step이라 k-token verify 불가 |
| 프로세스 모델 | Rank 0 = API + Scheduler, 나머지 = thin worker | vLLM 패턴 + Codex 권장 |
| 통신 패턴 | 모든 rank에서 복제된 BatchGenerator 상태, rank 0의 step 결정으로 동기화 | 텐서 분배보다 간단하고 안정적 |
| 샘플링 | Rank 0이 샘플링 후 token ID broadcast (독립 RNG 대신) | 결정론 보장이 가장 쉬움 |
| 타겟 런타임 | AsyncEngineCore + Scheduler 경로 (vLLM Worker/ModelRunner가 아닌) | 실제 사용되는 주력 경로 |
| KV 캐시 | Head-sharded per rank, 캐시 해시에 TP world size 포함 | 메모리 절감 + 일관성 |
| Spec Decode KV rollback | Lazy rollback (seq_lens 조정), transformer-only 우선 지원 | Mamba 등 비트리밍 캐시는 롤백 어려움 |

---

## 참조 프로젝트 및 소스

| 프로젝트 | 참조 내용 | 핵심 파일 |
|----------|----------|----------|
| mlx-lm | TP 빌딩 블록: shard(), sharded_load(), AllToShardedLinear/ShardedToAllLinear | `mlx_lm/models/*.py`, `mlx/nn/layers/distributed.py` |
| exo | Megatron-style TP, auto_parallel, JACCL/RDMA 통합 | `src/exo/worker/engines/mlx/auto_parallel.py` |
| vLLM | TP + continuous batching + spec decode 통합 아키텍처 | `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/spec_decode/` |
| JACCL/MLX | RDMA 통신 백엔드, mx.distributed API | MLX distributed docs |

---

## Track A: Tensor Parallelism

### Phase 0: 분산 런타임 부트스트랩

#### 새 파일 생성

- [x] `vllm_mlx/distributed.py` 생성
  - [x] `MLXCommunicator` 클래스 구현
    - [x] `init(backend="jaccl"|"ring")` 메서드
    - [x] `group: mx.distributed.Group` 속성
    - [x] `rank`, `world_size` 속성
    - [x] `broadcast_step_plan(plan: StepPlan)` 메서드
    - [x] `receive_step_plan() -> StepPlan` 메서드
    - [x] `broadcast_tensor(tensor: mx.array, src: int)` 메서드
    - [x] `barrier()` 메서드
  - [x] `StepPlan` 데이터클래스 정의
    - [x] `step_id: int` (단조 증가 카운터)
    - [x] `inserts: List[InsertOp]` (request_id, tokens, max_tokens, cache_info)
    - [x] `removes: List[str]` (request_id)
    - [x] `sampling_seeds: Dict[str, int]` (요청별 RNG seed)
    - [x] `fingerprint: str` (batch 구성 해시 -- 동기화 검증용)
  - [x] `StepResult` 데이터클래스 정의
    - [x] `step_id: int`
    - [x] `token_ids: Dict[str, int]` (request_id -> sampled token)
  - [x] `InsertOp` 데이터클래스 정의
    - [x] `request_id: str`
    - [x] `tokens: List[int]`
    - [x] `max_tokens: int`
    - [x] `cache_info: Optional[CacheInfo]`

- [x] `vllm_mlx/distributed_launcher.py` 생성
  - [x] `mlx.launch` 기반 분산 서버 진입점
  - [x] rank 0: FastAPI 서버 + EngineCore + Scheduler 실행
  - [x] rank N>=1: `worker_loop()` 실행
  - [x] JACCL hostfile 자동 감지 또는 CLI 인자 지원
  - [x] 환경변수 설정: `MLX_METAL_FAST_SYNCH=1`, `MLX_RANK`, `MLX_JACCL_COORDINATOR`

#### 기존 파일 수정

- [x] `vllm_mlx/worker.py` 수정
  - [x] `init_device()`에서 `mx.distributed.init()` 호출
  - [x] rank, world_size 속성 활성화
  - [x] distributed_init_method 실제 사용하도록 연결

- [ ] `vllm_mlx/platform.py` 수정
  - [ ] `get_device_communicator_cls()` -> `vllm_mlx.distributed.MLXCommunicator` 연결
  - [ ] distributed backend 설정 ("jaccl" | "ring")

- [ ] `vllm_mlx/server.py` 수정
  - [ ] rank 0만 API 서빙하도록 분기 로직 추가
  - [ ] `--distributed` / `--hostfile` CLI 인자 추가

#### 테스트 및 검증

- [ ] 2-node 프로세스 시동 테스트 (mlx.launch)
- [ ] `mx.distributed.all_sum()` 왕복 레이턴시 측정
- [ ] rank 0 API 요청 수신 확인
- [ ] rank 1 StepPlan 수신 및 echo 확인
- [ ] Ring backend 폴백 테스트 (JACCL 미지원 환경)

---

### Phase 1: 오프라인 TP 정합성

#### 모델 로딩

- [x] `vllm_mlx/models/llm.py` 수정
  - [x] `load()` 메서드에 TP 분기 추가
  - [x] `world_size > 1`일 때 `mlx_lm.utils.sharded_load()` 사용
  - [x] `model.shard(group)` 호출 확인 (sharded_load 내부에서 처리)
  - [x] 단일 노드 폴백 경로 유지 (`mlx_lm.load()`)

- [x] `vllm_mlx/model_runner.py` 수정
  - [x] sharded 모델 로딩 지원
  - [x] `load_model()`에서 distributed group 전달

#### 정합성 검증

- [x] 단순 생성 테스트 스크립트 작성
  - [x] greedy decode: "Hello, world" -> 100 tokens
  - [x] 1-rank 결과 vs 2-rank 결과 bit-level 비교
  - [x] temperature=0 (greedy) 결과 동일성 검증
- [ ] 모델별 테스트
  - [ ] Llama 계열 (Llama 3.x)
  - [ ] Qwen 계열 (Qwen 2.5/3)
  - [ ] DeepSeek 계열 (V2/V3)
- [ ] 메모리 프로파일링
  - [ ] rank별 메모리 사용량 측정
  - [ ] 1/N 분할 확인 (head-sharded)
  - [ ] peak memory vs steady state 비교

---

### Phase 2: TP + 엔진 루프 통합

#### Scheduler 확장 (rank 0)

- [x] `vllm_mlx/scheduler.py` 수정
  - [x] `step()` 메서드에 StepPlan broadcast 추가
  - [x] `_build_step_plan(scheduled)` 메서드 구현
    - [x] insert/remove 목록 구성
    - [x] sampling seed 할당
    - [x] batch fingerprint 계산
  - [x] `communicator.broadcast_step_plan(plan)` 호출
  - [x] rank 0만 output 처리하는 분기 추가

#### Worker Loop (rank N>=1)

- [x] `vllm_mlx/distributed.py`에 `worker_loop()` 구현
  - [x] BatchGenerator 생성 (sharded model)
  - [x] StepPlan 수신 루프
  - [x] StepPlan에 따른 BatchGenerator 상태 동기화
    - [x] insert 처리
    - [x] remove 처리
  - [x] `batch_generator.next()` 동기 실행
  - [ ] batch fingerprint 검증 (assert) → Phase 6으로 이관

#### 샘플링 동기화

- [x] rank 0 샘플링 -> broadcast 구현
  - [x] rank 0에서 logits -> token 샘플링
  - [x] `mx.distributed.broadcast()` 또는 커스텀 broadcast로 token ID 전파
  - [x] 모든 rank에서 동일 token ID 확인
- [ ] 대안: 동일 seed 기반 독립 샘플링 (향후 최적화용) → Phase 6으로 이관

#### Engine Core 분산 모드

- [x] `vllm_mlx/engine_core.py` 수정
  - [x] 분산 모드 초기화 로직
  - [x] rank 0: 기존 _engine_loop() + StepPlan broadcast
  - [x] rank N>=1: worker_loop() 실행

#### 테스트 및 검증

- [ ] 동시 10개 요청 스트리밍 테스트
- [ ] 다양한 프롬프트 길이 혼합 테스트
- [ ] 모든 rank에서 batch fingerprint 일치 확인
- [ ] 단일 노드 대비 출력 동일성 (greedy)
- [ ] throughput 측정 (tok/s) 및 단일 노드 대비 비교
- [ ] 요청 abort 시 모든 rank 정상 cleanup 확인
- [ ] 메모리 압박 상황에서의 안정성 테스트

---

### Phase 3: TP + KV/Prefix 캐시

> **Note**: 분산 모드에서는 prefix caching이 비활성화됩니다 (workers에 KV cache 전송 미구현). Phase 5에서 분산 캐시 공유를 구현할 예정입니다.

#### KV 캐시 Head Sharding

- [x] `vllm_mlx/memory_cache.py` 수정
  - [x] head-sharded 캐시 상태 저장/복원
  - [x] rank-local 데이터만 관리하도록 조정
  - [x] 메모리 계산에 TP shard 크기 반영

- [ ] `vllm_mlx/paged_cache.py` 수정 → Phase 5으로 이관
  - [ ] block table은 rank 0에서 관리, broadcast
  - [ ] 물리 블록은 rank-local (head-sharded)

#### 캐시 해시 도메인 확장

- [x] `vllm_mlx/scheduler.py` 캐시 관련 수정
  - [x] 캐시 키 해시에 `tp_world_size` 포함
  - [ ] 캐시 키 해시에 `kv_dtype` 포함 → Phase 5으로 이관
  - [ ] 캐시 키 해시에 `model_revision` 포함 → Phase 5으로 이관
  - [ ] 캐시 키 해시에 `quantization_mode` 포함 → Phase 5으로 이관

#### 글로벌 캐시 일관성

- [x] rank 0이 캐시 lookup 수행하는 로직
- [x] 캐시 hit/miss 결과를 StepPlan에 포함하여 broadcast
- [ ] min cached prefix 로직 (어느 rank이든 miss면 전체 miss) → Phase 5으로 이관
- [x] rank 0의 캐시를 ground truth로 사용

#### 캐시 Persistence

- [x] rank별 별도 저장 경로: `cache_dir/tp{N}/rank{R}/`
- [x] 재시작 시 TP world size 변경 감지 -> 캐시 무효화
- [x] 캐시 파일에 TP 메타데이터 포함

#### 테스트 및 검증

- [ ] 동일 프롬프트 2회 전송 -> 두 번째 캐시 hit 확인
- [ ] 멀티턴 대화에서 prefix 캐시 정상 동작
- [ ] rank 간 캐시 일관성 검증
- [ ] TP world size 변경 후 캐시 무효화 확인
- [ ] 캐시 eviction 시 모든 rank 동기화 확인
- [ ] KV cache quantization + TP 조합 테스트

---

## Track B: Speculative Decoding

### Phase 4a: Spec Decode 런타임 추상화

#### 새 패키지 생성

- [x] `vllm_mlx/spec_decode/` 패키지 생성
  - [x] `__init__.py`

- [x] `vllm_mlx/spec_decode/proposer.py` 구현
  - [x] `BaseProposer` ABC 정의
    - [x] `propose(token_ids, k) -> List[int]`
    - [x] `reset()` (상태 초기화)
  - [x] `ProposerConfig` 데이터클래스

- [x] `vllm_mlx/spec_decode/ngram_proposer.py` 구현
  - [x] `NgramProposer(BaseProposer)` 클래스
  - [x] n-gram 매칭 로직 (프롬프트 내 패턴 검색)
  - [x] 매칭 후속 토큰을 draft로 반환
  - [x] 설정: n (gram 크기), max_k (최대 draft 수)
  - [x] CPU-only 구현 (모델 불필요)

- [ ] `vllm_mlx/spec_decode/eagle_proposer.py` 구현 (Phase 4b 이후)
  - [ ] `EagleProposer(BaseProposer)` 클래스
  - [ ] EAGLE head 모델 로딩
  - [ ] target hidden states 활용
  - [ ] lm_head 공유 (target model과)
  - [ ] autoregressive k-step drafting

- [x] `vllm_mlx/spec_decode/rejection_sampler.py` 구현
  - [x] `RejectionSampler` 클래스
  - [x] Greedy rejection: target argmax vs draft token 비교
  - [x] Stochastic rejection: p_target/p_draft >= uniform 검사
  - [x] Recovered token sampling: max(0, p_target - p_draft) / Z
  - [x] Bonus token: 모두 accept 시 마지막 target distribution에서 샘플링
  - [x] `RejectionResult` 데이터클래스
    - [x] `accepted_token_ids: List[List[int]]`
    - [x] `num_accepted: List[int]`
    - [x] `bonus_token_ids: List[Optional[int]]`

- [x] `vllm_mlx/spec_decode/metadata.py` 구현
  - [x] `SpecDecodeMetadata` 데이터클래스
    - [x] `draft_token_ids: Dict[str, List[int]]`
    - [x] `num_draft_tokens: Dict[str, int]`
  - [x] `SpecDecodeConfig` 데이터클래스
    - [x] `method: str` ("ngram" | "eagle" | "draft_model")
    - [x] `num_speculative_tokens: int`
    - [x] `disable_by_batch_size: Optional[int]`

- [x] `vllm_mlx/spec_decode/runtime.py` 구현
  - [x] `SpecDecodeRuntime` 클래스
    - [x] `propose_drafts(request_states) -> SpecDecodeMetadata`
    - [x] `verify_forward(request_ids, draft_tokens, seq_lens) -> VerifyResult` (Phase 4b에서 모델 연결)
    - [x] `accept_and_commit(verify_result, draft_metadata) -> Dict[str, AcceptResult]`
    - [x] `rollback(request_ids, rollback_counts)` (lazy rollback, Phase 4b에서 KV 캐시 관리)
    - [x] seq_lens 추적 및 조정
    - [x] `should_disable(batch_size)` 배치 크기 기반 자동 비활성화

- [x] `vllm_mlx/spec_decode/metrics.py` 구현
  - [x] `SpecDecodeStats` 데이터클래스
    - [x] `num_drafts: int`
    - [x] `num_draft_tokens: int`
    - [x] `num_accepted_tokens: int`
    - [x] `acceptance_rate_per_position: List[float]`
  - [x] 평균 acceptance length 계산

#### 테스트 및 검증

- [x] NgramProposer 단위 테스트 (12개)
  - [x] 반복 패턴이 있는 텍스트에서 정확한 draft 생성
  - [x] 매칭 없을 때 빈 결과 반환
- [x] RejectionSampler 단위 테스트 (10개)
  - [x] greedy: 동일 토큰 -> accept, 다른 토큰 -> reject 확인
  - [x] stochastic: 확률적 accept/reject 분포 검증
  - [x] bonus token 정상 생성 확인
  - [x] edge cases: k=0, k=1, 전체 accept, 전체 reject
- [x] SpecDecodeStats 단위 테스트 (7개)
- [x] Metadata/Config 단위 테스트 (13개)
- [x] Runtime 단위 테스트 (17개) + 통합 테스트 (1개)

---

### Phase 4b: Spec Decode + Scheduler 통합

#### Phase 4a에서 미뤄진 항목 (Deferred)

- [x] **stochastic+zero-draft 유닛 테스트 추가** (accept_and_commit에서 stochastic 모드 + 빈 draft 테스트)
- [ ] **target_logits/draft_logits 누락 시 명확한 런타임 에러 메시지**
- [ ] **model-based proposer (EAGLE)용 per-request 상태 관리** (unconditional reset() 재검토)

#### Request 확장

- [x] `vllm_mlx/request.py` 수정
  - [x] `spec_token_ids: List[int]` 필드 추가
  - [x] `num_spec_accepted: int` 필드 추가
  - [ ] `last_hidden_state: Optional[mx.array]` 필드 추가 (EAGLE용)

#### Scheduler 통합

- [x] `vllm_mlx/scheduler.py` 수정
  - [x] `spec_decode_enabled` 설정 플래그
  - [x] `proposer` 인스턴스 관리
  - [x] `rejection_sampler` 인스턴스 관리
  - [x] `spec_decode_runtime` 인스턴스 관리
  - [x] `step()` 메서드 확장: spec decode 경로
    - [x] Draft phase: running requests에 대해 k개 토큰 제안
    - [x] Verify phase: target model로 k+1 토큰 동시 검증
    - [x] Accept/Reject phase: rejection sampling 실행
    - [x] Update phase: accepted 토큰 반영, rejected rollback
    - [x] Next Proposal phase: 다음 step용 draft 생성
  - [x] `_step_normal()`: 기존 비-spec 경로 분리 (step() 내 분기)
  - [x] `_step_spec_decode()`: spec decode 전용 경로
  - [x] batch size threshold 초과 시 자동 비활성화
  - [x] `_can_spec_decode()` guard (TP guard, unprocessed_prompts 체크 포함)
  - [x] `_cleanup_finished()`에 batch_generator.remove() 추가 (Codex 리뷰)

#### SchedulerConfig 확장

- [x] `SchedulerConfig`에 spec decode 설정 추가
  - [x] `speculative_method: Optional[str]`
  - [x] `num_speculative_tokens: int = 3`
  - [x] `spec_decode_disable_batch_size: Optional[int]`
  - [x] `draft_model_name: Optional[str]` (EAGLE/draft model용)

#### Engine Core 통합

- [x] `vllm_mlx/engine_core.py` 수정
  - [x] spec decode 모드 초기화 (로깅)
  - [x] spec decode 메트릭 수집 및 get_stats() 노출

#### API 확장

- [x] `vllm_mlx/server.py` 수정
  - [x] `--speculative-method` CLI 인자
  - [x] `--num-speculative-tokens` CLI 인자
  - [ ] `--draft-model` CLI 인자 (EAGLE용, 향후)
- [ ] spec decode 통계를 `/v1/models` 응답에 포함

#### Codex 교차검증 수정사항

- [x] `_cleanup_finished()`에 `batch_generator.remove([uid])` 추가 (finished UID 미제거 버그)
- [x] `_can_spec_decode()`에 `unprocessed_prompts` 체크 추가 (insert=enqueue만)
- [x] `_can_spec_decode()`에 TP guard 추가 (Phase 5까지 TP+Spec 비활성화)
- [x] `distributed_launcher.py`: `remove(uid)` → `remove([uid])` 수정
- [x] `distributed_launcher.py`: worker broadcast 토큰을 `active_batch.y`에 적용
- [x] `distributed_launcher.py`: BatchGenerator import 경로 수정 (`mlx_lm.generate`)
- [x] `engine_core.py`: `acceptance_rate` → `acceptance_rate()` 메서드 호출 수정

#### 테스트 및 검증

- [ ] Ngram proposer E2E 테스트
  - [ ] greedy decode: spec ON/OFF 결과 동일
  - [ ] stochastic decode: 분포 일치 (statistical test)
- [x] 수용률(acceptance rate) 메트릭 리포팅 정상 확인 (unit test)
- [ ] throughput 측정: spec decode ON vs OFF 비교
- [ ] latency (TTFT, inter-token) 측정
- [x] 멀티 요청 동시 처리 정상 (unit test)
- [x] 요청 abort 시 spec state 정상 cleanup (unit test)
- [x] disable_by_batch_size 동작 확인 (unit test)
- [ ] streaming 출력에서 spec decode 토큰 정상 전달

---

## Track C: 통합

### Phase 5: TP + Spec Decode 통합

#### Phase 2+3에서 미뤄진 항목 (Deferred)

- [ ] **Rank 0 서버 경로 sharded_load 통합**
  - [ ] `BatchedEngine._start_llm()`에서 분산 모드 감지 시 `sharded_load()` 사용
  - [ ] `SimpleEngine`에서도 분산 모드 감지 시 `sharded_load()` 사용
  - [ ] `engine_core.py`의 `_shard_applied` 경고를 실제 검증으로 교체
- [ ] **분산 모드 prefix cache 재활성화**
  - [ ] workers에게 KV cache payload 전송 방안 설계 (직렬화 or 재계산)
  - [ ] 또는 모든 rank에서 독립적으로 prefix cache 유지하는 방안
  - [ ] `set_communicator()`의 cache 비활성화 로직 제거
- [ ] **paged_cache.py TP 지원**
  - [ ] block table은 rank 0에서 관리, broadcast
  - [ ] 물리 블록은 rank-local (head-sharded)
- [ ] **캐시 키 도메인 확장 (나머지)**
  - [ ] 캐시 키 해시에 `kv_dtype` 포함
  - [ ] 캐시 키 해시에 `model_revision` 포함
  - [ ] 캐시 키 해시에 `quantization_mode` 포함
- [ ] **min cached prefix 로직**
  - [ ] 어느 rank이든 miss면 전체 miss 처리

#### Draft Proposal 분산

- [ ] Ngram proposer 분산 지원
  - [ ] rank 0에서만 ngram propose
  - [ ] draft token IDs를 StepPlan에 포함하여 broadcast
  - [ ] 모든 rank에서 동일한 draft tokens로 verify

- [ ] EAGLE proposer 분산 지원
  - [ ] draft_tp=1 모드: rank 0에서만 EAGLE forward
  - [ ] draft tokens broadcast 구현
  - [ ] target model verify는 모든 rank에서 TP로 실행

#### Verify Phase 분산

- [ ] 모든 rank에서 draft tokens 포함한 verify forward 실행
- [ ] TP all_sum 자동 처리 확인 (sharded model 내부)
- [ ] rank 0에서 rejection sampling 실행
- [ ] rejection result broadcast -> 모든 rank 반영

#### KV Cache Rollback 분산

- [ ] 모든 rank에서 동일한 rollback count 적용
- [ ] rank 0이 결정 -> broadcast -> 각 rank local cache 조정
- [ ] rollback 후 batch fingerprint 일관성 검증

#### StepPlan 프로토콜 확장

- [ ] StepPlan에 spec decode 필드 추가
  - [ ] `draft_token_ids: Dict[str, List[int]]`
  - [ ] `verify_results: Dict[str, RejectionResult]`
  - [ ] `rollback_counts: Dict[str, int]`

#### 테스트 및 검증

- [ ] TP=2 + Ngram spec decode E2E 테스트
- [ ] TP=2 + EAGLE spec decode E2E 테스트
- [ ] greedy 정합성: TP+Spec vs 단일노드 결과 동일
- [ ] throughput: TP+Spec vs TP-only vs Spec-only 비교
- [ ] 4-node 클러스터 테스트 (가능한 경우)

---

### Phase 6: 성능 + 안정성

#### Phase 2+3에서 미뤄진 테스트 항목 (Deferred)

- [ ] **worker_loop fingerprint 검증**
  - [ ] `batch_generator.next()` 전에 batch fingerprint assert
  - [ ] fingerprint mismatch 시 로깅 및 에러 처리
- [ ] **Phase 2 E2E 테스트**
  - [ ] 동시 10개 요청 스트리밍 테스트
  - [ ] 다양한 프롬프트 길이 혼합 테스트
  - [ ] 단일 노드 대비 출력 동일성 (greedy)
  - [ ] throughput 측정 (tok/s) 및 단일 노드 대비 비교
  - [ ] 요청 abort 시 모든 rank 정상 cleanup 확인
  - [ ] 메모리 압박 상황에서의 안정성 테스트
- [ ] **Phase 3 E2E 테스트**
  - [ ] 동일 프롬프트 2회 전송 -> 두 번째 캐시 hit 확인
  - [ ] 멀티턴 대화에서 prefix 캐시 정상 동작
  - [ ] rank 간 캐시 일관성 검증
  - [ ] TP world size 변경 후 캐시 무효화 확인
  - [ ] KV cache quantization + TP 조합 테스트
- [ ] **동일 seed 기반 독립 샘플링** (성능 최적화 대안)
  - [ ] token broadcast 대신 모든 rank에서 동일 seed로 독립 샘플링

#### 성능 최적화

- [ ] comm/compute overlap
  - [ ] dedicated MLX stream에서 all_sum 실행
  - [ ] forward compute와 통신 오버랩 확인
- [ ] 프로파일링 인프라
  - [ ] step별 시간 분해 (forward, all_sum, sampling, broadcast)
  - [ ] rank별 타이밍 비교
  - [ ] bottleneck 식별 도구
- [ ] vocab-parallel lm_head (대규모 vocab 최적화)
  - [ ] lm_head를 VocabParallelEmbedding으로 전환
  - [ ] local top-k -> merge (full logit gather 대신)
- [ ] padded drafter batch (EAGLE 최적화)
  - [ ] 고정 길이 패딩으로 CUDA graph 유사 최적화
  - [ ] rejected position 마스킹

#### 안정성

- [ ] Heartbeat 시스템
  - [ ] rank 간 주기적 health check (1초 간격)
  - [ ] timeout 감지 (5초 무응답 -> 경고)
- [ ] Step Timeout
  - [ ] step이 N초 이상 걸리면 drain 시작
  - [ ] 설정 가능한 timeout 값
- [ ] Graceful Shutdown
  - [ ] SIGTERM 수신 시 요청 drain -> process 종료
  - [ ] rank 탈락 시 나머지 rank drain
- [ ] Topology Change 대응
  - [ ] 캐시 무효화
  - [ ] 자동 재시작 (선택적)

#### 관측성

- [ ] TP 통신 지연시간 메트릭 (all_sum latency per layer)
- [ ] rank별 메모리 사용량 리포팅
- [ ] Spec decode 수용률 (per-position, per-model)
- [ ] Prometheus 메트릭 엔드포인트
- [ ] 구조화된 로깅 (rank-aware)

#### 문서화

- [ ] 분산 추론 설정 가이드 (hostfile, RDMA 설정)
- [ ] API 사용 예시 (--distributed, --speculative-method)
- [ ] 성능 튜닝 가이드 (TP degree, spec tokens 수, batch size threshold)
- [ ] 트러블슈팅 가이드 (RDMA 연결 실패, rank 동기화 문제)

---

## 타임라인 및 의존성

```
Week 1-2:   Phase 0 (분산 런타임)
             ├── distributed.py, StepPlan
             └── mlx.launch + worker_loop

Week 3-4:   Phase 1 (TP 정합성)          Phase 4a (Spec Decode 추상화)
             ├── sharded_load()            ├── proposer.py
             └── parity tests              └── rejection_sampler.py

Week 5-7:   Phase 2 (TP + Engine)         Phase 4b (Spec Decode + Scheduler)
             ├── StepPlan broadcast         ├── SpecDecodeRuntime
             └── sync batching              └── verify_forward()

Week 8-9:   Phase 3 (TP + Cache)
             ├── head-sharded cache
             └── global consistency

Week 10-12: Phase 5 (TP + Spec Decode 통합)
             ├── distributed draft/verify
             └── distributed rollback

Week 13+:   Phase 6 (성능 + 안정성)
             ├── profiling & optimization
             └── fault tolerance
```

### 의존성 그래프

```
Phase 0 ──> Phase 1 ──> Phase 2 ──> Phase 3 ──┐
                                                ├──> Phase 5 ──> Phase 6
            Phase 4a ─> Phase 4b ──────────────┘
```

- Phase 0은 모든 TP Phase의 전제조건
- Phase 4a/4b는 Phase 0-3과 병렬 진행 가능
- Phase 5는 Phase 3 + Phase 4b 완료 후 시작
- Phase 6는 Phase 5 완료 후 시작

---

## 핵심 파일 변경 요약

### 새로 생성

| 파일 | Phase | 용도 |
|------|-------|------|
| `vllm_mlx/distributed.py` | 0 | MLXCommunicator, StepPlan, worker_loop |
| `vllm_mlx/distributed_launcher.py` | 0 | 분산 서버 진입점 |
| `vllm_mlx/spec_decode/__init__.py` | 4a | 패키지 |
| `vllm_mlx/spec_decode/proposer.py` | 4a | BaseProposer ABC |
| `vllm_mlx/spec_decode/ngram_proposer.py` | 4a | Ngram proposer |
| `vllm_mlx/spec_decode/eagle_proposer.py` | 4b+ | EAGLE proposer |
| `vllm_mlx/spec_decode/rejection_sampler.py` | 4a | Rejection sampling |
| `vllm_mlx/spec_decode/metadata.py` | 4a | SpecDecodeMetadata |
| `vllm_mlx/spec_decode/runtime.py` | 4a | SpecDecodeRuntime |
| `vllm_mlx/spec_decode/metrics.py` | 4a | Spec decode 통계 |

### 수정

| 파일 | Phase | 변경 내용 |
|------|-------|----------|
| `vllm_mlx/worker.py` | 0 | distributed init |
| `vllm_mlx/platform.py` | 0 | MLXCommunicator 연결 |
| `vllm_mlx/server.py` | 0, 4b | rank 분기, CLI 인자 |
| `vllm_mlx/models/llm.py` | 1 | sharded_load() |
| `vllm_mlx/model_runner.py` | 1 | sharded model 지원 |
| `vllm_mlx/scheduler.py` | 2, 3, 4b | StepPlan, 캐시 해시, spec decode |
| `vllm_mlx/engine_core.py` | 2, 4b | 분산 루프, spec decode 모드 |
| `vllm_mlx/memory_cache.py` | 3 | head-sharded 캐시 |
| `vllm_mlx/paged_cache.py` | 3 | block table 동기화 |
| `vllm_mlx/request.py` | 4b | spec_token_ids 필드 |

---

## 리스크 및 완화 전략

| 리스크 | 영향 | 완화 |
|--------|------|------|
| Deadlock (rank 간 collective 불일치) | 전체 시스템 정지 | batch fingerprint + step timeout + barrier |
| 메모리 스파이크 (lazy eval + collective) | OOM | shard-first 로딩, 사전 할당 comm 버퍼 |
| Decode 통신 오버헤드 (2x all_sum / layer / token) | 성능 저하 | comm/compute overlap, 대형 모델만 TP |
| Full mesh TB5 포트 제한 (max 4-5 노드) | 확장성 | ring topology 지원 대기 (JACCL 로드맵) |
| Fault tolerance (rank 탈락 = 전체 중단) | 가용성 | heartbeat + drain + restart |
| Mamba/hybrid 모델 spec decode rollback | 기능 제한 | transformer-only 우선 지원, hybrid는 차후 |
| BatchGenerator 내부 상태 접근 제한 | spec decode 구현 복잡도 | SpecDecodeRuntime 새 추상화 |
