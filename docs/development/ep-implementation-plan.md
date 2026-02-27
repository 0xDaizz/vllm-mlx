# Expert Parallelism (EP) Implementation Plan

> Status: DRAFT v2 — Open Questions 해소 완료
> Author: hw
> Date: 2026-02-27
> Target: vllm-mlx 서빙 레이어 (향후 standalone vllm-ep 전환 가능)

## 1. Overview

순수 EP (Expert Parallelism) 를 vllm-mlx에 구현한다.
Attention은 각 rank에 복제하고, MoE의 routed expert만 rank 간 분할한다.

### 왜 순수 EP인가

- 주력 MoE 모델 (DeepSeek V3/V4, Kimi K2.5) 이 MLA → KV head 1개 → TP로 KV 분할 불가
- Qwen3.5는 GDN+GQA 하이브리드 → 75% 선형 어텐션이라 TP 이점 미미
- 순수 EP는 통신 포인트가 MoE layer에서만 발생 → 안정성 우수
- 커스텀 MLX 빌드 (~/mlx) 에 EP 풀스택 (dispatch/combine/Metal 커널) 이 이미 구현됨

### EP vs TP 벤치마크 (기 측정, Kimi K2.5 스케일)

| 구간 | EP/TP geomean | 해석 |
|------|--------------|------|
| 전체 (N=1~2048) | 0.55x | EP 1.8x 빠름 |
| **Decode (N≤64)** | **0.32x** | **EP 3.1x 빠름** |
| Prefill (N≥256) | 1.07x | TP 7% 빠름 |

실제 LLM 서빙은 decode 80%+ → EP 실전 유리. Crossover 약 N=200~256.

### 파티션 전략

```
복제 (양쪽 동일): Embedding, LM Head, Attention, LayerNorm, Gate, Shared Expert
분할 (반반):      Routed Expert [0..E/2-1] | [E/2..E-1]
통신:             MoE layer에서만 all_to_all (dispatch + combine)
```

### 장기 방향

- 현재: vllm-mlx의 `--expert-parallel` 모드로 구현
- 향후: 독립 패키지 (vllm-ep) 로 분리 가능성 염두
  - 전체 스택이 커스텀 (mlx → mlx-lm → vllm-mlx) 이라 upstream PR이 어려움
  - EP 서빙에 특화된 경량 레이어로 전환

## 2. Decisions (확정)

| 항목 | 결정 | 비고 |
|------|------|------|
| 커스텀 MLX | `pip install -e ~/mlx` 양쪽 배포 완료 | PR1A (#3164) upstream 제출됨 |
| 첫 타겟 모델 | **Qwen3.5-397B-A17B** | hwstudio1:~/models/, 512 experts, 60 layers 전부 MoE |
| model_type | `qwen3_5_moe` | mlx-lm v0.30.7+ upstream 지원 (`qwen3_5_moe.py`) |
| TP 공존 | `--expert-parallel` 별도 모드 | 기존 TP 유지 |
| Shared expert | 복제 (양쪽 독립 실행) | 통신 불필요 |
| MoE forward 수정 | **Adapter 패턴** (아래 상세) | monkey-patch 아님, mlx-lm 변경 없음 |
| Sampling 동기화 | Greedy: 독립 가능 / Sampling: seed 동기화 | EP logit bit-exact 확인됨 |
| 인프라 | hwstudio1 + hwstudio2, RDMA 사용 가능 | — |
| 장기 방향 | standalone vllm-ep 분리 가능성 염두 | 전체 스택 커스텀이라 upstream PR 어려움 |

## 3. MoE Forward 수정 전략: Adapter 패턴

### 왜 monkey-patch가 아닌가

- monkey-patch: `SwitchGLU.__call__`을 런타임 교체 → 암묵적, 추적 어려움
- mlx-lm 포크: 유지보수 부담, upstream 변경 시 충돌
- **Adapter**: 명시적 래핑, 독립 테스트 가능, 향후 standalone 분리에 유리

### Adapter 구조

```python
class EPMoEAdapter(nn.Module):
    """mlx-lm MoE 모듈을 EP 실행으로 래핑"""

    def __init__(self, original_moe, ep_group, rank, world_size):
        # Router (gate) — 전역 routing, 그대로 복제
        self.gate = original_moe.gate

        # Shared expert — 그대로 복제
        self.shared_experts = original_moe.shared_experts

        # Routed expert — 로컬 범위만 추출
        E_total = original_moe.switch_mlp.num_experts
        E_local = E_total // world_size
        local_start = rank * E_local
        self.local_experts = slice_experts(
            original_moe.switch_mlp, local_start, E_local
        )
        self.E_total = E_total
        self.E_local = E_local
        self.ep_group = ep_group

    def __call__(self, x):
        # 1. Gate — 전역 routing
        indices, weights = self.gate(x)

        # 2. Dispatch — 토큰을 expert 소유 rank로 전송
        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            x, indices, weights,
            num_experts=self.E_total,
            capacity_factor=1.25,
            group=self.ep_group,
        )

        # 3. Local expert FFN
        expert_out = self._run_local_experts(dispatched)

        # 4. Combine — 결과 수집 + 가중 합산
        routed_out = mx.distributed.moe_combine_exchange(
            expert_out, route_idx, weights, x,
            group=self.ep_group,
        )

        # 5. Shared expert (로컬, 통신 없음)
        shared_out = self.shared_experts(x) if self.shared_experts else 0

        return routed_out + shared_out
```

### 적용 시점 (모델 로딩 후)

```python
# EP 모드에서 모델 로딩 후 MoE layer를 adapter로 교체
for layer in model.layers:
    if has_moe(layer):
        layer.mlp = EPMoEAdapter(layer.mlp, ep_group, rank, world_size)
```

**이것은 monkey-patch가 아닌 이유:**
- 원본 클래스의 메서드 테이블을 수정하지 않음
- 명시적인 module 교체 (속성 재할당)
- EPMoEAdapter는 독립 클래스로 단위 테스트 가능
- 향후 standalone 패키지에서 그대로 가져갈 수 있음

## 4. Prerequisites

- [x] 커스텀 MLX (~/mlx) `pip install -e` 양쪽 배포
- [x] EP primitives (dispatch/combine) 구현 완료 (Phase 5.5까지)
- [x] hwstudio1 + hwstudio2 RDMA 사용 가능
- [x] Qwen3.5 MoE 구조 확인: 60 layer 전부 MoE, 512 experts, 10+1 active
- [x] mlx-lm upstream Qwen3.5 지원 확인: `qwen3_5_moe.py` (v0.30.7+)
- [x] EP logit 결정론성 확인: bit-exact (greedy 독립 샘플링 가능)
- [ ] Qwen3.5 safetensors weight naming convention 확인 (Phase 1 첫 작업)
- [x] mlx-lm 0.30.7 pip 설치 완료 (hwstudio1) — `qwen3_5_moe.py` 이미 존재

## 5. Implementation Phases

### Phase 1: 모델 구조 분석 + EP 로딩

Qwen3.5-397B-A17B의 MoE 구조를 파악하고 EP-aware 로딩을 구현한다.

**핵심 작업:**
- [ ] Qwen3.5 config.json / safetensors index 분석
  - expert weight naming pattern
  - MoE layer vs dense layer 구분
  - GDN layer vs GQA layer 구분
  - shared expert 존재 여부 및 구조
- [ ] Selective safetensors loading 구현
  - weight name → (replicate | local_expert | skip) 분류
  - rank별 E_local개 expert만 로드
  - attention/gate/shared/embedding 전체 로드
- [ ] 메모리 검증: 각 rank의 로드 후 메모리 사용량 확인

**산출물:** 양쪽 rank에서 모델 로드 성공, 메모리 절반 미만 확인

### Phase 2: EP Forward Pass

EPMoEAdapter를 구현하고 2-rank forward pass를 검증한다.

**핵심 작업:**
- [ ] EPMoEAdapter 구현
  - Qwen3.5의 MoE 모듈 구조에 맞춤
  - 커스텀 MLX dispatch/combine 연동
  - Local expert FFN 실행 (batched or loop)
- [ ] 모델 로딩 후 MoE layer → EPMoEAdapter 교체
- [ ] `moe_ep_warmup()` 호출 (RDMA + Metal JIT)
- [ ] 단일 프롬프트 forward pass 검증
  - 2-rank EP 출력 vs 1-rank 전체 모델 출력 비교
  - Greedy decoding 일치 확인

**산출물:** EP 2-rank forward pass 정확성 검증 통과

### Phase 3: 서빙 통합

vllm-mlx의 distributed 인프라에 EP 모드를 연동한다.

**핵심 작업:**
- [ ] CLI: `--expert-parallel` 플래그
- [ ] distributed_launcher.py — EP 모드 분기
  - 모델 로딩: EP selective loading
  - Worker loop: **attention all_sum 없음** (핵심 차이)
    - StepPlan broadcast — 유지 (batch 상태 동기화)
    - Token sampling — Rank 0 샘플링 + broadcast (유지)
    - MoE 내부의 all_to_all — EPMoEAdapter가 처리 (worker가 관여 안 함)
- [ ] Scheduler EP 모드
  - _synced_step: MoE 내부에서 동기화되므로 외부 all_sum 불필요
    → 단, 샘플링 결과 동기화는 여전히 필요 (TODO: 정확한 동기 지점 설계)
  - Prefix cache: EP에서 활성화 가능 (attention 독립이므로)
- [ ] `moe_ep_warmup()` 서버 시작 시 자동 호출

**산출물:** `vllm serve --expert-parallel` 로 2-rank 서버, API 호출 성공

### Phase 4: 검증 + 벤치마크

- [ ] 출력 품질: 단일노드 대비 greedy 일치, sampling 분포 유사
- [ ] Throughput: 단일노드 vs EP 2-rank (decode tok/s)
- [ ] Latency: TTFT, ITL 측정
- [ ] `moe_ep_stats()` 메트릭: dispatch/combine 횟수, overflow, backend 선택
- [ ] Capacity factor 튜닝
- [ ] Stress test: concurrent requests, long generation

### Phase 5 (향후): 확장

- [ ] 추가 모델: DeepSeek V3, Kimi K2.5, DeepSeek V4
- [ ] Spec decode + EP 호환
- [ ] 3+ rank (커스텀 MLX ws>2 Metal 지원 시)
- [ ] Standalone vllm-ep 패키지 분리

## 6. Resolved Questions

### Q5: Sampling 동기화 → **해결됨**
- EP logit은 모든 rank에서 bit-exact 동일 (dispatch/combine 결정론적)
- **Greedy**: 각 rank 독립 샘플링 가능 (broadcast 불필요)
- **Sampling**: RNG seed 동기화 필요 → 기존 StepPlan.sampling_seeds로 해결

### Q6: Qwen3.5 GDN Layer → **해결됨**
- **모든 60개 layer에 MoE** (GDN+MoE / GQA+MoE)
- `mlp_only_layers: []` → dense FFN layer 없음
- EP 적용 범위: 60개 layer 전부

### Q7: mlx-lm Qwen3.5 → **해결됨**
- upstream mlx-lm 0.30.7 pip 설치 완료 (hwstudio1)
- `qwen3_5_moe.py` + `qwen3_5.py` 이미 존재 — 추가 작업 불필요
- vllm-mlx는 `mlx-lm>=0.30.5` pip 의존성으로 사용 (포크 아님)
- GDN projection 구조가 Qwen3-Next와 다름 주의

### Q8: Expert weight 형태 → **Phase 1에서 확인 예정**
- Qwen3.5 safetensors index 분석으로 확인
- SwitchGLU stacked `[E, D, inter]` vs 개별 `experts.N.gate_proj`
- EPMoEAdapter.slice_experts() 구현에 직결

## 7. Risk & Mitigation

| 리스크 | 영향 | 완화 |
|--------|------|------|
| Qwen3.5 MoE 구조가 DeepSeek과 다름 | Adapter 재설계 | Phase 1에서 구조 먼저 파악 |
| dispatch/combine 비결정론성 | 출력 발산 | Rank 0 샘플링 + broadcast |
| EP 로딩에서 weight 형태 불일치 | 로딩 실패 | safetensors 사전 분석 |
| Capacity overflow (expert 과집중) | 토큰 드롭 | capacity_factor 튜닝 |
| 커스텀 MLX ws=2 전용 Metal | 3+ rank 불가 | 당분간 2-rank 고정 |
| RDMA 불안정 | 전체 차단 | 기존 TP에서 검증된 JACCL 인프라 재활용 |

## 8. File Change Summary (예상)

```
vllm-mlx/
├── vllm_mlx/
│   ├── cli_args.py                 # --expert-parallel 플래그
│   ├── ep_adapter.py               # (신규) EPMoEAdapter + slice_experts
│   ├── ep_loader.py                # (신규) Selective safetensors loading
│   ├── distributed_launcher.py     # EP 모드 분기
│   ├── scheduler.py                # EP 모드 동기화 간소화
│   └── models/
│       └── llm.py                  # EP 로딩 경로 추가
│
├── docs/development/
│   └── ep-implementation-plan.md   # 이 파일
│
├── tests/
│   ├── test_ep_adapter.py          # (신규) EPMoEAdapter 단위 테스트
│   └── test_ep_loader.py           # (신규) EP 로딩 테스트
```

**mlx-lm 변경 없음** — Adapter 패턴으로 외부에서 교체
