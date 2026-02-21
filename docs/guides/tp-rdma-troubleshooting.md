# TP 분산 추론 RDMA 트러블슈팅 가이드

2x Mac Studio M3 Ultra 512GB 환경에서 Thunderbolt 5 RDMA를 활용한 Tensor Parallel 분산 추론 디버깅 과정에서 발견된 문제들과 해결 방법을 정리한 문서입니다.

---

## 1. 환경

| 항목 | 상세 |
|------|------|
| **하드웨어** | 2x Mac Studio M3 Ultra 512GB (hwstudio1, hwstudio2) |
| **RDMA 연결** | Thunderbolt 5 (en5), 직결 케이블 |
| **RDMA IP** | hwstudio1: `10.254.0.5`, hwstudio2: `10.254.0.6` (서브넷 /30) |
| **RDMA 백엔드** | JACCL (MLX 내장) |
| **MLX 버전** | 0.30.6 (pip wheel) |
| **Python** | 3.14 (`/opt/homebrew/bin/python3.14`) |
| **OS** | macOS Sequoia |
| **테스트 모델** | Moonlight-16B-A3B (9GB), Kimi K2.5 (612GB MoE) |

---

## 2. 검증 완료 사항

### 2.1 deepseek_v3 아키텍처 TP 샤딩 수학적 정확성

Moonlight-16B-A3B 모델로 단일 노드 출력과 TP=2 분산 출력을 비교한 결과, **17개 토큰의 token ID가 완벽하게 일치**하는 것을 확인했습니다. 이는 attention, MoE gate, expert 분배 등 모든 TP 샤딩 로직이 정확함을 의미합니다.

### 2.2 JACCL MeshGroup QP 리소스 병목 없음

`MAX_SEND_WR=32`로 설정되어 있으나, TP=2 환경에서 동시에 outstanding 상태인 Work Request는 최대 4개입니다. 각 `all_reduce` 연산이 종료 시 `in_flight=0`까지 drain하므로 WR 누적이 발생하지 않습니다. 따라서 QP 리소스 병목은 설계상 발생하지 않습니다.

---

## 3. 발견된 문제들 (시행착오 포함)

### 3.1 RDMA en5 GID 테이블 손상

**증상:**
- `ifconfig en5 down/up`을 반복 실행한 후 JACCL 연결 시도 시 다음 에러 발생:
  - `errno 96` (EAFNOSUPPORT)
  - `errno 22` (EINVAL)

**원인:**
- en5 인터페이스를 반복적으로 내렸다 올리면 커널의 GID(Global Identifier) 테이블이 손상됩니다. RDMA 통신에 필요한 주소 정보가 불일치 상태가 되어 연결이 실패합니다.

**해결:**
1. 양쪽 스튜디오를 완전히 재부팅합니다.
2. 재부팅 후 `rdma_setup.py`를 재실행하여 RDMA 환경을 초기화합니다.

**교훈:**
> **en5를 절대 반복적으로 down/up 하지 마십시오.** RDMA 복구가 필요한 경우에도 1회만 수행하고, 문제가 지속되면 재부팅하십시오.

---

### 3.2 IBV 디바이스 설정 파일 포맷 불일치

**증상:**
```
[jaccl] The device file should start with an array
```

**원인:**
- `rdma_setup.py`가 생성하는 `/tmp/mlx_ibv_devices.json` 파일 포맷:
  ```json
  {"devices": ["rdma_en5"]}
  ```
- JACCL이 기대하는 매트릭스 포맷:
  ```json
  [[null, "rdma_en5"], ["rdma_en5", null]]
  ```
- JACCL은 NxN 매트릭스 형태로 각 rank 쌍 간의 RDMA 디바이스를 지정합니다. `matrix[i][j]`는 rank i에서 rank j로 통신할 때 사용하는 RDMA 디바이스를 의미하며, 자기 자신(`matrix[i][i]`)은 `null`입니다.

**해결:**
- `/tmp/mlx_ibv_devices.json`에 수동으로 매트릭스 포맷을 작성합니다:
  ```json
  [[null, "rdma_en5"], ["rdma_en5", null]]
  ```

**TODO:** `rdma_setup.py`를 수정하여 JACCL 매트릭스 포맷을 자동 생성하도록 해야 합니다.

---

### 3.3 Kimi K2.5 TP=2 Step 22 ENOMEM 크래시

**증상:**
- Step 1~21까지 완벽하게 동작합니다 (양쪽 rank의 logprobs가 동일).
- Step 22에서 크래시가 발생합니다:
  ```
  [jaccl] Recv failed with error code -12
  ```
  (에러 코드 -12 = ENOMEM)

**시행착오:**
1. `MLX_METAL_FAST_SYNCH=1` 환경 변수 제거 후 실행 --> 동일 크래시
2. `FAST_SYNCH` 없이 실행 --> 동일 크래시
3. 매 step마다 `mx.eval()` 강제 실행 --> 동일 크래시

위 시도들로 문제가 해결되지 않아, JACCL 내부 구조를 분석하게 되었습니다.

**분석 결과:**

JACCL의 `MAX_SEND_WR=32`는 원인이 아닙니다. 각 `all_reduce`가 자체적으로 drain하므로 WR이 누적되지 않습니다.

핵심 원인은 **macOS Sequoia(< 26.3)에서 모든 RDMA 전송이 4KB 청크로 강제**되는 것입니다:

```cpp
// mlx/mlx/backend/common/jaccl/utils.h
std::pair<int, int64_t> buffer_size_from_message(int64_t msg) {
  if (__builtin_available(macOS 26.3, ...)) {
    // Tahoe 이상에서만 8KB ~ 512KB 버퍼 선택
    for (int i = BUFFER_SIZES - 1; i >= 0; i--) { ... }
  }
  return {0, FRAME_SIZE};  // Sequoia에서는 항상 4KB (FRAME_SIZE=4096)
}
```

이로 인한 RDMA 연산 수 폭발:
```
61 레이어 x 2 all_reduce/레이어 x 4 청크/op x 2 방향 x 21 steps
= ~40,000+ RDMA operations
```

Apple RDMA 드라이버 내부의 리소스(DMA 매핑, 페이지 테이블 등)가 고갈되면서 ENOMEM이 발생하는 것으로 추정됩니다.

**잠정 해결 방안:**
1. MLX를 소스에서 빌드합니다.
2. `buffer_size_from_message()` 함수에서 `__builtin_available(macOS 26.3)` 가드를 제거합니다.
3. 512KB 청크를 사용하면 RDMA 연산 수가 약 **128배 감소**합니다.

> 주의: pip wheel은 이미 컴파일된 바이너리이므로 이 수정이 불가능합니다. 반드시 소스 빌드가 필요합니다.

---

### 3.4 버그 5: BatchGenerator dist_group 미전달

**증상:**
- vllm-mlx 서버에서 TP=2로 서빙 시, 양 rank가 독립적으로 샘플링합니다.
- KV cache가 발산하여 출력이 corruption됩니다.
- 짧은 출력은 정상처럼 보이지만, 장문 생성 시 깨진 텍스트가 출력됩니다.

**원인:**
- `scheduler.py`와 `distributed_launcher.py`에서 `BatchGenerator` 생성 시 `dist_group` 파라미터를 전달하지 않았습니다.
- `dist_group`이 없으면 각 rank가 독립적으로 토큰을 샘플링하므로, step이 진행될수록 KV cache 내용이 달라집니다.

**수정:**
```python
# scheduler.py:1429
BatchGenerator(..., dist_group=communicator.group)

# distributed_launcher.py:513
BatchGenerator(..., dist_group=communicator.group)
```

**추가 발견:**
- `_broadcast_sampled_tokens`와 `_synced_step`이 이중 동기화를 수행할 가능성이 있습니다. `dist_group` 전달로 `_synced_step`이 정상 동작하면, `_broadcast_sampled_tokens`는 불필요해질 수 있습니다.

---

### 3.5 macOS Local Network Privacy 차단

**증상:**
- Homebrew Python 3.14 (ad-hoc signed)로 JACCL coordinator에 연결 시, macOS가 로컬 네트워크 접근을 차단합니다.
- 연결 타임아웃 또는 무반응이 발생합니다.

**원인:**
- macOS는 ad-hoc 서명된 바이너리의 로컬 네트워크 접근을 기본적으로 차단합니다.
- Homebrew로 설치한 Python은 ad-hoc 서명이 적용되어 있습니다.

**해결:**
TCC.db에 직접 권한을 삽입합니다:
```bash
sqlite3 ~/Library/Application\ Support/com.apple.TCC/TCC.db \
  "INSERT OR REPLACE INTO access \
    (service, client, client_type, auth_value, auth_reason, auth_version, \
     csreq, indirect_object_identifier, flags, last_modified, boot_uuid) \
   VALUES \
    ('kTCCServiceLocalNetwork', '<codesign-identifier>', 1, 2, 4, 1, \
     NULL, 'UNUSED', 0, CAST(strftime('%s','now') AS INTEGER), 'UNUSED')"
```

> `<codesign-identifier>`는 `codesign -dv /opt/homebrew/bin/python3.14 2>&1 | grep Identifier`로 확인합니다.

---

### 3.6 TB5 RDMA IP 미설정 / TB5를 SSH에도 사용 가능

**사실 확인:**
- TB5 (en5)는 RDMA 전용이 아닙니다. 일반 TCP/IP 통신(SSH, rsync 등)도 가능합니다.
- RDMA IP를 설정하면 같은 인터페이스로 RDMA와 TCP 양쪽 모두 사용할 수 있습니다.

**설정값:**
| 노드 | RDMA IP |
|------|---------|
| hwstudio1 | `10.254.0.5/30` |
| hwstudio2 | `10.254.0.6/30` |

---

## 4. JACCL 아키텍처 분석 (MLX 0.30.7 기준)

### 4.1 주요 상수 (`utils.h`)

```cpp
MAX_SEND_WR  = 32    // 최대 동시 Send Work Request
MAX_RECV_WR  = 32    // 최대 동시 Recv Work Request
BUFFER_SIZES = 8     // 버퍼 크기 단계 수 (4KB ~ 512KB)
NUM_BUFFERS  = 2     // 이중 버퍼링용 버퍼 수
FRAME_SIZE   = 4096  // 기본 프레임 크기 (4KB)
```

### 4.2 MeshGroup 동작 (TP=2)

TP=2 환경에서 MeshGroup의 구성:

| 리소스 | 수량 |
|--------|------|
| Connection | 1개 |
| Queue Pair (QP) | 1개 |
| Completion Queue (CQ) | 64 엔트리 |

**all_reduce 동작 흐름:**
1. `PIPELINE=2`로 이중 버퍼링을 사용합니다.
2. 데이터를 청크 단위로 분할하여 전송합니다.
3. 각 연산 종료 시 `in_flight=0`이 될 때까지 drain합니다.
4. 따라서 WR이 누적되지 않으며, QP 리소스 병목은 발생하지 않습니다.

### 4.3 4KB 청크 제한 (Sequoia)

```cpp
std::pair<int, int64_t> buffer_size_from_message(int64_t msg) {
  if (__builtin_available(macOS 26.3, ...)) {
    // Tahoe(macOS 26.3) 이상에서만 큰 버퍼 사용 가능
    // 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB 중 선택
    for (int i = BUFFER_SIZES - 1; i >= 0; i--) {
      if (msg >= (1 << (i + 1)) * FRAME_SIZE) {
        return {i, (1 << (i + 1)) * FRAME_SIZE};
      }
    }
  }
  // Sequoia에서는 항상 4KB 프레임 사용
  return {0, FRAME_SIZE};
}
```

**영향:**
- Sequoia에서는 모든 RDMA 전송이 4KB 단위로 분할됩니다.
- 대형 모델(Kimi K2.5 등)에서 레이어 수가 많으면 RDMA 연산 수가 기하급수적으로 증가합니다.
- Apple RDMA 드라이버의 내부 리소스 한계에 도달할 수 있습니다.

---

## 5. 운영 주의사항

### 5.1 서버 시작 전 확인 사항

1. **Wired memory 확인:**
   ```bash
   vm_stat | grep wired
   ```
   - 정상: ~340,000 pages (약 5GB)
   - 비정상: 23,000,000+ pages이면 Metal 메모리 누수 --> 반드시 재부팅

2. **좀비 프로세스 확인:**
   - 이전 세션의 벤치마크 스크립트(`bench_concurrent.py`)가 살아남아 서버에 추가 요청을 보낼 수 있습니다.

3. **JACCL coordinator 포트에 TCP 사전 테스트 금지:**
   - coordinator가 테스트 연결을 실제 rank 연결로 오인하여 `Recv failed with errno=2` 에러가 발생합니다.

### 5.2 프로세스 종료 순서

1. **반드시 `kill -15` (SIGTERM)을 먼저** 보냅니다.
2. 응답이 없으면 `kill -9` (SIGKILL)을 사용합니다.
3. **양쪽 노드를 동시에 종료**합니다.

> **SIGKILL 위험성:** RDMA 연산 중 SIGKILL을 보내면 Metal GPU hang이 발생하고, 이는 커널 패닉으로 이어질 수 있습니다. SIGKILL 후 Metal wired memory가 해제되지 않으며, 이 경우 재부팅만이 유일한 해결책입니다.

### 5.3 RDMA 복구

- en5 인터페이스 복구 (양쪽 동시 실행):
  ```bash
  sudo ifconfig en5 down && sleep 10 && sudo ifconfig en5 10.254.0.X/30 up
  ```
- RDMA 디바이스 확인: `ibv_devices` (`ifconfig`에는 RDMA 디바이스가 표시되지 않습니다)
- RDMA 활성화 상태 확인: `rdma_ctl status`

### 5.4 nohup 필수

`ssh -f`로 서버를 시작하면 SSH 세션 종료 시 uvicorn이 SIGTERM을 받아 종료됩니다. 반드시 `/usr/bin/nohup`을 사용하십시오.

---

## 6. 다음 단계

1. **MLX 소스 빌드**: `__builtin_available(macOS 26.3)` 가드를 제거하여 512KB 청크를 활성화합니다.
2. **Kimi K2.5 TP=2 재테스트**: 512KB 청크로 Step 22 이후 안정성을 검증합니다.
3. **`rdma_setup.py` 수정**: JACCL 매트릭스 포맷(`[[null, "rdma_en5"], ...]`)을 자동 생성하도록 합니다.
4. **버그 5 수정 검증**: `dist_group` 전달 후 장문 출력 corruption이 해결되는지 확인합니다.
5. **`_broadcast_sampled_tokens` 제거 테스트**: `_synced_step`만으로 충분한지 검증합니다.
