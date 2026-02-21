# vllm-mlx Project Instructions

## 정지 지시 (STOP) — 최우선 규칙

사용자가 "하지마", "멈춰", "스톱", "중지" 등 정지 지시를 하면 **즉시 모든 작업을 멈출 것.**
- 도구 호출 금지 (Bash, Read, Task, 서브에이전트 등 전부)
- "네 알겠습니다" 하고 다음 줄에서 도구 호출하는 행위 절대 금지
- 작업이 미완료여도 상관없음 — 사용자 지시가 태스크 완료보다 항상 우선
- 사용자가 명시적으로 다시 작업을 요청할 때까지 대기

## 필수: 프로젝트 문서 참조

작업 전 `docs/` 디렉토리의 문서를 반드시 읽고 프로젝트 구조와 설계, 그리고 최근 발생한 이슈 등을 파악할 것.

## RDMA Setup (재부팅 후 필수)

재부팅하면 TB5 RDMA IP와 `/tmp/mlx_ibv_devices.json`이 초기화됩니다. **반드시 이 macbook에서** 아래 명령어를 실행하세요:

```bash
python3 scripts/rdma_setup.py \
    --node hwstudio1:en5:10.254.0.5 \
    --node hwstudio2:en5:10.254.0.6 \
    --netmask 30
```

스크립트가 JACCL 호환 NxN 매트릭스 포맷(`[[null, "rdma_en5"], ["rdma_en5", null]]`)을 자동 생성합니다.

자세한 내용: `docs/guides/rdma-setup.md`

## 분산 테스트 실행

Rank 0을 먼저 시작하고, 5초 후 Rank 1을 시작합니다:

```bash
ssh hwstudio1 "bash /tmp/run_tp_kimi.sh 0" &
sleep 5
ssh hwstudio1 "ssh hw@10.254.0.6 'bash /tmp/run_tp_kimi.sh 1'" &
wait
```

## IBV Config (`/tmp/mlx_ibv_devices.json`)

- **수동 배포 불필요** — `mlx.launch`가 JACCL backend + hostfile로 시작하면 자동으로 RDMA device mapping 적용
- 시작 스크립트(`~/start_server_rank.sh`)가 hostfile 경로를 지정하므로 별도 IBV config 배포 필요 없음
- 재부팅 후 `/tmp/mlx_ibv_devices.json`이 사라져도, 서버 시작 시 자동 생성됨

## 주의사항

- **RDMA en5를 반복적으로 down/up 하지 말 것** — GID 테이블 손상됨, 재부팅만이 복구 방법
- **서버 종료 시 kill -15 먼저**, 안 되면 kill -9 — 양쪽 동시에
- **kill -9 후 wired memory 확인** — `vm_stat | grep wired`, 23M+ pages면 재부팅 필수
- **JACCL coordinator 포트(32323)에 TCP 테스트 금지** — coordinator가 오인함
- **/tmp에 중요 스크립트 저장 금지** — 재부팅 시 삭제됨
