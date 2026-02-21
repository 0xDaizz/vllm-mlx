# vllm-mlx Project Instructions

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

## 주의사항

- **RDMA en5를 반복적으로 down/up 하지 말 것** — GID 테이블 손상됨, 재부팅만이 복구 방법
- **서버 종료 시 kill -15 먼저**, 안 되면 kill -9 — 양쪽 동시에
- **kill -9 후 wired memory 확인** — `vm_stat | grep wired`, 23M+ pages면 재부팅 필수
- **JACCL coordinator 포트(32323)에 TCP 테스트 금지** — coordinator가 오인함
- **/tmp에 중요 스크립트 저장 금지** — 재부팅 시 삭제됨
