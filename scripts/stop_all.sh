#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# stop_all.sh
#
# Orchestrates graceful shutdown of vllm-mlx distributed servers across
# both Mac Studio nodes from the local macbook.
#
# Topology:
#   macbook -> hwstudio1 (Tailscale: 100.120.177.62)  [direct SSH]
#   macbook -> hwstudio2 (via hwstudio1 proxy)         [SSH hop]
#
# Usage:
#   ./stop_all.sh
# ============================================================================

set -uo pipefail

HWSTUDIO1_IP="100.120.177.62"
STOP_SCRIPT="/Users/hw/vllm-mlx/scripts/stop_server.sh"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] [orchestrator] $*"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "=========================================="
    log "Stopping vllm-mlx on ALL nodes"
    log "=========================================="

    # Stop hwstudio2 first (worker node / rank 1)
    # It goes through hwstudio1 as a proxy since Tailscale to hwstudio2
    # can be unreliable from the macbook
    log ""
    log ">>> Stopping hwstudio2 (via hwstudio1 proxy)..."
    log "------------------------------------------"
    ssh $SSH_OPTS "$HWSTUDIO1_IP" "ssh $SSH_OPTS hwstudio2 'bash $STOP_SCRIPT'" 2>&1
    local rc2=$?

    log ""

    # Stop hwstudio1 (coordinator / rank 0)
    log ">>> Stopping hwstudio1..."
    log "------------------------------------------"
    ssh $SSH_OPTS "$HWSTUDIO1_IP" "bash $STOP_SCRIPT" 2>&1
    local rc1=$?

    # Summary
    log ""
    log "=========================================="
    log "SUMMARY"
    log "=========================================="
    if [ "$rc1" -eq 0 ]; then
        log "  hwstudio1: OK"
    else
        log "  hwstudio1: FAILED (exit code $rc1)"
    fi
    if [ "$rc2" -eq 0 ]; then
        log "  hwstudio2: OK"
    else
        log "  hwstudio2: FAILED (exit code $rc2)"
    fi

    if [ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]; then
        log ""
        log "All nodes stopped successfully."
    else
        log ""
        log "Some nodes had issues. Check output above for details."
    fi

    # If either node returned non-zero (which includes SIGKILL usage),
    # suggest cooldown for JACCL RDMA resources
    if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
        log ""
        log "WARNING: One or more nodes required forced shutdown (SIGKILL)."
        log "  Waiting 30s for JACCL RDMA resource cooldown to avoid EBUSY errors..."
        log "  (Set SKIP_COOLDOWN=1 to bypass)"
        if [ "${SKIP_COOLDOWN:-0}" != "1" ]; then
            sleep 30
            log "  Cooldown complete."
        else
            log "  Cooldown skipped (SKIP_COOLDOWN=1)."
        fi
    fi

    log "=========================================="

    # Return non-zero if any node failed
    if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
        exit 1
    fi
}

main "$@"
