#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# stop_server.sh
#
# Graceful shutdown script for vllm-mlx distributed server processes.
# Designed to run standalone on each Mac Studio node (hwstudio1, hwstudio2).
#
# Steps:
#   1. Find all python3.14 processes running vllm_mlx.distributed_launcher
#   2. Send SIGTERM for graceful shutdown
#   3. Wait up to 10 seconds for processes to exit
#   4. Send SIGKILL if any remain
#   5. Check ports 32323 (JACCL) and 8000 (HTTP) for lingering binds
#   6. Print final status
#
# Usage:
#   ./stop_server.sh
# ============================================================================

set -uo pipefail

HOSTNAME_LABEL="$(hostname -s)"
GRACEFUL_TIMEOUT="${GRACEFUL_TIMEOUT:-10}"
PORT_JACCL=32323
PORT_HTTP=8000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] [$HOSTNAME_LABEL] $*"
}

# ---------------------------------------------------------------------------
# Find vllm_mlx python processes
# ---------------------------------------------------------------------------
find_vllm_pids() {
    # Match python3.14 processes running the distributed launcher specifically
    # Avoids matching unrelated python processes (e.g. tests, CLI tools)
    pgrep -f 'python3\.14.*vllm_mlx\.distributed_launcher' 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Check if a port is still bound
# ---------------------------------------------------------------------------
check_port() {
    local port="$1"
    local label="$2"
    # lsof is available on macOS by default
    local result
    result=$(lsof -iTCP:"$port" -sTCP:LISTEN -P -n 2>/dev/null || true)
    if [ -n "$result" ]; then
        log "WARNING: Port $port ($label) is still bound:"
        echo "$result" | while IFS= read -r line; do
            log "  $line"
        done
        return 1
    else
        log "OK: Port $port ($label) is free"
        return 0
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "=========================================="
    log "vllm-mlx server shutdown starting"
    log "=========================================="

    # Step 1: Find processes
    local pids
    pids=$(find_vllm_pids)

    local used_sigkill=0

    if [ -z "$pids" ]; then
        log "No vllm_mlx processes found. Nothing to stop."
    else
        local pid_list
        pid_list=$(echo "$pids" | tr '\n' ' ')
        local pid_count
        pid_count=$(echo "$pids" | wc -l | tr -d ' ')
        log "Found $pid_count vllm_mlx process(es): $pid_list"

        # Step 2: Send SIGTERM
        log "Sending SIGTERM for graceful shutdown..."
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null && log "  SIGTERM -> PID $pid" || true
            fi
        done

        # Step 3: Wait up to GRACEFUL_TIMEOUT seconds
        log "Waiting up to ${GRACEFUL_TIMEOUT}s for graceful shutdown..."
        local elapsed=0
        while [ "$elapsed" -lt "$GRACEFUL_TIMEOUT" ]; do
            local remaining_pids
            remaining_pids=$(find_vllm_pids)
            if [ -z "$remaining_pids" ]; then
                log "All processes exited gracefully after ${elapsed}s."
                break
            fi
            sleep 1
            elapsed=$((elapsed + 1))
        done

        # Step 4: SIGKILL if any remain
        local remaining_pids
        remaining_pids=$(find_vllm_pids)
        if [ -n "$remaining_pids" ]; then
            local remaining_list
            remaining_list=$(echo "$remaining_pids" | tr '\n' ' ')
            log "WARNING: Processes still alive after ${GRACEFUL_TIMEOUT}s: $remaining_list"
            log "Sending SIGKILL..."
            for pid in $remaining_pids; do
                if kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null && log "  SIGKILL -> PID $pid" || true
                fi
            done
            sleep 1

            used_sigkill=1

            # Verify kill worked
            local final_pids
            final_pids=$(find_vllm_pids)
            if [ -n "$final_pids" ]; then
                log "ERROR: Some processes could not be killed: $(echo "$final_pids" | tr '\n' ' ')"
            else
                log "All processes terminated after SIGKILL."
                log "WARNING: SIGKILL was used — JACCL RDMA resources may need ~30s cooldown before restart."
            fi
        fi
    fi

    # Step 5: Check ports
    log "------------------------------------------"
    log "Checking ports..."
    local port_issues=0
    check_port "$PORT_JACCL" "JACCL coordinator" || port_issues=$((port_issues + 1))
    check_port "$PORT_HTTP" "HTTP server" || port_issues=$((port_issues + 1))

    # Step 6: Final status
    log "------------------------------------------"
    local final_pids
    final_pids=$(find_vllm_pids)
    if [ -z "$final_pids" ] && [ "$port_issues" -eq 0 ]; then
        log "STATUS: CLEAN - All processes stopped, all ports free."
        if [ "$used_sigkill" -eq 1 ]; then
            log "NOTE: SIGKILL was used. Wait ~30s before restarting to avoid JACCL RDMA EBUSY errors."
        fi
    else
        log "STATUS: WARNING"
        if [ -n "$final_pids" ]; then
            log "  - Remaining processes: $(echo "$final_pids" | tr '\n' ' ')"
        fi
        if [ "$port_issues" -gt 0 ]; then
            log "  - $port_issues port(s) still bound (see above)"
        fi
        log "=========================================="
        exit 1
    fi
    log "=========================================="
}

main "$@"
