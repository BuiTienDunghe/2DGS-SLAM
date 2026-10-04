#!/usr/bin/env bash
# Wait until ptrace_scope is 0 (CUDA IPC needs it), then run #2 (Replica baseline) -> GPU window -> method runs.
set -uo pipefail
cd "$(dirname "$0")/../.."
log=results_exp/logs/p0_chain.log
echo "[p0] waiting for ptrace_scope=0 $(date -Is)" >> "$log"
until [[ "$(cat /proc/sys/kernel/yama/ptrace_scope)" == "0" ]]; do sleep 15; done
echo "[p0] ptrace_scope=0, starting run #2 $(date -Is)" >> "$log"
bash scripts/exp/run_queue.sh scripts/exp/queue_run2.txt >> "$log" 2>&1
echo "[p0] RUN2B_DONE $(date -Is)" >> "$log"
bash tools/stop_all.sh > /dev/null
bash scripts/exp/gpu_window.sh
echo "[p0] gpu_window exit=$? $(date -Is)" >> "$log"
