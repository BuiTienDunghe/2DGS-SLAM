#!/usr/bin/env bash
# Run a queue file sequentially (one SLAM run at a time, R6). A failed run is logged and the
# queue moves on. Queue file: one run per line: CONFIG SEED TAG [extra slam.py args...]; '#' = comment.
# usage: scripts/exp/run_queue.sh QUEUE_FILE
set -uo pipefail
cd "$(dirname "$0")/../.."
q=$1
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
  # shellcheck disable=SC2086
  bash scripts/exp/run_one.sh $line || echo "[run_queue] run failed: $line"
done < "$q"
echo "[run_queue] queue finished $(date -Is)"
