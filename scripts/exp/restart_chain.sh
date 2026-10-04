#!/usr/bin/env bash
# After the backend fix: one smoke run with --dump-loops (exercises the rigid loop branch + dT_online),
# output check + GPU replay (G1 on the smoke dump), then the baseline queue (#1, #2).
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
log=results_exp/logs/p0_chain.log
echo "[p0] restart after backend fix $(date -Is)" | tee -a "$log"
bash scripts/exp/run_one.sh configs/tum/fr1_room.yaml 0 smoke2_dump -l --range 0 200 2 -r 300 --dump-loops
d=$(ls -td results_exp/tum/room/*_smoke2_dump 2>/dev/null | head -1)
if [[ -n "$d" ]] && python tools/check_run_outputs.py "$d" --refine >> "$log" 2>&1 \
   && python tools/replay_rigid.py "$d" 2>&1 | tee -a "$log" | grep -q REPLAY_ALL_PASS; then
  echo "[p0] smoke2_dump OK + replay PASS ($d)" | tee -a "$log"
else
  echo "[p0] smoke2_dump FAILED ($d)" | tee -a "$log"
  exit 1
fi
echo "[p0] G0A_PASS (re-check) $(date -Is) -> baseline runs" | tee -a "$log"
bash scripts/exp/run_queue.sh scripts/exp/queue_baseline.txt 2>&1 | tee -a "$log"
echo "[p0] BASELINE_QUEUE_DONE $(date -Is)" | tee -a "$log"
