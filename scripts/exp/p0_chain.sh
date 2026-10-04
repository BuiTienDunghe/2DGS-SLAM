#!/usr/bin/env bash
# P0 end: two smoke runs (with / without --dump-loops) -> gate G0a -> baseline-lock runs #1, #2 (sequential).
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
mkdir -p results_exp/logs
log=results_exp/logs/p0_chain.log
echo "[p0] start $(date -Is)" | tee -a "$log"
ok=0
for tag in smoke_dump smoke_nodump; do
  extra=""
  [[ "$tag" == smoke_dump ]] && extra="--dump-loops"
  bash scripts/exp/run_one.sh configs/tum/fr1_room.yaml 0 "$tag" -l --range 0 200 2 -r 300 $extra
  d=$(ls -td results_exp/tum/room/*_"$tag" 2>/dev/null | head -1)
  if [[ -n "$d" ]] && python tools/check_run_outputs.py "$d" --refine >> "$log" 2>&1; then
    echo "[p0] $tag OK ($d)" | tee -a "$log"
  else
    echo "[p0] $tag FAILED ($d)" | tee -a "$log"
    ok=1
  fi
done
if (( ok != 0 )); then
  echo "[p0] G0A_FAIL $(date -Is)" | tee -a "$log"
  exit 1
fi
echo "[p0] G0A_PASS $(date -Is) -> baseline runs" | tee -a "$log"
bash scripts/exp/run_queue.sh scripts/exp/queue_baseline.txt 2>&1 | tee -a "$log"
echo "[p0] BASELINE_QUEUE_DONE $(date -Is)" | tee -a "$log"
