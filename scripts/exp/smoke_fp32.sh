#!/usr/bin/env bash
# Causation test for the TF32 finding: same smoke as before, now with allow_tf32=False in frontend/backend.
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
log=results_exp/logs/p0_chain.log
echo "[p0] fp32 smoke $(date -Is)" | tee -a "$log"
bash scripts/exp/run_one.sh configs/tum/fr1_room.yaml 0 smoke3_fp32 -l --range 0 200 2 -r 300 --dump-loops
d=$(ls -td results_exp/tum/room/*_smoke3_fp32 2>/dev/null | head -1)
python tools/check_run_outputs.py "$d" --refine 2>&1 | tee -a "$log"
python tools/replay_rigid.py "$d" 2>&1 | tee -a "$log"
grep -h "allow_tf32" "$(ls -t results_exp/logs/*smoke3_fp32.log | head -1)" | head -2 | tee -a "$log"
cat "$d/metrics_prerefine.csv" | tee -a "$log"
grep -E '"fps_hz"|slam_time_s' "$d/resources.json" | tee -a "$log"
echo "[p0] fp32 smoke done $(date -Is)" | tee -a "$log"
