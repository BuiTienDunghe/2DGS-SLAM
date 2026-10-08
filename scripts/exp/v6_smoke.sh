#!/usr/bin/env bash
# plan v6 pre-run checks: smoke of L1 with the new logs (frames 0-200), then a short run of every variant config.
set -uo pipefail
cd "$(dirname "$0")/../.."
export SAVE_DIR=results_m
A="-l --dump-loops"
bash scripts/exp/run_one.sh configs/tum/fr1_room.yaml 0 smoke_v6 $A --range 0 200 2
bash scripts/exp/run_one.sh configs/tum/v6/fr1_room_it30_prev.yaml 0 smoke_it30_prev $A --range 0 60 2
bash scripts/exp/run_one.sh configs/tum/v6/fr1_room_it30_cv.yaml 0 smoke_it30_cv $A --range 0 60 2
bash scripts/exp/run_one.sh configs/tum/v6/fr1_room_it15_cv.yaml 0 smoke_it15_cv $A --range 0 60 2
bash scripts/exp/run_one.sh configs/tum/v6/fr1_room_loaderD.yaml 0 smoke_loaderD $A --range 0 60 2
bash scripts/exp/run_one.sh configs/tum/v6/fr1_room_loaderR.yaml 0 smoke_loaderR $A --range 0 60 2
bash scripts/exp/run_one.sh configs/tum/fr1_room.yaml 0 smoke_odom $A -o --range 0 60 2
echo "[v6_smoke] finished $(date -Is)"
