#!/usr/bin/env bash
# plan v6 quick: short smoke of the E5 code paths (frames 0-240, lowered revisit gate), not a measurement
cd "$(dirname "$0")/../.."
bash scripts/exp/run_one_wt.sh configs/tum/v6q/fr1_room_E5_smoke.yaml 0 smoke_e5 -l --range 0 240 2 -r 300 --dump-loops
echo "V6Q_SMOKE_DONE $(date -Is)"
