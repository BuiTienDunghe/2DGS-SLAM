#!/usr/bin/env bash
# Plan v2 step 11 (TUM only): freeze the selected config as selected_v3.yaml -> G4 replay on the 3 TUM dumps
# -> smoke run fr1_room 0-200 with selected_v3 -> write queue_method_v3.txt (no run #3-#6 is started).
# usage: bash scripts/exp/v2_freeze_g4_smoke.sh CHOSEN
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
CH=$1
T=results_exp/tum/room/20261002013837_rigid_s0
log=results_exp/logs/v2_freeze.log
step() { echo "== $1 $(date -Is)" | tee -a "$log"; }
stop() { echo "STOP $1 $(date -Is)" | tee -a "$log"; exit "${2:-1}"; }

step "freeze $CH"
python tools/freeze_v3.py "$CH" "$T" 2>&1 | grep -v Warn | tee -a "$log"
(( ${PIPESTATUS[0]} == 0 )) || stop "FREEZE_MISMATCH" 8
sha256sum configs/deform/selected_v3.yaml | tee -a "$log"

step "G4 replay on the TUM dumps"
for dp in "$T"/loop_dumps/*.pt; do
  out=$(timeout 900 python tests/test_online_hook.py "$dp" configs/deform/selected_v3.yaml 2>&1); echo "$out" >> "$log"
  echo "$(basename "$dp"): $(echo "$out" | grep -E "^\[|G4_" | tr '\n' ' ')" | tee -a "$log"
  grep -q G4_REPLAY_PASS <<< "$out" || stop "G4_REPLAY $dp" 9
done

step "smoke (selected_v3, fr1_room 0-200)"
bash scripts/exp/run_one.sh configs/tum/fr1_room.yaml 0 smoke_v3 -l --range 0 200 2 -r 300 --dump-loops \
  --deform-config configs/deform/selected_v3.yaml >> "$log" 2>&1
sd=$(ls -td results_exp/tum/room/*_smoke_v3 | head -1)
python tools/check_run_outputs.py "$sd" --refine >> "$log" 2>&1 || stop "SMOKE_V3 $sd" 10
echo "smoke dir: $sd" | tee -a "$log"

step "queue_method_v3.txt (copy of queue_method.txt with selected_v3; queue_method.txt unchanged, not started)"
sed 's#configs/deform/selected.yaml#configs/deform/selected_v3.yaml#; s#deform_s\([01]\)#deform3_s\1#' \
  scripts/exp/queue_method.txt > scripts/exp/queue_method_v3.txt
grep -c selected_v3 scripts/exp/queue_method_v3.txt | tee -a "$log"
echo "[v2] FREEZE_G4_SMOKE_DONE $(date -Is) - runs #3-#6 still need the user's approval" | tee -a "$log"
