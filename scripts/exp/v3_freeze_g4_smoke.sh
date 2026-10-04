#!/usr/bin/env bash
# handoffPlan_v3_solver step D: freeze CHOSEN as selected_v4.yaml -> V4 (G4 online == offline on all 5 tuning
# dumps, <= 1e-4 m, e_dl diff < 0.1 mm, <= 120 s per event) -> smoke fr1_room 0-200 -> STOP (no run #3/#5).
# usage: bash scripts/exp/v3_freeze_g4_smoke.sh CHOSEN
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
CH=$1
T=results_exp/tum/room/20261002013837_rigid_s0
R=results_exp/replica/office2/20261002024857_rigid_s0
log=results_exp/logs/v3_freeze.log
step() { echo "== $1 $(date -Is)" | tee -a "$log"; }
stop() { echo "STOP $1 $(date -Is)" | tee -a "$log"; exit "${2:-1}"; }

step "freeze $CH -> selected_v4.yaml"
python tools/freeze_v4.py "$CH" "$T" "$R" 2>&1 | grep -v Warn | tee -a "$log"
(( ${PIPESTATUS[0]} == 0 )) || stop "FREEZE_MISMATCH" 8
sha256sum configs/deform/selected_v4.yaml | tee -a "$log"

step "V4: G4 replay + time on all tuning dumps"
fail=0
for dp in "$T"/loop_dumps/*.pt "$R"/loop_dumps/*.pt; do
  out=$(timeout 1800 python tests/test_online_hook.py "$dp" configs/deform/selected_v4.yaml 2>&1); echo "$out" >> "$log"
  t=$(echo "$out" | grep -oP "t_online=\K[0-9.]+" | head -1)
  echo "$(basename "$dp"): $(echo "$out" | grep -E "^\[|G4_" | tr '\n' ' ') t_online=${t:-NA}" | tee -a "$log"
  grep -q G4_REPLAY_PASS <<< "$out" || fail=1
  if [[ -n "$t" ]] && awk "BEGIN{exit !($t > 120)}"; then echo "TIME > 120 s: $dp" | tee -a "$log"; fail=1; fi
done
(( fail == 0 )) || stop "V4_FAIL" 9

step "smoke (selected_v4, fr1_room 0-200)"
bash scripts/exp/run_one.sh configs/tum/fr1_room.yaml 0 smoke_v4 -l --range 0 200 2 -r 300 --dump-loops \
  --deform-config configs/deform/selected_v4.yaml >> "$log" 2>&1
sd=$(ls -td results_exp/tum/room/*_smoke_v4 | head -1)
python tools/check_run_outputs.py "$sd" --refine >> "$log" 2>&1 || stop "SMOKE_V4 $sd" 10
echo "smoke dir: $sd" | tee -a "$log"
echo "[v3] STEP_D_DONE $(date -Is) - stop here, waiting for the user's approval of step E" | tee -a "$log"
