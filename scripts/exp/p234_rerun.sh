#!/usr/bin/env bash
# Re-run of the registered repair round after the reliability determinism fix (P3_repair_rule.md, 04:52 note):
# archive v1 -> P2 -> repair override -> P3 (+freeze) -> freeze check -> G4 replay on all 5 dumps. No runs #3-#6.
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
log=results_exp/logs/p234_rerun.log
T=results_exp/tum/room/20261002013837_rigid_s0
R=results_exp/replica/office2/20261002024857_rigid_s0
r=results_exp/reports
step() { echo "== $1 $(date -Is)" | tee -a "$log"; }
stop() { echo "STOP $1 $(date -Is)" | tee -a "$log"; exit "${2:-1}"; }

step "archive v1"
[[ -d $r/P2_v1 ]] || mv $r/P2 $r/P2_v1
[[ -d $r/P3_v1 ]] || mv $r/P3 $r/P3_v1
[[ -f configs/deform/selected_v1.yaml ]] || cp configs/deform/selected.yaml configs/deform/selected_v1.yaml
[[ -f configs/deform/repair_override_v1.yaml ]] || cp configs/deform/repair_override.yaml configs/deform/repair_override_v1.yaml

step "P2"; python tools/build_nodes.py --out "$r/P2" "$T" "$R" >> "$log" 2>&1 || stop "P2_CRASH" 5
step "repair override"
python tools/make_repair_override.py "$r/P2/P2_metrics.json" configs/deform/repair_override.yaml 2>&1 | tee -a "$log"
(( ${PIPESTATUS[0]} == 0 )) || stop "REPAIR_RULE" 6
step "P3"
python tools/run_offline_deform.py --out "$r/P3" --override configs/deform/repair_override.yaml \
  --freeze configs/deform/selected.yaml "$T" "$R" >> "$log" 2>&1 || stop "P3_CRASH" 5
read -r chosen block nocon <<< "$(python -c "
import json; r=json.load(open('$r/P3/P3_metrics.json'))['result']
print(r['chosen'], r['G3_block_worse_in_all_scenes'], r['A_noCon_max_disp_mm'])")"
echo "P3 chosen=$chosen block=$block A_noCon_max_disp_mm=$nocon" | tee -a "$log"
[[ "$block" == "False" ]] || stop "G3_WORSE_IN_ALL_SCENES" 7
step "freeze check"
out=$(python tools/check_freeze.py configs/deform/selected.yaml configs/deform/repair_override.yaml "$chosen" "$T" "$R" 2>&1)
echo "$out" | tee -a "$log"
grep -q FREEZE_OK <<< "$out" || stop "FREEZE_MISMATCH" 8
sha256sum configs/deform/selected.yaml | tee -a "$log"
step "G4 replay on all dumps"
for dp in "$T"/loop_dumps/*.pt "$R"/loop_dumps/*.pt; do
  out=$(timeout 900 python tests/test_online_hook.py "$dp" configs/deform/selected.yaml 2>&1); echo "$out" >> "$log"
  echo "$(basename "$dp"): $(echo "$out" | grep -E "^\[deform\] online accepted|online vs offline|e_dl offline|G4_" | tr '\n' ' ')" | tee -a "$log"
done
echo "[p234] RERUN DONE $(date -Is)" | tee -a "$log"
