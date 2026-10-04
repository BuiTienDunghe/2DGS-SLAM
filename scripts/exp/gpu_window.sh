#!/usr/bin/env bash
# GPU window after run #2: G0b Replica -> G1 -> P1/P2 (both scenes) -> registered repair -> P3 (one round, both scenes)
# -> freeze + check -> P4 (online-hook replay + smoke) -> method queue #3-#6. Stops (exit != 0) on blocking gates.
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
log=results_exp/logs/gpu_window.log
T=results_exp/tum/room/20261002013837_rigid_s0
R=$(ls -td results_exp/replica/office2/*_rigid_s0 2>/dev/null | head -1)
r=results_exp/reports
step() { echo "== $1 $(date -Is)" | tee -a "$log"; }
stop() { echo "STOP $1 $(date -Is)" | tee -a "$log"; exit "${2:-1}"; }

step "check run #2 ($R)"
python tools/check_run_outputs.py "$R" --refine >> "$log" 2>&1 || echo "WARN run2 output check failed" | tee -a "$log"
nR=$(wc -l < "$R/loop_events.jsonl" 2>/dev/null || echo 0)
echo "G0B_REPLICA events=$nR" | tee -a "$log"
(( nR > 0 )) || stop "G0B_REPLICA_ZERO_EVENTS" 3

step "G1 replay"
out=$(python tools/replay_rigid.py "$T" "$R" 2>&1); echo "$out" >> "$log"
grep -q REPLAY_ALL_PASS <<< "$out" || stop "G1_REPLAY" 4

step "P1"; python tools/eval_event_metrics.py --out "$r/P1" "$T" "$R" >> "$log" 2>&1 || stop "P1_CRASH" 5
step "P2"; python tools/build_nodes.py --out "$r/P2" "$T" "$R" >> "$log" 2>&1 || stop "P2_CRASH" 5
step "repair override"
python tools/make_repair_override.py "$r/P2/P2_metrics.json" configs/deform/repair_override.yaml 2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}
(( rc == 0 )) || stop "REPAIR_RULE_rc=$rc (2 = bug signature)" 6

step "P3"
python tools/run_offline_deform.py --out "$r/P3" --override configs/deform/repair_override.yaml \
  --freeze configs/deform/selected.yaml "$T" "$R" >> "$log" 2>&1 || stop "P3_CRASH" 5
read -r chosen block nocon <<< "$(python -c "
import json; r=json.load(open('$r/P3/P3_metrics.json'))['result']
print(r['chosen'], r['G3_block_worse_in_all_scenes'], r['A_noCon_max_disp_mm'])")"
echo "P3 chosen=$chosen block=$block A_noCon_max_disp_mm=$nocon" | tee -a "$log"
[[ "$chosen" != "None" ]] || stop "P3_NO_CHOICE" 7
[[ "$block" == "False" ]] || stop "G3_WORSE_IN_ALL_SCENES" 7
python -c "import sys; sys.exit(0 if float('$nocon') < 0.1 else 1)" || stop "G3_ANOCON_NOT_RIGID" 7
step "freeze check"
python tools/check_freeze.py configs/deform/selected.yaml configs/deform/repair_override.yaml "$chosen" "$T" "$R" 2>&1 | tee -a "$log"
grep -q FREEZE_OK "$log" || stop "FREEZE_MISMATCH" 8
sha256sum configs/deform/selected.yaml | tee -a "$log"

step "P4 online-hook replay"
dumps=$(python -c "
import json
acc, other = [], []
for l in open('$r/P3/events.jsonl'):
    e = json.loads(l); p = e['run'] + '/loop_dumps/' + e['dump']
    (acc if e['logs']['$chosen']['accepted'] else other).append((e['scene'], p))
seen, pick = set(), []
for sc, p in acc + other:
    if sc not in seen: seen.add(sc); pick.append(p)
print(' '.join(pick))")
for dp in $dumps; do
  out=$(timeout 900 python tests/test_online_hook.py "$dp" configs/deform/selected.yaml 2>&1); echo "$out" >> "$log"
  echo "$out" | grep -E "^\[|G4_" | tee -a "$log"
  grep -q G4_REPLAY_PASS <<< "$out" || stop "G4_REPLAY $dp" 9
done
step "P4 smoke (deform config, fr1_room 0-200)"
bash scripts/exp/run_one.sh configs/tum/fr1_room.yaml 0 smoke_deform -l --range 0 200 2 -r 300 --dump-loops \
  --deform-config configs/deform/selected.yaml >> "$log" 2>&1
sd=$(ls -td results_exp/tum/room/*_smoke_deform | head -1)
python tools/check_run_outputs.py "$sd" --refine >> "$log" 2>&1 || stop "P4_SMOKE" 10
echo "[p4] GATE_G4_PASS $(date -Is)" | tee -a "$log"

# runs #3-#6 need the user's explicit approval (2026-10-02 04:15): stop here
echo "[p4] P1-P4 DONE, waiting for user approval before runs #3-#6 $(date -Is)" | tee -a "$log" results_exp/logs/p0_chain.log
