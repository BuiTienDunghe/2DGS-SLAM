#!/usr/bin/env bash
# handoffPlan_v5_persistence step C analysis (after both v6 runs finished; GPU, sequential):
#  - end-of-run table (6 runs), per-event online == offline + R on Pi* (birth layers) for the v6 runs,
#  - durability series (t0 layers) for the v6 runs (baseline / v5 series come from results_exp/reports/v4/E/dur),
#  - exact PGO replay on the v6 dumps (stored loop_transform): does the next event damage the previous region,
#  - figures.
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
B0=results_exp/tum/room/20261002013837_rigid_s0; B1=results_exp/tum/room/20261003123233_rigid_s1
M0=results_exp/tum/room/20261003105219_deform5_s0; M1=results_exp/tum/room/20261003114156_deform5_s1
V0=$(ls -td results_exp/tum/room/*_deform6_s0 | head -1); V1=$(ls -td results_exp/tum/room/*_deform6_s1 | head -1)
O=results_exp/reports/v5/C
mkdir -p $O
echo "== runs $V0 $V1 $(date -Is)"
python tools/summarize_v3E.py --out $O $B0 $B1 $M0 $M1 $V0 $V1 2>&1 | grep -v "^    "
echo "== events v6 (birth layers) $(date -Is)"
timeout 3600 python tools/e_v4_events.py --config configs/deform/selected_v6.yaml --layers config --out $O/ev_v6 $V0 $V1 2>&1 | grep -v Warn | grep "^{\|^\[e_v4"
echo "== events v6 (active layers, comparable with v4) $(date -Is)"
timeout 3600 python tools/e_v4_events.py --config configs/deform/selected_v6.yaml --layers active --no-offline --out $O/ev_v6_active $V0 $V1 2>&1 | grep -v Warn | grep "^{\|^\[e_v4"
echo "== durability v6 $(date -Is)"
timeout 3600 python tools/e_v4_durability.py --config configs/deform/selected_v6.yaml --out $O/dur_v6 $V0 $V1 2>&1 | grep -v Warn | grep "^{\|^\[e_v4\|Error\|Traceback"
echo "== h1 replay v6 $(date -Is)"
timeout 3600 python tools/h1_replay.py --config configs/deform/selected_v6.yaml --out $O/h1_v6 $V0 $V1 2>&1 | grep -v Warn | grep -E "^\[loops\]|^\{|^   \(|h1_replay|Error|Traceback"
echo "== plots $(date -Is)"
cat results_exp/reports/v4/E/dur/durability.jsonl $O/dur_v6/durability.jsonl > $O/durability_all.jsonl
CUDA_VISIBLE_DEVICES= python tools/plot_v4E.py --out $O/fig --durability $O/durability_all.jsonl "baseline seed 0 (#1)=$B0" "baseline seed 1=$B1" "v5 seed 0=$M0" "v5 seed 1=$M1" "v6 seed 0=$V0" "v6 seed 1=$V1" 2>&1 | tail -5
echo "V5_STEP_C_ANALYSIS_DONE $(date -Is)"
