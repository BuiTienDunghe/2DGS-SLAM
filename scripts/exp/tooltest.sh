#!/usr/bin/env bash
# Crash test of the P1-P4 tooling on one small real dump (TF32 smoke2). Numbers are irrelevant here.
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
d=$(ls -td results_exp/tum/room/*_smoke2_dump | head -1)
o=results_exp/tooltest
rm -rf "$o"; mkdir -p "$o"
echo "== P1 eval_event_metrics"; timeout 600 python tools/eval_event_metrics.py --out "$o/P1" "$d" 2>&1 | tail -15
echo "== P2 build_nodes"; timeout 600 python tools/build_nodes.py --out "$o/P2" "$d" 2>&1 | tail -6
echo "== P3 run_offline_deform (grid min)"; timeout 900 python tools/run_offline_deform.py --grid min --out "$o/P3" "$d" 2>&1 | tail -25
echo "== P4 test_online_hook"; timeout 600 python tests/test_online_hook.py "$d"/loop_dumps/*.pt 2>&1 | tail -8
echo "== done $(date -Is)"
