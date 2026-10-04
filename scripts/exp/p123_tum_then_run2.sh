#!/usr/bin/env bash
# GPU window between run #1 and run #2: G1 replay + P1/P2/P3 on the TUM tuning dumps, then start run #2.
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
d=results_exp/tum/room/20261002013837_rigid_s0
r=results_exp/reports
log=results_exp/logs/p123_tum.log
mkdir -p "$r"
{
echo "== G1 replay $(date -Is)"; python tools/replay_rigid.py "$d"
echo "== P1 $(date -Is)"; python tools/eval_event_metrics.py --out "$r/P1_tum" "$d"
echo "== P2 $(date -Is)"; python tools/build_nodes.py --out "$r/P2_tum" "$d"
echo "== P3 $(date -Is)"; python tools/run_offline_deform.py --out "$r/P3_tum" "$d"
echo "== done $(date -Is)"
} > "$log" 2>&1
echo "[p0] P1-P3 TUM done $(date -Is); starting run #2" >> results_exp/logs/p0_chain.log
bash scripts/exp/run_queue.sh scripts/exp/queue_run2.txt >> results_exp/logs/p0_chain.log 2>&1
echo "[p0] RUN2_DONE $(date -Is)" >> results_exp/logs/p0_chain.log
