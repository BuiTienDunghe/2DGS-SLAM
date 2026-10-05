#!/usr/bin/env bash
# P vs D diagnostic: gap on the regions of earlier loops after D / P0 (GPU, no SLAM).
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
R=results_exp/tum/room
if [ -n "$(ps aux | grep "[s]lam.py --config")" ]; then echo "a SLAM run is active: not starting"; exit 1; fi
echo "== start $(date -Is)"
timeout 5400 python -u tools/pd_prev_loop.py --out results_exp/reports/pd $R/20261003123233_rigid_s1 $R/20261002013837_rigid_s0 2>&1 | grep --line-buffered -v "Warn\|^  warn"
echo "PD_PREV_DONE $(date -Is)"
