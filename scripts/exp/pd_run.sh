#!/usr/bin/env bash
# P vs D offline experiment on the loop dumps of four fr1/room runs (GPU, sequential, no SLAM).
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
R=results_exp/tum/room
B0=$R/20261002013837_rigid_s0; B1=$R/20261003123233_rigid_s1; M0=$R/20261003105219_deform5_s0; M1=$R/20261003114156_deform5_s1
O=results_exp/reports/pd
mkdir -p $O
if [ -n "$(ps aux | grep "[s]lam.py --config")" ]; then echo "a SLAM run is active: not starting"; exit 1; fi
echo "== start $(date -Is)"
# the optional D-active branch only on the first run (4 events)
timeout 7200 python tools/pd_experiment.py --d-active --out $O $B1 2>&1 | grep -v "Warn\|^  warn"
timeout 14400 python tools/pd_experiment.py --append --out $O $M0 $M1 $B0 2>&1 | grep -v "Warn\|^  warn"
echo "PD_DONE $(date -Is)"
