#!/usr/bin/env bash
# plan v6 quick: E3, E1, E4 on the existing runs (GPU, sequential, no SLAM). E5-dependent parts run later.
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
R=results_exp/tum/room
B0=$R/20261002013837_rigid_s0; B1=$R/20261003123233_rigid_s1; M0=$R/20261003105219_deform5_s0; M1=$R/20261003114156_deform5_s1
V0=$R/20261004075609_deform6_s0; V1=$R/20261004085056_deform6_s1
O=results_exp/reports/v6q
mkdir -p $O
if [ -n "$(ps aux | grep "[s]lam.py --config")" ]; then echo "a SLAM run is active: not starting"; exit 1; fi
echo "== E3 $(date -Is)"
python -u tools/e3_floor_event.py --out $O/e3 --final $B1 $M0 $M1 $V0 $V1 2>&1 | grep --line-buffered -v "Warn\|^  warn"
echo "== E3 (3 compensation passes) $(date -Is)"
python -u tools/e3_floor_event.py --out $O/e3_it3 --comp-iters 3 --final $B1 $M0 $M1 $V0 $V1 2>&1 | grep --line-buffered -v "Warn\|^  warn"
echo "== E1 $(date -Is)"
python -u tools/e1_refine_split.py --out $O/e1 $B0 $B1 $M0 $M1 $V0 $V1 2>&1 | grep --line-buffered -v "Warn\|^  warn"
echo "== E4 $(date -Is)"
python -u tools/e4_merge.py --out $O/e4 $V0 $V1 2>&1 | grep --line-buffered -v "Warn\|^  warn"
echo "V6Q_OFFLINE_DONE $(date -Is)"
