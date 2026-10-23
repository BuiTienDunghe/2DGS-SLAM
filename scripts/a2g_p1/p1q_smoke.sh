#!/usr/bin/env bash
# PLAN A2G-P1 (rút gọn) §8: Q10 regression (no --pw, must equal E1c) and smoke runs with invariant asserts.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results_exp/a2g_p1q
mkdir -p $OUT
bash scripts/e1/run_e1.sh $OUT/regress configs/tum/a2g_p1/fr3_snt_far.yaml -l --range 0 120 2 --method dec --seed 0 --save-dir $OUT/regress
echo "[p1q] regress rc=$?"
for s in snt_far st_far; do
  bash scripts/e1/run_e1.sh $OUT/smoke/$s configs/tum/a2g_p1/fr3_$s.yaml -l --range 0 120 2 --method dec --seed 0 --pw --pw-debug --save-dir $OUT/smoke/$s
  echo "[p1q] smoke $s rc=$?"
done
touch $OUT/smoke.done
