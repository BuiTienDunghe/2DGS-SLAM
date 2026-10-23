#!/usr/bin/env bash
# PLAN A2G-P1 (rút gọn) §8–9: main runs (1 seed x 2 sequences, sequential), invariants, metrics, report.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results_exp/a2g_p1q
R=/home/wslphu/2DGS-SLAM/results_exp
P=python
source tools/exp_env.sh
for s in snt_far st_far; do
  bash scripts/e1/run_e1.sh $OUT/$s/p1/seed0 configs/tum/a2g_p1/fr3_$s.yaml -l --method dec --seed 0 --pw --save-dir $OUT/$s/p1/seed0
  echo "[p1q] main $s rc=$?"
done
for s in snt_far st_far; do
  $P tools/a2g_p1/check_invariants.py $OUT/$s/p1/seed0
done
$P tools/a2g_p1/compare.py --seqs snt_far,st_far \
  --base snt_far=$R/e1/fr3_snt_far/base/seed0,st_far=$R/e1f/fr3_st_far/base/seed0 \
  --e1c snt_far=$R/e1c/fr3_snt_far/dec/seed0,st_far=$R/e1f/fr3_st_far/dec/seed0 \
  --p1 snt_far=$OUT/snt_far/p1/seed0,st_far=$OUT/st_far/p1/seed0 \
  --out $OUT/REPORT_P1_QUICK.md --notes $OUT/report_notes.md
touch $OUT/main.done
