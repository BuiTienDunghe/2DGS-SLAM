#!/usr/bin/env bash
# PLAN E1 Phase 3 (reduced by the user): fr3_snt_far x {base, ours} x seed 0, sequential (R6)
cd "$(dirname "$0")/../.."
cfg=configs/tum/e1/fr3_snt_far.yaml
for m in base ours; do
  RUN_TIMEOUT=2h bash scripts/e1/run_e1.sh results_exp/e1/fr3_snt_far/$m/seed0 $cfg -l --method $m --seed 0 --save-dir results_exp/e1/fr3_snt_far/$m/seed0
done
touch results_exp/e1/phase3.done
