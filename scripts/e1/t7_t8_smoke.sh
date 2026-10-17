#!/usr/bin/env bash
# T7 (flag off) / T8 (lambda 0) / ours smoke on frames 0-120 step 2
cd "$(dirname "$0")/../.."
cfg=configs/tum/e1/fr3_snt_far.yaml
bash scripts/e1/run_e1.sh results_exp/e1/t7/modified_base $cfg -l --range 0 120 2 --method base --save-dir results_exp/e1/t7/modified_base
bash scripts/e1/run_e1.sh results_exp/e1/t7/t8_lambda0 configs/tum/e1/t8_lambda0.yaml -l --range 0 120 2 --method ours_loss --save-dir results_exp/e1/t7/t8_lambda0
bash scripts/e1/run_e1.sh results_exp/e1/t7/ours_smoke $cfg -l --range 0 120 2 --method ours --seed 0 --save-dir results_exp/e1/t7/ours_smoke
touch results_exp/e1/t7/integration.done
