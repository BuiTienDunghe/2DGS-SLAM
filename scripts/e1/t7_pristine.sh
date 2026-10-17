#!/usr/bin/env bash
# T7 reference: pristine b46da10 code (slam.py / frontend untouched), run twice -> nondeterminism floor.
cd "$(dirname "$0")/../.."
cfg=configs/tum/e1/fr3_snt_far.yaml
for r in A A2; do
  bash scripts/e1/run_e1.sh results_exp/e1/t7/pristine_$r "$cfg" -l --range 0 120 2
done
touch results_exp/e1/t7/pristine.done
