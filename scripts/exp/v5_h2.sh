#!/usr/bin/env bash
# plan v5 step B, H2 (report only): variant B vs rigid on the two baseline runs.
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
B0=results_exp/tum/room/20261002013837_rigid_s0; B1=results_exp/tum/room/20261003123233_rigid_s1
echo "== H2 $(date -Is)"
timeout 3600 python tools/h2_variant_b.py --out results_exp/reports/v5/h2 $B0 $B1 2>&1 | grep -v Warn | grep -E "^\{|Error|Traceback|line "
echo "V5_H2_DONE $(date -Is)"
