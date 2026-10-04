#!/usr/bin/env bash
# handoffPlan_v5_persistence step B (offline, GPU): H0 noise floor, H1 PGO replay, H2 variant B. Sequential.
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
B0=results_exp/tum/room/20261002013837_rigid_s0; B1=results_exp/tum/room/20261003123233_rigid_s1
M0=results_exp/tum/room/20261003105219_deform5_s0; M1=results_exp/tum/room/20261003114156_deform5_s1
O=results_exp/reports/v5
echo "== H0 $(date -Is)"; timeout 1800 python tools/h0_noise_floor.py --out $O/h0 $B0 $B1 $M0 $M1 2>&1 | grep -v Warn | grep -v "^\s*$"
echo "== H1 $(date -Is)"; timeout 3600 python tools/h1_replay.py --out $O/h1 $B0 $B1 $M0 $M1 2>&1 | grep -v Warn | grep -E "^\{|^   \(|h1_replay|Error|Traceback|line "
echo "== H2 $(date -Is)"; timeout 3600 python tools/h2_variant_b.py --out $O/h2 $B0 $B1 2>&1 | grep -v Warn | grep -E "^\{|Error|Traceback|line "
echo "V5_STEP_B_DONE $(date -Is)"
