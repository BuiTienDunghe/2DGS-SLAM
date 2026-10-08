#!/usr/bin/env bash
# plan v6: all offline analyses of the core report. CPU parts can run any time; the GPU parts (A5) only when no SLAM
# run is alive (R6) - pass "gpu" as the first argument to include them.
# usage: bash scripts/exp/v6_analysis.sh [gpu]
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
A=results_m/analysis
OLD=results_exp/tum/room
REF="$OLD/20261002013837_rigid_s0 $OLD/20261003123233_rigid_s1 $OLD/20261003105219_deform5_s0 $OLD/20261003114156_deform5_s1 $OLD/20261004075609_deform6_s0 $OLD/20261004085056_deform6_s1"
new() {  # run folders of the given tags that finished (have metrics.csv), newest per tag
  for t in "$@"; do
    d=$(ls -d results_m/tum/room/*_"$t" 2>/dev/null | sort | tail -1)
    [[ -n "$d" && -f "$d/metrics.csv" ]] && echo "$d"
  done
}
ALL=$(new R0 R1 R2 R5 R6 R7 R8 R3 R4)
echo "== finished runs: $ALL"

echo "== A2 ATE split"
python tools/v6_ate_split.py --out $A/A2 $REF $ALL
echo "== Q1/Q4 timing + tracking statistics"
python tools/v6_track_stats.py --out $A/Q14 $ALL
echo "== A3 pose-graph oracle"
python tools/v6_pgo_oracle.py --out $A/A3_rigid_s1 $OLD/20261003123233_rigid_s1
R0=$(new R0)
[[ -n "$R0" ]] && python tools/v6_pgo_oracle.py --out $A/A3_R0 $R0
echo "== A4 loader chamfer"
[[ -f $A/A4/loader_chamfer.json ]] || python tools/v6_loader_chamfer.py --out $A/A4

if [[ "${1:-}" == "gpu" ]]; then
  G=$(new R0 R3 R4)
  echo "== A5(i) radial residual: $G"
  [[ -f $A/A5_ref/residual_radial.json ]] || python tools/v6_residual_radial.py --out $A/A5_ref $REF 2>&1 | grep -v Warning
  python tools/v6_residual_radial.py --out $A/A5_new $G 2>&1 | grep -v Warning
  echo "== A5(ii) same-pass noise floor (keyframe-wise and random split)"
  for s in rank random; do
    python tools/h0_noise_floor.py --out $A/A5_h0 --split $s $OLD/20261002013837_rigid_s0 $OLD/20261003123233_rigid_s1 $G 2>&1 | grep -v Warning
  done
fi
python tools/v6_report_figs.py --analysis $A --out results_m/reports/figs
