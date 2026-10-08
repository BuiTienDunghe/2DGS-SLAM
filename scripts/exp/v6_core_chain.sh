#!/usr/bin/env bash
# plan v6 (core measurement), fr1/room, one run at a time:
#   R0 (L1 + logs) -> gate: ATE(all) in 7.0-9.0 cm, else STOP
#   -> R1 R2 (-o) -> R5 R6 (30 iterations, prev / const_vel) -> R3 R4 (loader D / R) -> R7 (15, const_vel)
#   -> R8 only if R5 and R6 are both > 9.0 cm (the better of the two + track_pad_ms)
# A run whose folder already has metrics.csv is skipped, so the chain can be restarted.
# usage: setsid nohup bash scripts/exp/v6_core_chain.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export SAVE_DIR=results_m
mkdir -p results_m/logs
log=results_m/logs/v6_chain.log
say() { echo "[v6_chain] $(date -Is) $*" | tee -a "$log"; }
rundir() { ls -d results_m/tum/room/*_"$1" 2>/dev/null | sort | tail -1; }
ate() {  # ATE over all tracked frames in cm, empty if the run has no metrics.csv
  local d; d=$(rundir "$1")
  [[ -n "$d" && -f "$d/metrics.csv" ]] && awk -F, 'NR==2 {printf "%.2f", $3 * 100}' "$d/metrics.csv"
}
run() {  # run CONFIG SEED TAG [args]; skipped if already complete
  local tag=$3
  if [[ -n "$(ate "$tag")" ]]; then say "skip $tag (done, ATE $(ate "$tag") cm)"; return 0; fi
  say "start $tag"
  bash scripts/exp/run_one.sh "$@" || say "run failed: $*"
  say "end $tag ATE(all) = $(ate "$tag") cm"
}
above9() { awk -v a="$1" 'BEGIN {exit !(a > 9.0)}'; }

C=configs/tum/fr1_room.yaml
V=configs/tum/v6
A="-l --dump-loops"

run $C 0 R0 $A
a0=$(ate R0)
if [[ -z "$a0" ]] || ! awk -v a="$a0" 'BEGIN {exit !(a >= 7.0 && a <= 9.0)}'; then
  say "GATE STOP: R0 ATE(all) = ${a0:-missing} cm is outside 7.0-9.0 cm"
  exit 2
fi
say "GATE PASS: R0 ATE(all) = $a0 cm"

run $C 0 R1 $A -o
run $C 1 R2 $A -o
run $V/fr1_room_it30_prev.yaml 0 R5 $A
run $V/fr1_room_it30_cv.yaml 0 R6 $A
run $V/fr1_room_loaderD.yaml 0 R3 $A
run $V/fr1_room_loaderR.yaml 0 R4 $A
run $V/fr1_room_it15_cv.yaml 0 R7 $A

a5=$(ate R5); a6=$(ate R6)
if [[ -n "$a5" && -n "$a6" ]] && above9 "$a5" && above9 "$a6"; then
  if awk -v x="$a5" -v y="$a6" 'BEGIN {exit !(x <= y)}'; then best=R5; base=$V/fr1_room_it30_prev.yaml; else best=R6; base=$V/fr1_room_it30_cv.yaml; fi
  source tools/exp_env.sh
  pad=$(python - "$(rundir R0)" "$(rundir $best)" <<'EOF'
import json, sys, statistics
def med(d):
    v = [json.loads(l)["t_track"] for l in open(d + "/timing.jsonl")]
    return statistics.median(v)
print(f"{max(0.0, (med(sys.argv[1]) - med(sys.argv[2])) * 1000):.0f}")
EOF
)
  say "R5 = $a5, R6 = $a6 (both > 9.0) -> R8 = $best + track_pad_ms $pad"
  printf '# plan v6 R8 (generated): %s plus a sleep after tracking\ninherit_from: "%s"\n\nTraining:\n  track_pad_ms: %s\n' "$best" "$base" "$pad" > $V/fr1_room_R8.yaml
  run $V/fr1_room_R8.yaml 0 R8 $A
else
  say "R8 not needed (R5 = ${a5:-missing}, R6 = ${a6:-missing})"
fi
say "chain finished"
