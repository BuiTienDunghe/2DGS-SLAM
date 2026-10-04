#!/usr/bin/env bash
# One SLAM run with logging + GPU monitor; refuses to start if another SLAM run is alive (R6).
# usage: scripts/exp/run_one.sh CONFIG SEED TAG [extra slam.py args...]
# Success is judged by metrics.csv in the run folder (the upstream teardown can exit non-zero).
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh

cfg=$1; seed=$2; tag=$3; shift 3
logdir=results_exp/logs
mkdir -p "$logdir"
scene=$(basename "$cfg" .yaml)
stamp=$(date +%Y%m%d%H%M%S)
log="$logdir/${stamp}_${scene}_${tag}.log"
gpulog="$logdir/${stamp}_${scene}_${tag}.gpu.csv"

if pgrep -f "python.*slam.py" >/dev/null || pgrep -f "multiprocessing.spawn" >/dev/null; then
  echo "[run_one] another SLAM process is alive; refusing to start (R6)" | tee -a "$log"
  exit 3
fi

bash tools/gpu_monitor.sh "$gpulog" &
mon=$!
t0=$(date +%s)
# backstop per scene (Replica ~2.2 h, TUM ~1 h measured); override with RUN_TIMEOUT
case "$cfg" in
  *replica*) tmo=${RUN_TIMEOUT:-3.5h} ;;
  *) tmo=${RUN_TIMEOUT:-2h} ;;
esac
echo "[run_one] start $(date -Is) cfg=$cfg seed=$seed tag=$tag timeout=$tmo args=$*" | tee -a "$log"
timeout --signal=TERM --kill-after=120 "$tmo" \
  python slam.py --config "$cfg" --seed "$seed" --tag "$tag" --save-dir results_exp "$@" >> "$log" 2>&1 &
spid=$!
# watchdog: a hung pipeline stops writing to the log (e.g. frontend waiting forever on a dead backend)
(
  last=$(stat -c %s "$log"); idle=0
  while kill -0 "$spid" 2>/dev/null; do
    sleep 60
    cur=$(stat -c %s "$log")
    if (( cur == last )); then idle=$((idle + 60)); else idle=0; last=$cur; fi
    if (( idle >= ${WATCHDOG_IDLE_S:-1200} )); then
      echo "[run_one] WATCHDOG: log idle ${idle}s -> killing run $(date -Is)" >> "$log"
      pkill -TERM -P "$spid" 2>/dev/null; kill -TERM "$spid" 2>/dev/null
      sleep 20; pkill -9 -f "python slam.py" 2>/dev/null; pkill -9 -f "multiprocessing.spawn" 2>/dev/null
      break
    fi
  done
) &
wpid=$!
wait "$spid"
rc=$?
kill "$wpid" 2>/dev/null
t1=$(date +%s)
kill "$mon" 2>/dev/null
sleep 2
# a dead main process can leave the spawned backend mapping forever at 100% GPU
pkill -9 -f "multiprocessing.spawn" 2>/dev/null

rundir=$(find results_exp -mindepth 3 -maxdepth 3 -type d -name "*_${tag}" -newermt "@$t0" | sort | tail -1)
status=FAILED
if [[ -n "$rundir" && -f "$rundir/metrics.csv" ]]; then status=COMPLETE; fi
if [[ -n "$rundir" ]]; then
  cp "$gpulog" "$rundir/gpu_log.csv"
  cp "$log" "$rundir/run.log"
fi
line="$(date -Is)\t${scene}\t${tag}\t${seed}\t${status}\trc=${rc}\t$((t1 - t0))s\t${rundir}"
echo -e "$line" >> "$logdir/runs.tsv"
echo -e "[run_one] $line" | tee -a "$log"
[[ "$status" == COMPLETE ]]
