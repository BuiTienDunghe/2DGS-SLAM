#!/usr/bin/env bash
# PLAN MONOGS-E1C §2.6: one MonoGS run. Resume (skip when OUTDIR has traj_all.txt), single-SLAM guard, ptrace check,
# GPU log, timeout (RUN_TIMEOUT, default 3h), idle watchdog (WATCHDOG_IDLE_S, default 1800 s), runs.tsv.
# usage: bash scripts/monogs/run.sh OUTDIR CONFIG [slam.py args...]      (OUTDIR relative to the repo root)
set -uo pipefail
cd "$(dirname "$0")/../.."
root=$(pwd)
source tools/monogs_env.sh

out=$1; cfg=$2; shift 2
case "$out" in /*) ;; *) out="$root/$out" ;; esac
if [[ -f "$out/traj_all.txt" ]]; then
  echo "[monogs] $out already complete; skipping (resume)"; exit 0
fi
if [[ -e "$out" ]]; then
  echo "[monogs] $out exists without traj_all.txt; refusing to overwrite (move it away first)"; exit 5
fi
logdir="$root/results_monogs/logs"
mkdir -p "$logdir" "$(dirname "$out")"
tag=$(echo "${out#$root/results_monogs/}" | tr '/' '_')
stamp=$(date +%Y%m%d%H%M%S)
log="$logdir/${stamp}_${tag}.log"
gpulog="$logdir/${stamp}_${tag}.gpu.csv"

if ps -eo args | grep -E '^[^ ]*python[^ ]* .*slam[.]py|multiprocessing[.]spawn' | grep -vq grep; then
  echo "[monogs] another SLAM process is alive; refusing to start" | tee -a "$log"; exit 3
fi
if [[ "$(cat /proc/sys/kernel/yama/ptrace_scope)" != "0" ]]; then
  sudo -n sysctl -w kernel.yama.ptrace_scope=0 >/dev/null 2>&1
  if [[ "$(cat /proc/sys/kernel/yama/ptrace_scope)" != "0" ]]; then
    echo "[monogs] kernel.yama.ptrace_scope != 0 and no sudo: the user must run (Windows): wsl -d Ubuntu-22.04 -u root -- sysctl -w kernel.yama.ptrace_scope=0" | tee -a "$log"
    exit 4
  fi
fi

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv -l 5 > "$gpulog" &
mon=$!
t0=$(date +%s)
tmo=${RUN_TIMEOUT:-3h}
echo "[monogs] start $(date -Is) out=$out cfg=$cfg timeout=$tmo commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) args=$*" | tee -a "$log"
(cd monogs && exec timeout --signal=TERM --kill-after=120 "$tmo" python slam.py --config "$cfg" --save-dir "$out" "$@") >> "$log" 2>&1 &
spid=$!
(
  last=$(stat -c %s "$log"); idle=0
  while kill -0 "$spid" 2>/dev/null; do
    sleep 60
    cur=$(stat -c %s "$log")
    if (( cur == last )); then idle=$((idle + 60)); else idle=0; last=$cur; fi
    if (( idle >= ${WATCHDOG_IDLE_S:-1800} )); then
      echo "[monogs] WATCHDOG: log idle ${idle}s -> killing run $(date -Is)" >> "$log"
      pkill -TERM -P "$spid" 2>/dev/null; kill -TERM "$spid" 2>/dev/null
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
for p in $(ps -eo pid,args | grep -E 'multiprocessing[.]spawn' | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done

status=FAILED
[[ -f "$out/traj_all.txt" ]] && status=COMPLETE
if [[ -d "$out" ]]; then
  cp "$gpulog" "$out/gpu_log.csv"
  cp "$log" "$out/stdout.log"
fi
line="$(date -Is)\t$(git rev-parse --short HEAD)\t${cfg}\t$*\t${status}\trc=${rc}\t$((t1 - t0))s\t${out}"
echo -e "$line" >> "$root/results_monogs/runs.tsv"
echo -e "[monogs] $line" | tee -a "$log"
[[ "$status" == COMPLETE ]]
