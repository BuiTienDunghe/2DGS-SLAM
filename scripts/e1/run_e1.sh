#!/usr/bin/env bash
# One E1 SLAM run (PLAN E1 §7): R6 guard, ptrace check, GPU log, idle watchdog, resume, runs.tsv.
# usage: bash scripts/e1/run_e1.sh OUTDIR CONFIG [slam.py args...]
#   OUTDIR is the exact run folder (results_exp/e1/<seq>/<method>/seed<k>). If slam.py was not given
#   --save-dir (pristine upstream code), the newest folder under results_exp/e1/tum/ is moved to OUTDIR.
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/exp_env.sh

out=$1; cfg=$2; shift 2
if [[ -f "$out/metrics.csv" ]]; then
  echo "[run_e1] $out already complete; skipping (resume)"; exit 0
fi
if [[ -e "$out" ]]; then
  echo "[run_e1] $out exists without metrics.csv; refusing to overwrite (move it away first)"; exit 5
fi
logdir=results_exp/logs
mkdir -p "$logdir" "$(dirname "$out")"
tag=$(echo "${out#results_exp/e1/}" | tr '/' '_')
stamp=$(date +%Y%m%d%H%M%S)
log="$logdir/${stamp}_e1_${tag}.log"
gpulog="$logdir/${stamp}_e1_${tag}.gpu.csv"

if ps -eo args | grep -E '^[^ ]*python[^ ]* .*slam[.]py|multiprocessing[.]spawn' | grep -vq grep; then
  echo "[run_e1] another SLAM process is alive; refusing to start (R6)" | tee -a "$log"; exit 3
fi
if [[ "$(cat /proc/sys/kernel/yama/ptrace_scope)" != "0" ]]; then
  echo "[run_e1] kernel.yama.ptrace_scope != 0; backend CUDA IPC would fail" | tee -a "$log"; exit 4
fi

bash tools/gpu_monitor.sh "$gpulog" &
mon=$!
t0=$(date +%s)
tmo=${RUN_TIMEOUT:-3h}
echo "[run_e1] start $(date -Is) out=$out cfg=$cfg timeout=$tmo commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) args=$*" | tee -a "$log"
timeout --signal=TERM --kill-after=120 "$tmo" python slam.py --config "$cfg" "$@" >> "$log" 2>&1 &
spid=$!
(
  last=$(stat -c %s "$log"); idle=0
  while kill -0 "$spid" 2>/dev/null; do
    sleep 60
    cur=$(stat -c %s "$log")
    if (( cur == last )); then idle=$((idle + 60)); else idle=0; last=$cur; fi
    if (( idle >= ${WATCHDOG_IDLE_S:-1200} )); then
      echo "[run_e1] WATCHDOG: log idle ${idle}s -> killing run $(date -Is)" >> "$log"
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
# a dead main process can leave the spawned backend mapping forever at 100% GPU
for p in $(ps -eo pid,args | grep -E 'multiprocessing[.]spawn' | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done

if [[ ! -d "$out" ]]; then
  src=$(find results_exp/e1/tum -mindepth 2 -maxdepth 2 -type d -newermt "@$t0" 2>/dev/null | sort | tail -1)
  [[ -n "$src" ]] && mv "$src" "$out"
fi
status=FAILED
if [[ -f "$out/metrics.csv" ]]; then status=COMPLETE; fi
if [[ -d "$out" ]]; then
  cp "$gpulog" "$out/gpu_log.csv"
  cp "$log" "$out/stdout.log"
  peak=$(awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3+0>m)m=$3+0} END{print m+0}' "$gpulog")
  echo "nvidia_smi_peak_mib $peak" >> "$out/vram.txt"
fi
line="$(date -Is)\te1_${tag}\t${tag}\t-\t${status}\trc=${rc}\t$((t1 - t0))s\t${out}"
echo -e "$line" >> "$logdir/runs.tsv"
echo -e "[run_e1] $line" | tee -a "$log"
[[ "$status" == COMPLETE ]]
