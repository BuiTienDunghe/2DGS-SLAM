#!/usr/bin/env bash
# Stream important lines from every experiment log (new files picked up automatically).
cd ~/2DGS-SLAM/results_exp/logs || exit 1
declare -A pos
pat='\[p0\]|\[run_one\]|\[run_queue\]|Traceback|Error|error:|Killed|out of memory|Wrote metrics|Loop detected|deform event|map pgo|check_bookkeeping|G0A|Map refinement done|SLAM FPS'
while true; do
  for f in p0_chain.log *.log; do
    [[ -f "$f" ]] || continue
    sz=$(stat -c %s "$f")
    p=${pos[$f]:-0}
    if (( sz > p )); then
      tail -c +$((p + 1)) "$f" | tr '\r' '\n' | grep -E "$pat" | sed "s|^|${f%%.log}: |" | cut -c1-300
      pos[$f]=$sz
    fi
  done
  sleep 20
done
