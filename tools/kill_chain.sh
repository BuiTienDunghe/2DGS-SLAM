#!/usr/bin/env bash
# Kill wrapper scripts (waiters / chains) by script name, never matching this shell, then stop_all.
me=$$
for pat in "p123_tum_then_run2" "gpu_window" "RUN2_DONE" "restart_chain" "p0_chain"; do
  for p in $(pgrep -f -- "$pat"); do
    [[ "$p" == "$me" ]] && continue
    kill "$p" 2>/dev/null
  done
done
sleep 1
bash "$(dirname "$0")/stop_all.sh"
