#!/usr/bin/env bash
# Stop every experiment process (queue, runs, spawned backends, GPU monitors). Safe to call from any shell.
for pat in "scripts/exp/p0_chain" "scripts/exp/run_queue" "scripts/exp/run_one" "python slam.py" "multiprocessing.spawn" "nvidia-smi --query-gpu"; do
  pgrep -f -- "$pat" | grep -v "^$$\$" | xargs -r kill 2>/dev/null
done
sleep 3
for pat in "python slam.py" "multiprocessing.spawn"; do
  pgrep -f -- "$pat" | xargs -r kill -9 2>/dev/null
done
sleep 1
ps -eo pid,args | grep -E "slam.py|multiprocessing.spawn|run_one|run_queue|p0_chain" | grep -v -E "grep|stop_all" || echo ALL_STOPPED
nvidia-smi --query-gpu=memory.used --format=csv,noheader
