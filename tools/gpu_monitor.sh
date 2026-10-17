#!/usr/bin/env bash
# usage: tools/gpu_monitor.sh OUT.csv   -- samples GPU memory/utilization every 5 s until killed
exec nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$1"
