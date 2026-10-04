#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:/home/vha72/miniconda3/envs/2dgs-slam-e0-1/bin:$PATH"
export CUDA_HOME="/home/vha72/miniconda3/envs/2dgs-slam-e0-1"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TORCH_HOME="$PWD/.cache/torch"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$XDG_CACHE_HOME"
exec python "$@"
