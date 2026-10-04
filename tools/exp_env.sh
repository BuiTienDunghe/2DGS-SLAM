# source me: environment for every experiment command (handoff plan, R6 resource rules)
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate 2dgs-slam
export CUDA_HOME="$HOME/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export MPLCONFIGDIR="$HOME/2DGS-SLAM/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
# plan v4 A3: cuBLAS picks only deterministic GEMM algorithms (required by torch deterministic mode)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
