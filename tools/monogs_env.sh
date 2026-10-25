# PLAN MONOGS-E1C §2.6: source before every MonoGS command
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate monogs
export CUDA_HOME="$HOME/cuda-12.8" PATH="$HOME/cuda-12.8/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
