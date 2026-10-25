"""PLAN MONOGS-E1C §7.2: environment of the MonoGS runs -> docs/monogs/env.json
usage (repo root, env monogs): python tools/monogs/env_info.py
"""
import importlib.metadata as md
import json
import os
import subprocess


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def main():
    import numpy
    import open3d
    import torch
    env = dict(
        gpu=sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
        cuda_toolkit=sh("$HOME/cuda-12.8/bin/nvcc --version | tail -2 | head -1"),
        torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        open3d=open3d.__version__, numpy=numpy.__version__,
        evo=md.version("evo"), matplotlib=md.version("matplotlib"), wandb=md.version("wandb"),
        python=sh("python --version"), cpu=sh("lscpu | grep 'Model name' | sed 's/.*: *//'"),
        wsl_kernel=sh("uname -r"),
        tf32_default_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_default_cudnn=torch.backends.cudnn.allow_tf32,
        repo_commit=sh("git rev-parse --short HEAD"), repo_branch=sh("git rev-parse --abbrev-ref HEAD"),
        monogs_commit="6c9254c (muskie82/MonoGS, vendored unmodified in commit 749c5bd)",
        submodules={"diff-gaussian-rasterization": "43e21bff91cd24986ee3dd52fe0bb06952e50ec7",
                    "simple-knn": "44f764299fa305faf6ec5ebd99939e0508331503"},
        datasets={},
    )
    root = os.path.realpath("monogs/datasets/tum")
    for d in ("rgbd_dataset_freiburg1_desk", "rgbd_dataset_freiburg3_structure_notexture_far",
              "rgbd_dataset_freiburg3_structure_notexture_near"):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            n = sum(1 for l in open(os.path.join(p, "rgb.txt")) if not l.startswith("#"))
            env["datasets"][d] = dict(path=p, rgb_lines=n)
    os.makedirs("docs/monogs", exist_ok=True)
    json.dump(env, open("docs/monogs/env.json", "w"), indent=1)
    print(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
