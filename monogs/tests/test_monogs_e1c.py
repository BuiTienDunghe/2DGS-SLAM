"""PLAN MONOGS-E1C §5.4: T1 (pose forms), T2 (pose-gradient convention of the MonoGS rasterizer), T3 (ManualAdam).
run from monogs/:  python tests/test_monogs_e1c.py [--out ../docs/monogs/tests_monogs_e1c.json]
T2 needs the GPU and datasets/tum/rgbd_dataset_freiburg1_desk.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils.obs_decouple import ManualAdam  # noqa: E402
from utils.pose_utils import SE3_exp  # noqa: E402


class _V:  # minimal viewpoint with MonoGS's R, T, update_RT
    def __init__(self, R, T):
        self.R, self.T = R, T

    def update_RT(self, R, t):
        self.R, self.T = R, t


def t1():
    from utils.slam_frontend import FrontEnd
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(100):
        T = SE3_exp(torch.randn(6) * torch.tensor([0.5, 0.5, 0.5, 0.8, 0.8, 0.8]))
        v = _V(T[:3, :3].clone(), T[:3, 3].clone())
        R0, T0 = v.R.clone(), v.T.clone()
        FrontEnd.set_T_w2c(v, FrontEnd.T_w2c(v))
        worst = max(worst, float((v.R - R0).abs().max()), float((v.T - T0).abs().max()))
    return dict(passed=worst < 1e-7, max_abs_change=worst)


def t3():
    torch.manual_seed(1)
    lr = 0.003
    p = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    man = ManualAdam(lr, device="cpu", dtype=torch.float32)
    worst = 0.0
    for _ in range(200):
        g = torch.randn(3) * 10 ** torch.empty(1).uniform_(-4, 1)
        with torch.no_grad():
            p.zero_()
        p.grad = g.clone()
        opt.step()
        ref = p.detach().clone()
        out = man.step(g.clone())
        worst = max(worst, float((ref - out).abs().max() / max(float(ref.abs().max()), 1e-12)))
    return dict(passed=worst < 1e-6, max_rel_diff=worst)


def t2():
    from munch import munchify
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    from utils.camera_utils import Camera
    from utils.config_utils import load_config
    from utils.dataset import load_dataset
    from utils.slam_utils import get_loss_tracking

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = load_config("configs/rgbd/tum/fr1_desk_sp.yaml")
    cfg["Training"]["monocular"] = False
    mp_ = munchify(cfg["model_params"])
    mp_.sh_degree = 0
    ds = load_dataset(mp_, mp_.source_path, config=cfg)
    proj = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=ds.fx, fy=ds.fy, cx=ds.cx, cy=ds.cy, W=ds.width,
                                H=ds.height).transpose(0, 1).cuda()
    g = GaussianModel(0, config=cfg)
    g.init_lr(6.0)
    g.training_setup(munchify(cfg["opt_params"]))
    pipe = munchify(cfg["pipeline_params"])
    bg = torch.zeros(3, device="cuda")
    # map from frame 40 at its GT pose; tracked frame = frame 42, pose = GT of 42 perturbed (1 cm, 0.5 deg)
    cam0 = Camera.init_from_dataset(ds, 40, proj)
    cam0.update_RT(cam0.R_gt.float(), cam0.T_gt.float())
    depth0 = cam0.depth.copy()
    g.extend_from_pcd_seq(cam0, kf_id=0, init=True, scale=2.0, depthmap=depth0)
    v = Camera.init_from_dataset(ds, 42, proj)
    v.compute_grad_mask(cfg)
    Tgt = torch.eye(4, device="cuda")
    Tgt[:3, :3], Tgt[:3, 3] = v.R_gt.float(), v.T_gt.float()
    T0 = SE3_exp(torch.tensor([0.01, -0.006, 0.004, 0.006, -0.004, 0.003], device="cuda")) @ Tgt

    def loss_at(T, grad=False):
        v.update_RT(T[:3, :3].clone(), T[:3, 3].clone())
        v.cam_rot_delta.data.zero_()
        v.cam_trans_delta.data.zero_()
        v.cam_rot_delta.grad = None
        v.cam_trans_delta.grad = None
        pkg = render(v, g, pipe, bg)
        loss = get_loss_tracking(cfg, pkg["render"], pkg["depth"], pkg["opacity"], v)
        if grad:
            loss.backward()
            return float(loss), torch.cat([v.cam_trans_delta.grad, v.cam_rot_delta.grad]).double().cpu().numpy()
        return float(loss)

    res = {}
    for delta in (1e-4, 1e-3):
        l0, g_an = loss_at(T0, grad=True)
        g_fd = np.zeros(6)
        with torch.no_grad():
            for i in range(6):
                e = torch.zeros(6, device="cuda")
                e[i] = delta
                g_fd[i] = (loss_at(SE3_exp(e) @ T0) - loss_at(SE3_exp(-e) @ T0)) / (2 * delta)
        cos = float(g_an @ g_fd / (np.linalg.norm(g_an) * np.linalg.norm(g_fd) + 1e-30))
        big = np.abs(g_fd) > 0.2 * np.abs(g_fd).max()
        sign_ok = bool(np.all(np.sign(g_an[big]) == np.sign(g_fd[big])))
        res[f"delta_{delta:g}"] = dict(loss=l0, grad_rasterizer=g_an.tolist(), grad_fd=g_fd.tolist(), cos=cos,
                                       sign_large_ok=sign_ok)
    main = res["delta_0.0001"]
    return dict(passed=main["cos"] >= 0.95 and main["sign_large_ok"], **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()
    out = {}
    for name, fn in (("T1", t1), ("T3", t3)) + (() if a.skip_gpu else (("T2", t2),)):
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            out[name] = dict(passed=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        print(name, "PASSED" if out[name]["passed"] else "FAILED", {k: v for k, v in out[name].items() if k != "trace"})
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
