"""PLAN MONOGS-PW §2.6 (1b): secant stiffness of the render loss and of the plane loss along the 6 camera axes.

Axes e_k in the tau order of SE3_exp ([tx, ty, tz, rx, ry, rz], left perturbation of T_w2c). Steps delta_t (m),
delta_r (rad). Delta_k = (L(+) + L(-)) / 2 - L(0); c_k = 2 Delta_k / delta_k^2 (L1 is piecewise linear: secant
stiffness at the scale of the plane noise). Measurement only: never used inside the tracking loop.
"""
import time

import numpy as np
import torch

from gaussian_splatting.gaussian_renderer import render
from utils.pose_utils import SE3_exp
from utils.slam_utils import get_loss_tracking


def _deltas(delta_t_m, delta_r_deg):
    return [delta_t_m] * 3 + [float(np.deg2rad(delta_r_deg))] * 3


def _T(viewpoint):
    T = torch.eye(4, device=viewpoint.R.device, dtype=torch.float32)
    T[:3, :3] = viewpoint.R
    T[:3, 3] = viewpoint.T
    return T


@torch.no_grad()
def render_stiffness(viewpoint, gaussians, pipe, bg, config, delta_t_m=0.01, delta_r_deg=0.5):
    """c_render (6,), L0. R and T of the camera are restored to the exact cloned tensors."""
    t0 = time.perf_counter()
    R0, T0 = viewpoint.R.clone(), viewpoint.T.clone()
    Tw = _T(viewpoint)

    def loss_at(Tm):
        viewpoint.update_RT(Tm[:3, :3].clone(), Tm[:3, 3].clone())
        pkg = render(viewpoint, gaussians, pipe, bg)
        return float(get_loss_tracking(config, pkg["render"], pkg["depth"], pkg["opacity"], viewpoint))

    try:
        L0 = loss_at(Tw)
        c = np.zeros(6)
        for k, dk in enumerate(_deltas(delta_t_m, delta_r_deg)):
            e = torch.zeros(6, device=Tw.device)
            e[k] = dk
            lp = loss_at(SE3_exp(e) @ Tw)
            lm = loss_at(SE3_exp(-e) @ Tw)
            c[k] = 2.0 * ((lp + lm) / 2.0 - L0) / dk ** 2
    finally:
        viewpoint.update_RT(R0, T0)
    return c, L0, 1000 * (time.perf_counter() - t0)


@torch.no_grad()
def plane_stiffness(loss_fn, T_w2c, delta_t_m=0.01, delta_r_deg=0.5):
    """c_plane (6,) of a pose loss P(T) (kappa = 1), same axes and steps as render_stiffness."""
    c = np.zeros(6)
    P0 = float(loss_fn(T_w2c))
    for k, dk in enumerate(_deltas(delta_t_m, delta_r_deg)):
        e = torch.zeros(6, device=T_w2c.device, dtype=T_w2c.dtype)
        e[k] = dk
        c[k] = 2.0 * ((float(loss_fn(SE3_exp(e) @ T_w2c)) + float(loss_fn(SE3_exp(-e) @ T_w2c))) / 2.0 - P0) / dk ** 2
    return c
