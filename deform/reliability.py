"""Reliability score S_i (plan §4.3), computed from scratch at every loop event.

Each Gaussian is scored only by keyframes of its own scan (|t0_i - u_j| <= Delta_t), on the state
before the correction and at the pre-PGO poses, so drift between scans does not lower the score.
"""
import time

import torch
from einops import rearrange

from deform.render_utils import DEV, render_subset
from gaussian_splatting.utils.general_utils import depth2normal


@torch.no_grad()
def compute_reliability(G, cams, frame_fn, t0, delta_t, cfg, tr, kf_uids):
    """G: Gaussian dict (pre state); cams: {uid: Camera at pre-PGO pose}; frame_fn(uid) -> (rgb, depth scaled).

    Returns dict with S [N], H [N] bool, R [N] bool, n [N], rD [N], en [N] and timing.
    """
    rc = cfg["reliability"]
    t_start = time.perf_counter()
    N = G["xyz"].shape[0]
    t0 = t0.to(DEV).float().reshape(-1)
    n_vis = torch.zeros(N, device=DEV)
    num_D = torch.zeros(N, device=DEV)
    num_n = torch.zeros(N, device=DEV)
    den = torch.zeros(N, device=DEV)
    # Deviation from §4.3 (decided 2026-10-02 04:50): the wall-clock "use every second keyframe if > 30 s" switch
    # made S depend on timing (cold first event offline, warm backend online), breaking offline == online (G4).
    # Every keyframe is always used; exceeding the budget is only logged (A4's 120 s still bounds the event).
    uids = sorted(kf_uids)[:: max(1, int(rc.get("kf_step", 1)))]
    halved = False
    used = 0
    for i, uid in enumerate(uids):
        used += 1
        cam = cams[uid]
        E = (t0 - float(uid)).abs() <= delta_t
        if not bool(E.any()):
            continue
        _, depth = frame_fn(uid)
        p = render_subset(cam, G)
        D_hat = p["rend_depth_median"][0]
        N_hat = p["rend_normal"]
        N_hat = N_hat / N_hat.norm(dim=0, keepdim=True).clamp_min(1e-8)
        m = (depth > tr["depth_min_threshold"]) & (depth < tr["depth_max_threshold"]) & (p["rend_alpha"][0] > 0.95)
        ND = rearrange(depth2normal(depth[None], cam.fx, cam.fy, cam.cx, cam.cy), "h w c -> c h w")
        ND = ND / ND.norm(dim=0, keepdim=True).clamp_min(1e-8)
        mf = m.float()
        e_d = mf * (D_hat - depth).abs()
        e_n = mf * (1.0 - (N_hat * ND).sum(0).abs())
        C = p["contrib_full"]
        c_m = render_subset(cam, G, error_img=mf)["contrib_full"]
        c_d = render_subset(cam, G, error_img=e_d)["contrib_full"]
        c_n = render_subset(cam, G, error_img=e_n)["contrib_full"]
        Ef = E.float()
        n_vis += Ef * (C > rc["contrib_min"]).float()
        den += Ef * c_m
        num_D += Ef * c_d
        num_n += Ef * c_n
    ok = den > 1e-8
    rD = torch.where(ok, num_D / den.clamp_min(1e-8), torch.full_like(den, float("inf")))
    en = torch.where(ok, num_n / den.clamp_min(1e-8), torch.full_like(den, float("inf")))
    alpha = G["opacity"].reshape(-1)
    S = (1 - torch.exp(-n_vis / rc["n0"])) * alpha * torch.exp(-(rD / rc["sigma_D"]) ** 2) \
        * torch.exp(-(en / rc["sigma_n"]) ** 2)
    S = torch.where(ok, S, torch.zeros_like(S))
    smax = G["scale"].max(dim=1).values
    H = (alpha > rc["tau_alpha"]) & (smax >= rc["s_min"]) & (smax <= rc["s_max"]) & (n_vis >= rc["n_min"])
    R = H & (S >= rc["tau_S"])
    t_total = time.perf_counter() - t_start
    return {"S": S, "H": H, "R": R, "n": n_vis, "rD": rD, "en": en, "n_kfs_used": used, "halved": halved,
            "over_budget": t_total > rc.get("time_budget_s", 30.0), "time_s": t_total}
