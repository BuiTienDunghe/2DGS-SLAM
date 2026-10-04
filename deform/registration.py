"""Dense registration of the new layer onto the old layer in one keyframe (plan v2 §3.3, D3).

Point-to-plane ICP with projective data association on the rendered layers (median depth, normals,
alpha) in the keyframe's camera frame, Gauss-Newton with Huber IRLS weights. The "pyramid" is a
source-pixel stride schedule (cfg levels, default 4 -> 2 -> 1) with full-resolution target lookups.

Quality checks Q1-Q4 (plan v2 table):
  Q1 inliers at the final iteration >= q1_frac * overlap and >= q1_min
  Q2 RMS point-to-plane residual of the inliers <= q2_rms
  Q3 pose uncertainty from Sigma = (J^T J)^-1 (residuals normalised by sigma_c): sqrt of the largest
     eigenvalue of the translation block <= q3_t and of the rotation block <= q3_r_deg
  Q4 |translation of H_c| <= q4_t and rotation angle <= q4_r_deg
H_c is expressed in the camera frame (translation measured at the camera centre); the world-frame
transform is H_w = T_c2w H_c T_c2w^-1.
"""
import math

import torch

from deform.metrics import ALPHA_THR, ray_dirs, stride_grid
from deform.render_utils import DEV


def _hat(w):
    O = torch.zeros((3, 3), dtype=w.dtype, device=w.device)
    O[0, 1], O[0, 2], O[1, 2] = -w[2], w[1], -w[0]
    O[1, 0], O[2, 0], O[2, 1] = w[2], -w[1], w[0]
    return O


def se3_exp(xi):
    """xi = (omega, v) -> 4x4 (float64)."""
    X = torch.zeros((4, 4), dtype=torch.float64, device=xi.device)
    X[:3, :3] = _hat(xi[:3])
    X[:3, 3] = xi[3:]
    return torch.linalg.matrix_exp(X)


def _associate(P_src, N_src, H, tgt, K, rc):
    """Transform source points [M,3] by H, project into the target images, gate. Returns dict."""
    R, t = H[:3, :3], H[:3, 3]
    p = P_src @ R.T + t
    n_s = N_src @ R.T
    Ht, Wt = tgt["P"].shape[1], tgt["P"].shape[2]
    z = p[:, 2]
    zs = z.clamp_min(1e-9)
    u = torch.round(K["fx"] * p[:, 0] / zs + K["cx"]).long()
    v = torch.round(K["fy"] * p[:, 1] / zs + K["cy"]).long()
    ok = (z > 1e-6) & (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)
    uc, vc = u.clamp(0, Wt - 1), v.clamp(0, Ht - 1)
    ok &= tgt["valid"][vc, uc]
    q = tgt["P"][:, vc, uc].T
    nq = tgt["N"][:, vc, uc].T
    d = p - q
    ok &= d.norm(dim=1) < rc["d_max"]
    ok &= (n_s * nq).sum(1) > math.cos(math.radians(rc["theta_deg"]))
    return {"p": p, "q": q, "nq": nq, "ok": ok, "r": (nq * d).sum(1), "pix": vc * Wt + uc}


def _normal_eq(a, sigma_c, huber):
    """Gauss-Newton system for the twist (omega, v) at the associations a (ok rows only)."""
    m = a["ok"]
    p, nq, r = a["p"][m], a["nq"][m], a["r"][m] / sigma_c
    J = torch.cat([torch.cross(p, nq, dim=1), nq], 1) / sigma_c
    ar = r.abs()
    w = torch.where(ar <= huber, torch.ones_like(ar), huber / ar.clamp_min(1e-12))
    A = J.T @ (w[:, None] * J)
    b = -(J.T @ (w * r))
    return A, b, J, r


def register(L, cam, rc, sigma_c, allowed=None):
    """L: render_layers() output for one keyframe; allowed: bool [H,W] restricting both source pixels
    and target lookups (D3-all on J_eval keyframes: cells A only). Returns dict with H_c (4x4 float64),
    the final inlier source pixels / associated target pixels, stats and Q1-Q4."""
    H_img, W_img = L["Do"].shape[-2], L["Do"].shape[-1]
    K = {"fx": float(cam.fx), "fy": float(cam.fy), "cx": float(cam.cx), "cy": float(cam.cy)}
    r = ray_dirs(cam).double()
    Pn = r * L["Dn"].double()[None]
    Po = r * L["Do"].double()[None]
    Nn = L["Nn"].double()
    No = L["No"].double()
    allow = torch.ones((H_img, W_img), dtype=torch.bool, device=DEV) if allowed is None else allowed
    src_ok = (L["On"] > ALPHA_THR) & allow
    tgt = {"P": Po, "N": No, "valid": (L["Oo"] > ALPHA_THR) & allow}
    n_overlap = int((src_ok & tgt["valid"]).sum())
    H = torch.eye(4, dtype=torch.float64, device=DEV)
    iters = []
    for s in rc["levels"]:
        msk = src_ok & stride_grid(H_img, W_img, int(s))
        P_src, N_src = Pn[:, msk].T, Nn[:, msk].T
        n_it = 0
        for _ in range(int(rc["max_iter"])):
            a = _associate(P_src, N_src, H, tgt, K, rc)
            if int(a["ok"].sum()) < 6:
                break
            A, b, _, _ = _normal_eq(a, sigma_c, rc["huber"])
            try:
                xi = torch.linalg.solve(A, b)
            except RuntimeError:
                break
            if not bool(torch.isfinite(xi).all()):
                break
            H = se3_exp(xi) @ H
            n_it += 1
            if float(xi.norm()) < rc["eps_conv"]:
                break
        iters.append(n_it)
    # final statistics at full source density
    msk = src_ok
    src_idx = torch.nonzero(msk.reshape(-1)).reshape(-1)
    P_src, N_src = Pn[:, msk].T, Nn[:, msk].T
    a0 = _associate(P_src, N_src, torch.eye(4, dtype=torch.float64, device=DEV), tgt, K, rc)
    a = _associate(P_src, N_src, H, tgt, K, rc)
    n_inl = int(a["ok"].sum())
    rms0 = float(a0["r"][a0["ok"]].pow(2).mean().sqrt()) if int(a0["ok"].sum()) else float("nan")
    rms = float(a["r"][a["ok"]].pow(2).mean().sqrt()) if n_inl else float("nan")
    sig_t = sig_r = float("inf")
    eig_ratio = 0.0
    if n_inl >= 6:
        _, _, J, _ = _normal_eq(a, sigma_c, rc["huber"])
        JtJ = J.T @ J
        lam, V = torch.linalg.eigh(JtJ)
        lam_max = float(lam[-1])
        eig_ratio = float(lam[0] / lam[-1]) if lam_max > 0 else 0.0
        inv = 1.0 / lam.clamp_min(1e-300)
        Sig = (V * inv[None]) @ V.T
        sig_r = math.degrees(math.sqrt(max(float(torch.linalg.eigvalsh(Sig[:3, :3])[-1]), 0.0)))
        sig_t = math.sqrt(max(float(torch.linalg.eigvalsh(Sig[3:, 3:])[-1]), 0.0))
    t_mag = float(H[:3, 3].norm())
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (float(torch.trace(H[:3, :3])) - 1) / 2))))
    q = {"Q1": n_inl >= rc["q1_frac"] * max(1, n_overlap) and n_inl >= rc["q1_min"],
         "Q2": math.isfinite(rms) and rms <= rc["q2_rms"],
         "Q3": sig_t <= rc["q3_t"] and sig_r <= rc["q3_r_deg"],
         "Q4": t_mag <= rc["q4_t"] and ang <= rc["q4_r_deg"]}
    fail = [k for k, v in q.items() if not v]
    return {"H_c": H, "passed": not fail, "fail": fail, "Q": q,
            "src_pix": src_idx[a["ok"]], "tgt_pix": a["pix"][a["ok"]],
            "stats": {"n_overlap": n_overlap, "n_inlier": n_inl, "rms_before_m": rms0, "rms_after_m": rms,
                      "sig_t_m": sig_t, "sig_r_deg": sig_r, "eig_ratio": eig_ratio, "t_mm": 1e3 * t_mag,
                      "rot_deg": ang, "iters": iters}}
