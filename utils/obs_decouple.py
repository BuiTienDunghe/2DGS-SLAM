"""PLAN E1c: observability basis of the plane term and a per-axis Adam that runs in that basis.

With the repo's left update ``new_w2c = SE3_exp(tau) @ T`` (tau = [trans, rot]):
  distance residual  d_meas - (d_w - (R n)^T t)  depends only on the translation part (first order),
  normal residual    m - R n                     depends only on the rotation part,
so the plane information splits into two 3x3 blocks built from the measured normals m_j (camera frame):
  H_t = sum_j m_j m_j^T          H_r = sum_j (I - m_j m_j^T)
Unweighted on purpose: sigma of H_t then equals the singular values of the stacked m_j (E1's sigma_min).
"""
import math

import numpy as np
import torch


def obs_basis(m, threshold):
    """m: (k, 3) measured normals (any array-like, may be empty). Returns dict of numpy arrays:
    Vt, Vr (3x3, columns = directions, ascending eigenvalue), sig_t, sig_r (3,), Mt, Mr (3,) in {0, 1}."""
    m = np.asarray(m, dtype=np.float64).reshape(-1, 3)
    if m.shape[0] == 0:
        z = np.zeros(3)
        return dict(Vt=np.eye(3), Vr=np.eye(3), sig_t=z.copy(), sig_r=z.copy(), Mt=z.copy(), Mr=z.copy())
    Ht = m.T @ m
    Hr = m.shape[0] * np.eye(3) - Ht
    lt, Vt = np.linalg.eigh(Ht)
    lr, Vr = np.linalg.eigh(Hr)
    sig_t = np.sqrt(np.clip(lt, 0, None))
    sig_r = np.sqrt(np.clip(lr, 0, None))
    return dict(Vt=Vt, Vr=Vr, sig_t=sig_t, sig_r=sig_r,
                Mt=(sig_t >= threshold).astype(np.float64), Mr=(sig_r >= threshold).astype(np.float64))


class ManualAdam:
    """Same arithmetic as torch.optim.AdamW (single tensor, foreach off) for a parameter that is reset to 0
    after every step (camera deltas): weight decay then has no effect, so it is omitted.
    step(g) returns the new parameter value, i.e. the update applied to a zero parameter."""

    def __init__(self, lr, betas=(0.9, 0.999), eps=1e-8, device="cuda", dtype=torch.float32):
        self.lr, self.b1, self.b2, self.eps = float(lr), betas[0], betas[1], eps
        self.m = torch.zeros(3, device=device, dtype=dtype)
        self.v = torch.zeros(3, device=device, dtype=dtype)
        self.t = 0

    @torch.no_grad()
    def step(self, g):
        # same _foreach_* calls, in the same order, as torch.optim.adam._multi_tensor_adam (the default on CUDA,
        # capturable=False); plain tensor ops differ from the foreach kernels by 1–2 ulp
        self.t += 1
        m, v, g = [self.m], [self.v], [g]
        torch._foreach_lerp_(m, g, 1 - self.b1)
        torch._foreach_mul_(v, self.b2)
        torch._foreach_addcmul_(v, g, g, 1 - self.b2)
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        step_size = [(self.lr / bc1) * -1]
        denom = torch._foreach_sqrt(v)
        torch._foreach_div_(denom, [bc2 ** 0.5])
        torch._foreach_add_(denom, self.eps)
        out = [torch.zeros_like(self.m)]
        torch._foreach_addcdiv_(out, m, denom, step_size)
        return out[0]


class DecoupledPoseStep:
    """One frame of E1c pose updates (PLAN E1c §3.3). Basis and masks fixed for the frame:
        xi = V^T g_render + M * (V^T g_plane)     (per block: translation, rotation)
        d_xi = ManualAdam(xi);  d_tau = V d_xi
    Directions with sigma < threshold get no plane gradient and keep the render step unscaled by it."""

    def __init__(self, m, threshold, trans_lr, rot_lr, device="cuda", dtype=torch.float32):
        b = obs_basis(m, threshold)
        self.basis = b
        t = lambda x: torch.as_tensor(x, device=device, dtype=dtype)  # noqa: E731
        self.Vt, self.Vr, self.Mt, self.Mr = t(b["Vt"]), t(b["Vr"]), t(b["Mt"]), t(b["Mr"])
        self.adam_t = ManualAdam(trans_lr, device=device, dtype=dtype)
        self.adam_r = ManualAdam(rot_lr, device=device, dtype=dtype)
        z = lambda: torch.zeros(3, device=device, dtype=dtype)  # noqa: E731
        self.sum_abs_dxi_t, self.sum_abs_dxi_r, self.sum_abs_gr_xi_t = z(), z(), z()
        self.gp_removed_norm = 0.0
        self.n_iter = 0
        self._zero = z()

    @torch.no_grad()
    def step(self, g_r_t, g_r_r, g_p_t=None, g_p_r=None):
        g_r_t = self._zero if g_r_t is None else g_r_t
        g_r_r = self._zero if g_r_r is None else g_r_r
        xr_t = self.Vt.T @ g_r_t
        xr_r = self.Vr.T @ g_r_r
        xi_t, xi_r = xr_t, xr_r
        if g_p_t is not None:
            xp_t = self.Vt.T @ g_p_t
            xp_r = self.Vr.T @ g_p_r
            if self.n_iter == 0:
                self.gp_removed_norm = float(torch.cat([(1 - self.Mt) * xp_t, (1 - self.Mr) * xp_r]).norm())
            xi_t = xr_t + self.Mt * xp_t
            xi_r = xr_r + self.Mr * xp_r
        d_xi_t = self.adam_t.step(xi_t)
        d_xi_r = self.adam_r.step(xi_r)
        self.sum_abs_dxi_t += d_xi_t.abs()
        self.sum_abs_dxi_r += d_xi_r.abs()
        self.sum_abs_gr_xi_t += xr_t.abs()
        self.n_iter += 1
        return self.Vt @ d_xi_t, self.Vr @ d_xi_r

    def log_row(self):
        """Columns of dec_log.csv (PLAN E1c §5)."""
        b = self.basis
        n = max(self.n_iter, 1)
        return dict(sig_t=b["sig_t"].tolist(), sig_r=b["sig_r"].tolist(), n_obs_t=int(b["Mt"].sum()),
                    n_obs_r=int(b["Mr"].sum()), Vt=b["Vt"].reshape(-1).tolist(), Vr=b["Vr"].reshape(-1).tolist(),
                    gp_removed_norm=self.gp_removed_norm, sum_abs_dxi_t=self.sum_abs_dxi_t.tolist(),
                    sum_abs_dxi_r=self.sum_abs_dxi_r.tolist(), mean_abs_gr_xi_t=(self.sum_abs_gr_xi_t / n).tolist())
