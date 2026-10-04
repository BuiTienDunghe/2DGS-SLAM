"""Energy E = w_con E_con + w_reg E_reg + w_p E_prior and LBFGS solve in float64 (plan §4.8)."""
import math
import time

import torch

from deform.field import node_rotations, phi


def huber(r):
    a = r.abs()
    return torch.where(a <= 1.0, r * r, 2.0 * a - 1.0)


def huber_norm(v):
    """Huber of the vector norm (point-to-point residual), smooth at 0."""
    sq = (v * v).sum(-1)
    return torch.where(sq <= 1.0, sq, 2.0 * sq.clamp_min(1e-30).sqrt() - 1.0)


def wmedian(x, w=None):
    """Weighted median; with uniform (or no) weights it equals torch.median (lower middle element)."""
    if w is None:
        return x.median()
    o = torch.argsort(x, stable=True)
    wo = w[o].double()
    if torch.are_deterministic_algorithms_enabled() and wo.is_cuda:  # plan v4 A3: see nodes._group_weighted_median
        cw = torch.cumsum(wo.cpu(), 0).to(wo.device)
    else:
        cw = torch.cumsum(wo, 0)
    k = int(torch.searchsorted(cw, 0.5 * cw[-1]).item())
    return x[o][min(k, x.numel() - 1)]


class Problem:
    def __init__(self, g, R0, t_init, edges, pairs, inf_old, inf_new, cfg):
        e = cfg["energy"]
        self.g = g.double()
        self.R0 = R0.double()
        self.t_init = t_init.double()
        self.E = edges
        self.w_con, self.w_reg, self.w_p = e["w_con"], e["w_reg"], e["w_p"]
        self.s_c, self.s_r, self.s_t = e["sigma_c"], e["sigma_r"], e["sigma_t"]
        self.s_w = math.radians(e["sigma_w_deg"])
        self.p2p = cfg["corr"].get("residual", "p2l") == "p2p"
        self.has_pairs = pairs is not None and pairs.get("P", 0) > 0
        self.omega = None
        if self.has_pairs:
            self.xo, self.xn = pairs["x_old"].double(), pairs["x_new"].double()
            self.n = pairs["n"].double()
            self.io, self.wo = inf_old
            self.in_, self.wn = inf_new
            if pairs.get("omega") is not None:
                self.omega = pairs["omega"].double()

    def unpack(self, theta):
        om, t = theta[:, :3], theta[:, 3:]
        return node_rotations(om, self.R0), t, om

    def residual_vec(self, R, t):
        po = phi(self.xo, self.io, self.wo, self.g, R, t)
        pn = phi(self.xn, self.in_, self.wn, self.g, R, t)
        return pn - po

    def residual_con(self, R, t):
        """Signed point-to-plane residual (p2l) or point-to-point distance (p2p), metres."""
        d = self.residual_vec(R, t)
        if self.p2p:
            return d.norm(dim=-1)
        return (self.n * d).sum(-1)

    def terms(self, theta):
        R, t, om = self.unpack(theta)
        out = {}
        if self.has_pairs and self.w_con > 0:
            if self.p2p:
                h = huber_norm(self.residual_vec(R, t) / self.s_c)
            else:
                h = huber(self.residual_con(R, t) / self.s_c)
            out["con"] = self.w_con * (h.sum() if self.omega is None else (self.omega * h).sum())
        else:
            out["con"] = theta.new_zeros(())
        if self.E.shape[0] and self.w_reg > 0:
            k, l = self.E[:, 0], self.E[:, 1]
            r_kl = (R[k] @ (self.g[l] - self.g[k])[..., None]).squeeze(-1) + self.g[k] + t[k] - self.g[l] - t[l]
            r_lk = (R[l] @ (self.g[k] - self.g[l])[..., None]).squeeze(-1) + self.g[l] + t[l] - self.g[k] - t[k]
            out["reg"] = self.w_reg * ((r_kl ** 2).sum() + (r_lk ** 2).sum()) / self.s_r ** 2
        else:
            out["reg"] = theta.new_zeros(())
        # prior: g_k + t_k - DeltaT_kappa g_k = t_k - t_k^(0)  (t^(0) = DeltaT g - g, zero for V-A)
        out["prior"] = self.w_p * (((t - self.t_init) ** 2).sum() / self.s_t ** 2 + (om ** 2).sum() / self.s_w ** 2)
        return out


def _grad_norm(prob, theta, scale):
    """||dE/dtheta|| of the scaled energy at theta (theta is a leaf with requires_grad)."""
    if theta.grad is not None:
        theta.grad = None
    E = sum(prob.terms(theta).values()) * scale
    (g,) = torch.autograd.grad(E, theta)
    return float(g.norm()), float(E)


def _solve_grad(prob, theta, e_init, sc):
    """tol_mode "grad" (plan v3 step A): LBFGS on the energy normalised by its initial value E/E_init, stopped
    when ||grad|| <= grad_rtol * ||grad_0||. The normalisation makes the whole iteration (first step size,
    curvature pairs, stopping test) independent of the magnitude of E; the test on the gradient (not on the
    energy change or the step length) means a stopped solution is a stationary point. LBFGS memory is reset
    if a step leaves theta unchanged (failed line search)."""
    scale = 1.0 / e_init
    rtol = float(sc.get("grad_rtol", 1e-6))
    max_it = int(sc.get("max_iter_conv", 3000))
    hist = int(sc.get("history_conv", sc["history"]))

    def make_opt():
        return torch.optim.LBFGS([theta], lr=1.0, max_iter=1, history_size=hist, line_search_fn="strong_wolfe",
                                 tolerance_grad=0.0, tolerance_change=0.0)

    opt = make_opt()

    def closure():
        opt.zero_grad()
        E = sum(prob.terms(theta).values()) * scale
        E.backward()
        return E

    g0, _ = _grad_norm(prob, theta, scale)
    info = {"grad0": g0, "grad_final": g0, "grad_ratio": 1.0, "converged": g0 == 0.0, "hit_max": False,
            "resets": 0, "iters": 0}
    if g0 == 0.0:
        return info
    it = 0
    while it < max_it:
        prev = theta.detach().clone()
        opt.step(closure)
        it += 1
        gn, _ = _grad_norm(prob, theta, scale)
        info["grad_final"] = gn
        if not math.isfinite(gn):
            break
        if gn <= rtol * g0:
            info["converged"] = True
            break
        if torch.equal(prev, theta.detach()):  # line search made no progress: drop the curvature memory
            info["resets"] += 1
            if info["resets"] > 5:
                break
            opt = make_opt()
    info["iters"] = it
    info["hit_max"] = (not info["converged"]) and it >= max_it
    info["grad_ratio"] = info["grad_final"] / g0
    return info


def solve(prob, theta0, cfg):
    sc = cfg["solver"]
    t_start = time.perf_counter()
    theta = theta0.clone().double().requires_grad_(True)
    with torch.no_grad():
        E0 = {k: float(v) for k, v in prob.terms(theta).items()}
    e_init = sum(E0.values())
    # tol_mode "legacy" (frozen A-1): torch applies tolerance_change = rel_tol * E_init to the energy change
    # AND to the largest parameter step, so large E_init (many pairs) stops LBFGS after a few iterations.
    # tol_mode "energy" (plan v2 diagnostics): stop only on the relative energy change between iterations.
    # tol_mode "grad" (plan v3): normalised energy, stop on the relative gradient norm (see _solve_grad).
    mode = sc.get("tol_mode", "legacy")
    if mode == "grad":
        ginfo = _solve_grad(prob, theta, e_init, sc) if e_init > 0 else {
            "grad0": 0.0, "grad_final": 0.0, "grad_ratio": 0.0, "converged": True, "hit_max": False, "resets": 0, "iters": 0}
        theta = theta.detach()
        with torch.no_grad():
            E1 = {k: float(v) for k, v in prob.terms(theta).items()}
        return theta, {"E_init": E0, "E_final": E1, "lbfgs_iters": ginfo["iters"], "solve_s": time.perf_counter() - t_start,
                       "finite": bool(torch.isfinite(theta).all()) and all(math.isfinite(v) for v in E1.values()),
                       "grad": ginfo}
    if mode == "energy":
        opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=1, history_size=int(sc["history"]),
                                line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=0.0)
    else:
        opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=int(sc["max_iter"]), history_size=int(sc["history"]),
                                line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                                tolerance_change=float(sc["rel_tol"]) * max(e_init, 1e-12))

    def closure():
        opt.zero_grad()
        E = sum(prob.terms(theta).values())
        E.backward()
        return E

    if e_init > 0 and mode == "energy":
        prev = e_init
        for _ in range(int(sc["max_iter"])):
            opt.step(closure)
            with torch.no_grad():
                cur = float(sum(prob.terms(theta).values()))
            if not math.isfinite(cur) or abs(prev - cur) <= float(sc["rel_tol"]) * max(abs(prev), 1e-12):
                break
            prev = cur
    elif e_init > 0:
        opt.step(closure)
    st = opt.state[opt._params[0]]
    n_iter = int(st.get("n_iter", 0))
    theta = theta.detach()
    with torch.no_grad():
        E1 = {k: float(v) for k, v in prob.terms(theta).items()}
    return theta, {"E_init": E0, "E_final": E1, "lbfgs_iters": n_iter, "solve_s": time.perf_counter() - t_start,
                   "finite": bool(torch.isfinite(theta).all()) and all(math.isfinite(v) for v in E1.values())}
