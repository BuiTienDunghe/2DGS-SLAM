"""Which nodes influence a point (plan §4.5): spatial candidates, time-penalised cost, top-K, weights."""
import torch

from deform.render_utils import DEV


@torch.no_grad()
def influence(x, tau, g, t_nodes, r_node, delta_t, K=4, K_cand=32, beta=10.0, chunk=4096, return_dmax=False):
    """x [M,3], tau [M] (frame index), g [Kn,3], t_nodes [Kn]. Returns idx [M,K] long, w [M,K] float.

    gamma_k = ||x - g_k|| / r + beta * max(0, |tau - t_k| - Delta_t) / Delta_t ; keep the K smallest;
    d_max = Euclidean distance to the (K+1)-th by gamma; w ~ (1 - d/d_max)_+^2, normalised
    (uniform 1/K if all zero). Computed once on pre coordinates and kept fixed.
    """
    M, Kn = x.shape[0], g.shape[0]
    Ke = max(1, min(K, Kn - 1)) if Kn > 1 else 1
    Kc = min(K_cand, Kn)
    idx_out = torch.zeros((M, Ke), dtype=torch.long, device=DEV)
    w_out = torch.zeros((M, Ke), dtype=torch.float32, device=DEV)
    dmax_out = torch.zeros((M,), dtype=torch.float32, device=DEV)
    g = g.float()
    tn = t_nodes.float()
    for s in range(0, M, chunk):
        xs = x[s:s + chunk].float()
        ts = tau[s:s + chunk].float().reshape(-1)
        d = torch.cdist(xs, g)  # [m,Kn]
        dc, ic = torch.topk(d, Kc, dim=1, largest=False)
        dt = (ts[:, None] - tn[ic]).abs()
        gam = dc / r_node + beta * (dt - delta_t).clamp_min(0) / delta_t
        og = torch.argsort(gam, dim=1)
        dc_o = torch.gather(dc, 1, og)
        ic_o = torch.gather(ic, 1, og)
        sel_d = dc_o[:, :Ke]
        dmax = dc_o[:, Ke] if Kc > Ke else dc_o[:, -1] * 1.5 + 1e-6
        wt = (1 - sel_d / dmax[:, None].clamp_min(1e-9)).clamp_min(0) ** 2
        ssum = wt.sum(1, keepdim=True)
        wt = torch.where(ssum > 1e-12, wt / ssum.clamp_min(1e-12), torch.full_like(wt, 1.0 / Ke))
        idx_out[s:s + chunk] = ic_o[:, :Ke]
        w_out[s:s + chunk] = wt
        dmax_out[s:s + chunk] = dmax
    if return_dmax:
        return idx_out, w_out, dmax_out
    return idx_out, w_out


def weights_at(x, idx, g, dmax):
    """Re-evaluate the weights at (shifted) points x with the node set idx and d_max held fixed."""
    d = (x[:, None, :] - g[idx]).norm(dim=-1)
    wt = (1 - d / dmax[:, None].clamp_min(1e-9)).clamp_min(0) ** 2
    s = wt.sum(1, keepdim=True)
    return torch.where(s > 1e-12, wt / s.clamp_min(1e-12), torch.full_like(wt, 1.0 / idx.shape[1]))
