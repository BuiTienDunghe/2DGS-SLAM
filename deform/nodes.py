"""Deformation-graph nodes and edges (plan §4.4).

Nodes are keyed by (voxel of r_node, time cluster of t0 inside the voxel). A node is placed at the
S-weighted mean of the reliable Gaussians of its key (filler node if none is reliable).
Edges: temporal chain (k, k+1), (k, k+2) over nodes sorted by (t_k, morton) + up to 4 spatial
neighbours of the same scan (|t_k - t_l| <= Delta_t, ||g_k - g_l|| <= 2 r_node).
"""
import time

import numpy as np
import torch

from deform.render_utils import DEV


def morton3(v):
    """Interleave 10 bits of each non-negative int coordinate. v: [M,3] int64 tensor."""
    v = v.clamp(0, 1023).long()
    out = torch.zeros(v.shape[0], dtype=torch.long, device=v.device)
    for b in range(10):
        for a in range(3):
            out |= ((v[:, a] >> b) & 1) << (3 * b + a)
    return out


def _group_weighted_median(group, val, w, n_groups):
    """Weighted median of val per group."""
    o1 = torch.argsort(val, stable=True)
    o2 = torch.argsort(group[o1], stable=True)
    order = o1[o2]  # grouped, val ascending inside each group
    g, v, ww = group[order], val[order], w[order]
    tot = torch.zeros(n_groups, device=val.device, dtype=ww.dtype).scatter_add_(0, g, ww)
    if torch.are_deterministic_algorithms_enabled() and ww.is_cuda:
        # plan v4 A3: float cumsum has no deterministic CUDA kernel -> sequential float64 scan on the CPU
        cs = torch.cumsum(ww.detach().cpu().double(), 0).to(dtype=ww.dtype, device=ww.device)
    else:
        cs = torch.cumsum(ww, 0)
    start = torch.zeros(n_groups, device=val.device, dtype=ww.dtype)
    first = torch.ones_like(g, dtype=torch.bool)
    first[1:] = g[1:] != g[:-1]
    start[g[first]] = (cs - ww)[first]
    local = cs - start[g]
    reached = local >= 0.5 * tot[g]
    # first index per group where the half weight is reached
    big = torch.full((n_groups,), len(g), device=val.device, dtype=torch.long)
    idx = torch.arange(len(g), device=val.device)
    big = big.scatter_reduce(0, g[reached], idx[reached], reduce="amin")
    return v[big.clamp(max=len(g) - 1)]


def _group_weighted_mode(group, val, w, n_groups):
    key = group * (int(val.max()) + 2) + (val + 1)
    uk, inv = torch.unique(key, return_inverse=True)
    sw = torch.zeros(len(uk), device=val.device, dtype=w.dtype).scatter_add_(0, inv, w)
    ug = uk // (int(val.max()) + 2)
    uval = uk % (int(val.max()) + 2) - 1
    best = torch.full((n_groups,), -1.0, device=val.device, dtype=w.dtype).scatter_reduce(0, ug, sw, reduce="amax")
    is_best = sw >= best[ug]
    out = torch.full((n_groups,), -1, device=val.device, dtype=torch.long)
    # deterministic tie-break: largest value among the maxima
    out = out.scatter_reduce(0, ug[is_best], uval[is_best], reduce="amax")
    return out


@torch.no_grad()
def build_nodes(x, t0, tc, S, R, alpha, r_node, delta_t, min_alpha=0.1, use_time=True, use_S=True):
    """x [N,3] pre coordinates (GPU). Returns dict with node tensors and Gaussian->node membership."""
    t_start = time.perf_counter()
    cand = torch.nonzero(alpha.reshape(-1) > min_alpha).reshape(-1)
    xc = x[cand]
    v = torch.floor(xc / r_node).long()
    vmin = v.min(0).values
    v = v - vmin
    span = v.max(0).values + 1
    vkey = (v[:, 0] * span[1] + v[:, 1]) * span[2] + v[:, 2]
    tt = t0.reshape(-1)[cand].long()
    order = torch.argsort(vkey * (int(tt.max()) + 1) + tt)
    vk_s, t_s = vkey[order], tt[order]
    new = torch.ones_like(vk_s, dtype=torch.bool)
    new[1:] = vk_s[1:] != vk_s[:-1]
    if use_time:
        new[1:] |= (t_s[1:] - t_s[:-1]) > delta_t
    cl = torch.cumsum(new.long(), 0) - 1
    node_of_cand = torch.empty_like(cl)
    node_of_cand[order] = cl
    K = int(cl.max()) + 1 if cl.numel() else 0

    Rc = R.reshape(-1)[cand]
    Sc = S.reshape(-1)[cand].float() if use_S else torch.ones_like(S.reshape(-1)[cand].float())
    n_rel = torch.zeros(K, device=DEV).scatter_add_(0, node_of_cand, Rc.float())
    reliable = n_rel > 0
    use_rel = reliable[node_of_cand] & Rc  # member counts for its node's position
    w = torch.where(use_rel, Sc.clamp_min(1e-6), torch.zeros_like(Sc))
    w = torch.where(reliable[node_of_cand], w, torch.ones_like(w))  # filler: equal weights, all members
    wsum = torch.zeros(K, device=DEV).scatter_add_(0, node_of_cand, w)
    g = torch.zeros((K, 3), device=DEV).index_add_(0, node_of_cand, xc * w[:, None]) / wsum[:, None].clamp_min(1e-12)
    sel = w > 0
    t_k = _group_weighted_median(node_of_cand[sel], tt[sel].float(), w[sel], K)
    kappa = _group_weighted_mode(node_of_cand[sel], tc.reshape(-1)[cand][sel].long(), w[sel], K)
    nodes = {"g": g, "t": t_k, "kappa": kappa, "filler": ~reliable,
             "n_members": torch.zeros(K, device=DEV).scatter_add_(0, node_of_cand, torch.ones_like(w)),
             "cand": cand, "node_of_cand": node_of_cand}
    # voxel id of each node (for the scan-separation metric)
    vox_node = torch.zeros(K, dtype=torch.long, device=DEV).scatter_reduce(0, node_of_cand, vkey, reduce="amax", include_self=False)
    nodes["voxel_key"] = vox_node
    nodes["time_s"] = time.perf_counter() - t_start
    nodes["vmin"] = vmin
    return nodes


def build_edges(nodes, r_node, delta_t, k_spatial=4):
    """Undirected edge list [E,2] (k<l) = temporal chain U same-scan spatial kNN."""
    from scipy.spatial import cKDTree

    g = nodes["g"]
    K = g.shape[0]
    if K < 2:
        return torch.zeros((0, 2), dtype=torch.long, device=DEV)
    vox = torch.floor(g / r_node).long()
    mort = morton3(vox - vox.min(0).values)
    t = nodes["t"].double()
    order = torch.from_numpy(np.lexsort((mort.cpu().numpy(), t.cpu().numpy()))).to(DEV)
    e = [torch.stack([order[:-1], order[1:]], 1)]
    if K > 2:
        e.append(torch.stack([order[:-2], order[2:]], 1))
    gn = g.cpu().numpy().astype(np.float64)
    tn = t.cpu().numpy()
    tree = cKDTree(gn)
    kq = min(K, 33)
    d, j = tree.query(gn, k=kq, distance_upper_bound=2 * r_node)
    rows = []
    for a in range(K):
        cnt = 0
        for dd, b in zip(d[a], j[a]):
            if not np.isfinite(dd) or b == a or b >= K:
                continue
            if abs(tn[a] - tn[b]) <= delta_t:
                rows.append((a, b))
                cnt += 1
                if cnt >= k_spatial:
                    break
    if rows:
        e.append(torch.as_tensor(rows, dtype=torch.long, device=DEV))
    E = torch.cat(e, 0)
    E = torch.stack([E.min(1).values, E.max(1).values], 1)
    E = E[E[:, 0] != E[:, 1]]
    return torch.unique(E, dim=0)


def systematic_nodes(x, alpha, n_target, seed, min_alpha=0.1):
    """Ablation (i): ElasticFusion-style uniform sampling of Gaussians as node positions."""
    cand = torch.nonzero(alpha.reshape(-1) > min_alpha).reshape(-1)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    pick = cand[torch.randperm(len(cand), generator=gen)[: int(n_target)].to(cand.device)]
    return pick
