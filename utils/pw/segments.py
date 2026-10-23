"""PLAN A2G-P1 §3.1 (segment from per-cell sufficient statistics) and §3.5 (merge within a frame, F1)."""
from dataclasses import dataclass, field

import numpy as np
import torch

from utils.plane_anchor import MAX_SAMPLES, PlaneMeas, _even_subsample

from .model import CHI2_3, jac_a, jac_b, nis, residual, residual_cov


@dataclass
class Seg:                          # camera frame
    cells_ij: np.ndarray            # (N_s, 2) int
    cnt: np.ndarray                 # (N_s,)
    s1: np.ndarray                  # (N_s, 3)
    s2: np.ndarray                  # (N_s, 3, 3)
    n: np.ndarray
    p: np.ndarray
    B: np.ndarray
    R: np.ndarray
    N_s: int
    src: list
    meas: PlaneMeas = field(repr=False)
    mu: np.ndarray = field(default=None, repr=False)      # (N_s, 3) cell centres
    mse: float = 0.0

    @property
    def d(self):
        return -float(self.n @ self.p)


def fit_cells(ij, cnt, s1, s2, sensor, src, meas=None, cs=20, pts=None):
    """Exact refit from the sums of the cells (§3.1)."""
    N_px = float(cnt.sum())
    S1, S2 = s1.sum(0), s2.sum(0)
    p = S1 / N_px
    cov = S2 / N_px - np.outer(p, p)
    w, v = np.linalg.eigh(cov)
    n = v[:, 0].copy()
    if float(n @ p) > 0:
        n = -n
    B = np.stack([v[:, 2], v[:, 1]], 1)
    mu = s1 / cnt[:, None]
    lam = [max(float(np.var(mu @ B[:, k])), 1e-4) for k in range(2)]
    zbar = float(p[2])
    nv = float(n @ (p / p[2]))
    R = sensor.R(len(cnt), lam[0], lam[1], zbar, nv)
    if meas is None:
        mask = None
        if cs is not None and meas is None:
            pass
        meas = PlaneMeas(n=torch.from_numpy(n.copy()), d=-float(n @ p), mask_cells=None, n_px=int(N_px),
                         mse=max(float(w[0]), 0.0), pts_c=pts)
    return Seg(cells_ij=ij, cnt=cnt, s1=s1, s2=s2, n=n, p=p, B=B, R=R, N_s=int(len(cnt)), src=list(src),
               meas=meas, mu=mu, mse=max(float(w[0]), 0.0))


def build_segments(planes, sensor):
    """One Seg per PlaneMeas (which must carry .cells); meas is the untouched PlaneMeas (E1c bit-compatible)."""
    out = []
    for i, pm in enumerate(planes):
        c = pm.cells
        out.append(fit_cells(np.asarray(c["ij"]), np.asarray(c["cnt"], dtype=np.float64), np.asarray(c["s1"]),
                             np.asarray(c["s2"]), sensor, [i], meas=pm))
    return out


def seg_seg_nis(a, b):
    """NIS of r(a|b), K1."""
    r = residual(a.n, a.p, b.n, b.p, b.B)
    S = residual_cov("seg_seg", jac_a(a.n, a.B, b.n, b.B), a.R, jac_b(a.n, a.p, b.n, b.p, b.B), b.R)
    return nis(r, S)


def _member_nis(m, G):
    r = residual(m.n, m.p, G.n, G.p, G.B)
    Ja = jac_a(m.n, m.B, G.n, G.B)
    return nis(r, Ja @ m.R @ Ja.T)


def _union(segs, idx, planes, sensor):
    ij = np.concatenate([segs[i].cells_ij for i in idx])
    cnt = np.concatenate([segs[i].cnt for i in idx])
    s1 = np.concatenate([segs[i].s1 for i in idx])
    s2 = np.concatenate([segs[i].s2 for i in idx])
    src = sorted(s for i in idx for s in segs[i].src)
    if len(idx) == 1:
        return segs[idx[0]]
    G = fit_cells(ij, cnt, s1, s2, sensor, src)
    ms = [planes[s] for s in src]
    mask = ms[0].mask_cells.clone()
    for m in ms[1:]:
        mask |= m.mask_cells
    pts = _even_subsample(np.concatenate([m.pts_c for m in ms]), MAX_SAMPLES)
    G.meas = PlaneMeas(n=torch.from_numpy(G.n.copy()), d=G.d, mask_cells=mask, n_px=int(cnt.sum()),
                       mse=G.mse, pts_c=pts)
    G.meas.cells = dict(ij=ij, cnt=cnt, s1=s1, s2=s2)
    return G


def merge_in_frame(segs, planes, sensor, max_iter=3):
    """§3.5: two-way K1 test on every pair, union-find, refit, member re-check (failures split off)."""
    n = len(segs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if seg_seg_nis(segs[b], segs[a]) <= CHI2_3 and seg_seg_nis(segs[a], segs[b]) <= CHI2_3:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for root in sorted(groups):
        members = groups[root]
        singles = []
        for _ in range(max_iter):
            if len(members) <= 1:
                break
            G = _union(segs, members, planes, sensor)
            bad = [m for m in members if _member_nis(segs[m], G) > CHI2_3]
            if not bad:
                break
            members = [m for m in members if m not in bad]
            singles += bad
        else:
            pass
        if len(members) > 1:
            G = _union(segs, members, planes, sensor)
            bad = [m for m in members if _member_nis(segs[m], G) > CHI2_3]
            if bad:                                # not stable after max_iter refits: keep members apart
                singles += members
                members = []
        if members:
            out.append(_union(segs, members, planes, sensor))
        out += [segs[m] for m in sorted(singles)]
    out.sort(key=lambda s: s.src[0])
    return out
