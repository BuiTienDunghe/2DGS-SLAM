"""Plane-anchor tracking for 2DGS-SLAM (PLAN E1).

Conventions (PLAN E1 §4.1):
  * T_cw is world->camera (w2c): x_c = R x_w + t  (same as ``Camera.T``).
  * Plane: n.x + d = 0, |n| = 1. Measured planes are oriented with d > 0 (normal faces the camera).
  * world->camera: n_c = R n_w, d_c = d_w - n_c.t ;  camera->world: n_w = R^T n_c, d_w = d_c + n_c.t

Plane extraction is "CAPE-lite (re-implementation)": cell-wise PCA + region growing on the cell grid,
after Proenca & Gao (IROS 2018) without cylinders or boundary refinement. It is not CAPE itself.

Everything except ``plane_loss`` runs in float64 numpy on the CPU (a frame has at most a few hundred
cells and a handful of planes); ``plane_loss`` is torch and differentiable w.r.t. T_cw.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor

MAX_SAMPLES = 500


def sigma_z(z):
    """Kinect depth noise model (Proenca & Gao, RAS 2018), metres."""
    return 1.425e-3 * np.square(z)


def _to_np(x):
    if isinstance(x, Tensor):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


def _even_subsample(pts, n_max):
    """Deterministic subsample: evenly spaced indices, keeps spatial coverage of an ordered set."""
    if pts.shape[0] <= n_max:
        return pts
    idx = np.linspace(0, pts.shape[0] - 1, n_max).round().astype(np.int64)
    return pts[idx]


def _pca_from_sums(n, s1, s2):
    """Plane fit from point count, sum of points and sum of outer products. Returns n, d, mse, centroid."""
    mu = s1 / n
    cov = s2 / n - np.outer(mu, mu)
    w, v = np.linalg.eigh(cov)
    normal = v[:, 0]
    d = -float(normal @ mu)
    if d < 0:
        normal, d = -normal, -d
    return normal, d, max(float(w[0]), 0.0), mu


def plane_w2c(n_w, d_w, T_cw):
    T = _to_np(T_cw)
    R, t = T[:3, :3], T[:3, 3]
    n_c = R @ _to_np(n_w)
    return n_c, float(d_w) - float(n_c @ t)


def plane_c2w(n_c, d_c, T_cw):
    T = _to_np(T_cw)
    R, t = T[:3, :3], T[:3, 3]
    n_c = _to_np(n_c)
    return R.T @ n_c, float(d_c) + float(n_c @ t)


@dataclass
class PlaneMeas:              # one plane measured in the camera
    n: Tensor                 # (3,) float64, oriented so that d > 0
    d: float
    mask_cells: Tensor        # (Hc, Wc) bool — cells belonging to the plane
    n_px: int                 # number of valid pixels
    mse: float                # fit error (m^2), smallest PCA eigenvalue
    pts_c: np.ndarray = field(default=None, repr=False)   # (<=500, 3) camera-frame samples


@dataclass
class Anchor:
    id: int
    n_w: Tensor
    d_w: float
    samples_w: Tensor         # (<=500, 3) world samples, for the overlap test
    n_obs_kf: int
    active: bool
    created_kf: int
    weight: float = 0.0       # accumulated n_px used for the running average


class PlaneExtractor:
    """CAPE-lite (re-implementation)."""

    def __init__(self, cfg):
        self.cell = int(cfg.get("cell_size", 20))
        self.c_noise = float(cfg.get("c_noise", 3.0))
        self.merge_cos = float(np.cos(np.deg2rad(cfg.get("merge_angle_deg", 10))))
        self.merge_dist_k = float(cfg.get("merge_dist_k", 3.0))
        self.min_plane_pixels = int(cfg.get("min_plane_pixels", 3200))
        self.max_plane_depth = float(cfg.get("max_plane_depth", 4.0))
        self.min_valid_ratio = 0.9
        self.n_strip_rejected = 0         # regions dropped by the 2x2-block rule (diagnostics)
        self.n_merged = 0                 # segments merged in the last frame (diagnostics)

    @torch.no_grad()
    def __call__(self, depth, K) -> list:
        K = _to_np(K)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        z = depth.detach().double() if isinstance(depth, Tensor) else torch.as_tensor(depth, dtype=torch.float64)
        H, W = z.shape
        cs = self.cell
        Hc, Wc = H // cs, W // cs
        z = z[: Hc * cs, : Wc * cs]
        v, u = torch.meshgrid(
            torch.arange(Hc * cs, dtype=torch.float64, device=z.device),
            torch.arange(Wc * cs, dtype=torch.float64, device=z.device),
            indexing="ij",
        )
        valid = (z > 0) & (z <= self.max_plane_depth)
        P = torch.stack(((u - cx) / fx * z, (v - cy) / fy * z, z), dim=-1)          # (H, W, 3)
        P = torch.where(valid[..., None], P, torch.zeros_like(P))
        # (Hc, cs, Wc, cs, 3) -> (Hc, Wc, cs*cs, 3)
        Pc = P.reshape(Hc, cs, Wc, cs, 3).permute(0, 2, 1, 3, 4).reshape(Hc, Wc, cs * cs, 3)
        Vc = valid.reshape(Hc, cs, Wc, cs).permute(0, 2, 1, 3).reshape(Hc, Wc, cs * cs)
        cnt = Vc.sum(-1).cpu().numpy().astype(np.float64)                          # (Hc, Wc)
        s1 = Pc.sum(2).cpu().numpy()                                                 # (Hc, Wc, 3)
        s2 = torch.einsum("hwni,hwnj->hwij", Pc, Pc).cpu().numpy()                   # (Hc, Wc, 3, 3)
        Pc_np = Pc.cpu().numpy()
        Vc_np = Vc.cpu().numpy()

        # 1) cell-wise PCA and planarity test
        good = cnt >= self.min_valid_ratio * cs * cs
        cell_n = np.zeros((Hc, Wc, 3))
        cell_d = np.zeros((Hc, Wc))
        cell_mse = np.full((Hc, Wc), np.inf)
        cell_mu = np.zeros((Hc, Wc, 3))
        planar = np.zeros((Hc, Wc), dtype=bool)
        idx = np.argwhere(good)
        if idx.shape[0] > 0:
            n_ = cnt[good][:, None]
            mu = s1[good] / n_
            cov = s2[good] / n_[:, :, None] - np.einsum("ki,kj->kij", mu, mu)
            w, vec = np.linalg.eigh(cov)
            nrm = vec[:, :, 0]
            d = -np.einsum("ki,ki->k", nrm, mu)
            flip = d < 0
            nrm[flip] *= -1
            d[flip] *= -1
            mse = np.clip(w[:, 0], 0, None)
            ok = np.sqrt(mse) < self.c_noise * sigma_z(mu[:, 2]) + 0.001
            for k, (i, j) in enumerate(idx):
                cell_n[i, j], cell_d[i, j], cell_mse[i, j], cell_mu[i, j] = nrm[k], d[k], mse[k], mu[k]
                planar[i, j] = ok[k]

        # 2) region growing on the 4-neighbour grid, seeds by increasing mse; seed plane is the reference
        label = np.full((Hc, Wc), -1, dtype=np.int64)
        order = sorted((cell_mse[i, j], i, j) for i, j in np.argwhere(planar))
        regions = []
        for _, si, sj in order:
            if label[si, sj] >= 0:
                continue
            rid = len(regions)
            ref_n, ref_d = cell_n[si, sj], cell_d[si, sj]
            label[si, sj] = rid
            members = [(si, sj)]
            stack = [(si, sj)]
            while stack:
                ci, cj = stack.pop()
                for ni, nj in ((ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)):
                    if not (0 <= ni < Hc and 0 <= nj < Wc):
                        continue
                    if label[ni, nj] >= 0 or not planar[ni, nj]:
                        continue
                    if float(cell_n[ni, nj] @ ref_n) < self.merge_cos:
                        continue
                    mu = cell_mu[ni, nj]
                    if abs(float(ref_n @ mu) + ref_d) >= self.merge_dist_k * sigma_z(mu[2]) + 0.01:
                        continue
                    label[ni, nj] = rid
                    members.append((ni, nj))
                    stack.append((ni, nj))
            regions.append(members)

        # 3) merge adjacent segments with the same plane (CAPE's segment-merging step, added after the
        #    T9 overlays showed hung fabric panels split into interleaved fragments). Same thresholds as
        #    region growing, tested both ways on each segment's own refit plane; union-find on the grid.
        fits = []
        for members in regions:
            mi = np.array(members)
            n_tot = cnt[mi[:, 0], mi[:, 1]].sum()
            nrm, d, _, mu = _pca_from_sums(n_tot, s1[mi[:, 0], mi[:, 1]].sum(0), s2[mi[:, 0], mi[:, 1]].sum(0))
            fits.append((nrm, d, mu))
        parent = list(range(len(regions)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def same_plane(a, b):
            (na, da, ma), (nb, db, mb) = fits[a], fits[b]
            if float(na @ nb) < self.merge_cos:
                return False
            return (abs(float(na @ mb) + da) < self.merge_dist_k * sigma_z(mb[2]) + 0.01
                    and abs(float(nb @ ma) + db) < self.merge_dist_k * sigma_z(ma[2]) + 0.01)

        edges = set()
        for A, B in ((label[:, :-1], label[:, 1:]), (label[:-1, :], label[1:, :])):
            m = (A >= 0) & (B >= 0) & (A != B)
            edges.update(zip(A[m].tolist(), B[m].tolist()))
        for a, b in sorted(edges):
            if same_plane(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        merged = {}
        for rid in range(len(regions)):
            merged.setdefault(find(rid), []).extend(regions[rid])
        self.n_merged = len(regions) - len(merged)
        for root, members in merged.items():
            for i, j in members:
                label[i, j] = root
        regions = [merged.get(r, []) for r in range(len(regions))]

        # 4) refit each region on all its valid pixels (exact union of the cell sums), keep large ones
        planes = []
        for rid, members in enumerate(regions):
            if not members:
                continue
            mi = np.array(members)
            n_tot = cnt[mi[:, 0], mi[:, 1]].sum()
            if n_tot < self.min_plane_pixels:
                continue
            mask = label == rid
            # Cells straddling a crease between two planes pass the planarity test and chain into a
            # strip one cell wide (a digital line). A real plane contains at least one 2x2 block of cells.
            if not (mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).any():
                self.n_strip_rejected += 1
                continue
            S1 = s1[mi[:, 0], mi[:, 1]].sum(0)
            S2 = s2[mi[:, 0], mi[:, 1]].sum(0)
            nrm, d, mse, _ = _pca_from_sums(n_tot, S1, S2)
            pts = Pc_np[mask][Vc_np[mask]]                    # valid pixels, row-major cell order
            planes.append(
                PlaneMeas(
                    n=torch.from_numpy(nrm.copy()),
                    d=float(d),
                    mask_cells=torch.from_numpy(mask.copy()),
                    n_px=int(n_tot),
                    mse=float(mse),
                    pts_c=_even_subsample(pts, MAX_SAMPLES),
                )
            )
        return planes


class PlaneAnchorMap:
    def __init__(self, cfg, K, height, width):
        self.K = _to_np(K)
        self.H, self.W = int(height), int(width)
        self.cell = int(cfg.get("cell_size", 20))
        self.max_cos = float(np.cos(np.deg2rad(cfg.get("match_max_angle_deg", 10))))
        self.max_angle = float(np.deg2rad(cfg.get("match_max_angle_deg", 10)))
        self.max_dist = float(cfg.get("match_max_dist_m", 0.10))
        self.min_overlap = float(cfg.get("match_min_overlap", 0.3))
        self.min_sv = float(cfg.get("min_singular_value", 0.3))
        self.max_trans = float(cfg.get("solve_max_trans_m", 0.20))
        self.max_rot = float(np.deg2rad(cfg.get("solve_max_rot_deg", 15)))
        self.activate_after_kf = int(cfg.get("activate_after_kf", 2))
        self.sigma_n = float(cfg.get("sigma_normal", 0.02))
        self.sigma_d = float(cfg.get("sigma_dist", 0.01))
        self.anchors = {}
        self.next_id = 0
        self.first_kf = None
        # diagnostics of the last solve_pose call
        self.last_sigma_min = float("nan")
        self.last_solve_status = "none"   # none | plane | rejected
        self.n_back_facing = 0

    # ---------------------------------------------------------------- helpers
    def n_active(self):
        return sum(1 for a in self.anchors.values() if a.active)

    def _overlap(self, anchor, plane, R, t):
        X = anchor.samples_w.numpy() @ R.T + t
        z = X[:, 2]
        front = z > 1e-6
        if not front.any():
            return 0.0
        X = X[front]
        u = self.K[0, 0] * X[:, 0] / X[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * X[:, 1] / X[:, 2] + self.K[1, 2]
        mask = plane.mask_cells.numpy()
        Hc, Wc = mask.shape
        inside = (u >= 0) & (v >= 0) & (u < Wc * self.cell) & (v < Hc * self.cell)
        if not inside.any():
            return 0.0
        ci = (v[inside] // self.cell).astype(np.int64)
        cj = (u[inside] // self.cell).astype(np.int64)
        return float(mask[ci, cj].mean())

    # ---------------------------------------------------------------- API
    def match(self, planes, T_cw) -> list:
        """Greedy one-to-one (plane_idx, anchor_id) by cost angle/θmax + dist/dmax + (1 - overlap)."""
        T = _to_np(T_cw)
        R, t = T[:3, :3], T[:3, 3]
        cand = []
        for aid, a in self.anchors.items():
            n_c, d_c = plane_w2c(a.n_w, a.d_w, T)
            if d_c <= 0:
                self.n_back_facing += 1
                continue
            for pi, p in enumerate(planes):
                c = float(np.clip(p.n.numpy() @ n_c, -1.0, 1.0))
                if c < self.max_cos:
                    continue
                dist = abs(p.d - d_c)
                if dist > self.max_dist:
                    continue
                ov = self._overlap(a, p, R, t)
                if ov < self.min_overlap:
                    continue
                cost = np.arccos(c) / self.max_angle + dist / self.max_dist + (1.0 - ov)
                cand.append((cost, pi, aid))
        cand.sort()
        used_p, used_a, pairs = set(), set(), []
        for _, pi, aid in cand:
            if pi in used_p or aid in used_a:
                continue
            used_p.add(pi)
            used_a.add(aid)
            pairs.append((pi, aid))
        pairs.sort()
        return pairs

    def active_pairs(self, pairs):
        return [(pi, aid) for pi, aid in pairs if self.anchors[aid].active]

    def solve_pose(self, planes, pairs, T_prior) -> Optional[Tensor]:
        """Closed-form T_cw from >= 3 non-parallel active plane pairs; None if degenerate or rejected."""
        act = self.active_pairs(pairs)
        self.last_solve_status = "none"
        if len(act) == 0:
            self.last_sigma_min = 0.0
            return None
        M = np.stack([planes[pi].n.numpy() for pi, _ in act])                 # measured normals m_j
        N = np.stack([self.anchors[aid].n_w.numpy() for _, aid in act])       # anchor normals n_j
        dm = np.array([planes[pi].d for pi, _ in act])
        dw = np.array([self.anchors[aid].d_w for _, aid in act])
        w = np.array([planes[pi].n_px for pi, _ in act], dtype=np.float64)
        w = w / w.sum()
        sv = np.linalg.svd(M, compute_uv=False)
        self.last_sigma_min = float(sv[2]) if sv.shape[0] >= 3 else 0.0
        if len(act) < 3 or self.last_sigma_min < self.min_sv:
            return None
        # rotation: Kabsch on normals, R n_j ~ m_j
        Hm = (w[:, None, None] * np.einsum("ki,kj->kij", N, M)).sum(0)
        U, _, Vt = np.linalg.svd(Hm)
        V = Vt.T
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(V @ U.T))])
        Rm = V @ D @ U.T
        # translation: m_j^T t = d_w,j - d_meas,j (weighted least squares)
        A = M * np.sqrt(w)[:, None]
        b = (dw - dm) * np.sqrt(w)
        t = np.linalg.lstsq(A, b, rcond=None)[0]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rm, t
        # safety check against the prior (camera centres and relative rotation)
        Tp = _to_np(T_prior)
        c_new = -Rm.T @ t
        c_pri = -Tp[:3, :3].T @ Tp[:3, 3]
        dR = Rm @ Tp[:3, :3].T
        ang = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))
        if np.linalg.norm(c_new - c_pri) > self.max_trans or ang > self.max_rot:
            self.last_solve_status = "rejected"
            return None
        self.last_solve_status = "plane"
        dev = T_prior.device if isinstance(T_prior, Tensor) else "cpu"
        dt = T_prior.dtype if isinstance(T_prior, Tensor) else torch.float64
        return torch.from_numpy(T).to(device=dev, dtype=dt)

    def make_loss_ctx(self, planes, pairs, device, dtype):
        """Stack active pairs once per frame; reused by every tracking iteration."""
        act = self.active_pairs(pairs)
        if not act:
            return None
        w = torch.tensor([float(planes[pi].n_px) for pi, _ in act], dtype=torch.float64)
        return dict(
            m=torch.stack([planes[pi].n for pi, _ in act]).to(device=device, dtype=dtype),
            n_w=torch.stack([self.anchors[aid].n_w for _, aid in act]).to(device=device, dtype=dtype),
            d_meas=torch.tensor([planes[pi].d for pi, _ in act], dtype=dtype, device=device),
            d_w=torch.tensor([self.anchors[aid].d_w for _, aid in act], dtype=dtype, device=device),
            w=(w / w.sum()).to(device=device, dtype=dtype),
            inv_sn2=1.0 / self.sigma_n ** 2,
            inv_sd2=1.0 / self.sigma_d ** 2,
        )

    @staticmethod
    def plane_loss_ctx(T_cw: Tensor, ctx) -> Tensor:
        R, t = T_cw[:3, :3], T_cw[:3, 3]
        Rn = ctx["n_w"] @ R.T                                   # (k, 3) = (R n_j)^T
        r_n = ((ctx["m"] - Rn) ** 2).sum(-1)
        r_d = ctx["d_meas"] - (ctx["d_w"] - Rn @ t)
        return (ctx["w"] * (r_n * ctx["inv_sn2"] + r_d ** 2 * ctx["inv_sd2"])).sum()

    def plane_loss(self, T_cw: Tensor, planes, pairs) -> Tensor:
        ctx = self.make_loss_ctx(planes, pairs, T_cw.device, T_cw.dtype)
        if ctx is None:
            return T_cw.sum() * 0.0
        return self.plane_loss_ctx(T_cw, ctx)

    def _new_anchor(self, kf_id, p, n_w, d_w, X_w):
        a = Anchor(
            id=self.next_id,
            n_w=torch.from_numpy(n_w),
            d_w=float(d_w),
            samples_w=torch.from_numpy(X_w),
            n_obs_kf=1,
            active=False,
            created_kf=int(kf_id),
            weight=float(p.n_px),
        )
        a.active = (a.n_obs_kf >= self.activate_after_kf) or (a.created_kf == self.first_kf)
        self.anchors[a.id] = a
        self.next_id += 1

    def update_at_keyframe(self, kf_id, planes, pairs, T_cw) -> None:
        """Anchors are updated only from measured planes and keyframe poses, never from Gaussians."""
        if self.first_kf is None:
            self.first_kf = int(kf_id)
        T = _to_np(T_cw)
        R, t = T[:3, :3], T[:3, 3]
        matched = {pi: aid for pi, aid in pairs}
        for pi, p in enumerate(planes):
            n_w, d_w = plane_c2w(p.n.numpy(), p.d, T)
            X_w = (p.pts_c - t) @ R           # R^T (x_c - t), row vectors
            if pi not in matched:
                self._new_anchor(kf_id, p, n_w, d_w, X_w)
                continue
            a = self.anchors[matched[pi]]
            wa, wp = a.weight, float(p.n_px)
            n_new = (wa * a.n_w.numpy() + wp * n_w) / (wa + wp)
            n_new /= np.linalg.norm(n_new)
            a.n_w = torch.from_numpy(n_new)
            a.d_w = float((wa * a.d_w + wp * d_w) / (wa + wp))
            a.weight = wa + wp
            a.samples_w = torch.from_numpy(_even_subsample(np.concatenate([a.samples_w.numpy(), X_w]), MAX_SAMPLES))
            a.n_obs_kf += 1
            a.active = (a.n_obs_kf >= self.activate_after_kf) or (a.created_kf == self.first_kf)
