"""PLAN MONOGS-PW §2.5 (part 4): pose graph with plane landmarks ("ours") and the StructureGS-style closure ("sgs").

The installed GTSAM python build has no OrientedPlane3Factor, so the graph is solved with
scipy.optimize.least_squares (trf, sparse Jacobian, Huber loss), the fallback named in PLAN §6. Residuals:
  prior     : keyframe 0 fixed (sigma 1e-6 -> held constant);
  odometry  : Log(M_est M_meas^-1) / [sigma_t, sigma_r], M = T_j T_i^-1 (w2c, consecutive keyframes);
  plane     : [m - R n_w ; d_meas - (d_w - (R n_w).t)] / [sigma_n x3, sigma_d(z)]  (the E1c residual);
  sgs edge  : Log(T_j T_k^-1 T_jk^-1) / [sigma_t, sigma_r].
Pose parameter: T_w2c = Exp(xi) T0 (left, camera frame). Plane parameter: n = normalize(n0 + B dth), d = d0 + dd.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

CHI2_3 = 11.34


# ------------------------------------------------------------------ SE(3), [rho, theta] like SE3_exp of MonoGS
def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(w):
    a = np.linalg.norm(w)
    W = skew(w)
    if a < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + np.sin(a) / a * W + (1 - np.cos(a)) / a ** 2 * W @ W


def _V(w):
    a = np.linalg.norm(w)
    W = skew(w)
    if a < 1e-8:
        return np.eye(3) + 0.5 * W + W @ W / 6.0
    return np.eye(3) + (1 - np.cos(a)) / a ** 2 * W + (a - np.sin(a)) / a ** 3 * W @ W


def se3_exp(xi):
    T = np.eye(4)
    T[:3, :3] = so3_exp(xi[3:])
    T[:3, 3] = _V(xi[3:]) @ xi[:3]
    return T


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    a = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a < 1e-8:
        return 0.5 * w
    if np.pi - a < 1e-6:
        B = (R + np.eye(3)) / 2
        k = int(np.argmax(np.diag(B)))
        ax = B[:, k] / np.sqrt(max(B[k, k], 1e-12))
        return a * ax / np.linalg.norm(ax)
    return a / (2 * np.sin(a)) * w


def se3_log(T):
    w = so3_log(T[:3, :3])
    return np.concatenate([np.linalg.solve(_V(w), T[:3, 3]), w])


def inv(T):
    o = np.eye(4)
    o[:3, :3] = T[:3, :3].T
    o[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return o


def tangent_basis(n):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(n, a)
    b1 /= np.linalg.norm(b1)
    return np.stack([b1, np.cross(n, b1)], 1)


def pose_change(T_old, T_new):
    """(camera-centre displacement m, rotation deg) between two w2c poses."""
    c0 = -T_old[:3, :3].T @ T_old[:3, 3]
    c1 = -T_new[:3, :3].T @ T_new[:3, 3]
    dR = T_new[:3, :3] @ T_old[:3, :3].T
    return float(np.linalg.norm(c1 - c0)), float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def plane_residual(m, d_meas, n_w, d_w, T):
    R, t = T[:3, :3], T[:3, 3]
    Rn = R @ n_w
    return np.concatenate([m - Rn, [d_meas - (d_w - Rn @ t)]])


# ------------------------------------------------------------------ the graph
class Graph:
    """kfs: ordered keyframe ids (kfs[0] fixed). poses: {kf: T_w2c 4x4}. odo: [(i, j, M_meas)].
    planes: {aid: (n_w, d_w)}. obs: [(aid, kf, m, d, z, tag)]. edges: [(j, k, T_jk)] (sgs)."""

    def __init__(self, kfs, poses, odo, planes, obs, edges, noise, s_odo_t, s_odo_r):
        self.kfs = list(kfs)
        self.var = self.kfs[1:]
        self.pi = {k: i for i, k in enumerate(self.var)}
        self.aids = list(planes)
        self.ai = {a: i for i, a in enumerate(self.aids)}
        self.P0 = {k: poses[k].copy() for k in self.kfs}
        self.L0 = {a: (np.asarray(planes[a][0], dtype=np.float64), float(planes[a][1])) for a in self.aids}
        self.B = {a: tangent_basis(self.L0[a][0]) for a in self.aids}
        self.odo, self.obs, self.edges = odo, obs, edges
        self.noise, self.st, self.sr = noise, float(s_odo_t), float(s_odo_r)
        self.n6 = 6 * len(self.var)

    def unpack(self, x):
        P = {self.kfs[0]: self.P0[self.kfs[0]]}
        for k, i in self.pi.items():
            P[k] = se3_exp(x[6 * i:6 * i + 6]) @ self.P0[k]
        L = {}
        for a, i in self.ai.items():
            o = self.n6 + 3 * i
            n0, d0 = self.L0[a]
            n = n0 + self.B[a] @ x[o:o + 2]
            L[a] = (n / np.linalg.norm(n), d0 + x[o + 2])
        return P, L

    def _cols_pose(self, k):
        return [] if k not in self.pi else list(range(6 * self.pi[k], 6 * self.pi[k] + 6))

    def residuals(self, x, obs=None):
        obs = self.obs if obs is None else obs
        P, L = self.unpack(x)
        r = []
        w6 = np.array([self.st] * 3 + [self.sr] * 3)
        for i, j, M in self.odo:
            r.append(se3_log(P[j] @ inv(P[i]) @ inv(M)) / w6)
        for j, k, Tjk in self.edges:
            r.append(se3_log(P[j] @ inv(P[k]) @ inv(Tjk)) / w6)
        for a, kf, m, d, z, _ in obs:
            n_w, d_w = L[a]
            s = np.array([self.noise.sn] * 3 + [self.noise.sd(z)])
            r.append(plane_residual(m, d, n_w, d_w, P[kf]) / s)
        return np.concatenate(r) if r else np.zeros(0)

    def sparsity(self, obs=None):
        obs = self.obs if obs is None else obs
        rows = 6 * (len(self.odo) + len(self.edges)) + 4 * len(obs)
        S = lil_matrix((rows, self.n6 + 3 * len(self.aids)), dtype=int)
        r = 0
        for i, j, _ in list(self.odo) + [(e[0], e[1], None) for e in self.edges]:
            for c in self._cols_pose(i) + self._cols_pose(j):
                S[r:r + 6, c] = 1
            r += 6
        for a, kf, *_ in obs:
            for c in self._cols_pose(kf) + list(range(self.n6 + 3 * self.ai[a], self.n6 + 3 * self.ai[a] + 3)):
                S[r:r + 4, c] = 1
            r += 4
        return S

    def solve(self, obs=None, huber_k=None, max_nfev=100):
        obs = self.obs if obs is None else obs
        x0 = np.zeros(self.n6 + 3 * len(self.aids))
        if x0.size == 0:
            return dict(ok=True, x=x0, status=1, nfev=0)
        kw = dict(loss="huber", f_scale=huber_k) if huber_k else {}
        try:
            res = least_squares(lambda x: self.residuals(x, obs), x0, jac_sparsity=self.sparsity(obs), method="trf",
                                max_nfev=max_nfev * max(1, x0.size // 50), x_scale="jac", **kw)
        except Exception as e:  # noqa: BLE001
            return dict(ok=False, error=str(e), x=x0, status=-99, nfev=0)
        # status 0 = evaluation cap reached (PLAN: at most 100 iterations), not a solver failure
        return dict(ok=bool(res.status >= 0 and np.all(np.isfinite(res.x))), x=res.x, status=int(res.status),
                    nfev=int(res.nfev), cost=float(res.cost))

    def factor_e2(self, x, ob):
        P, L = self.unpack(x)
        a, kf, m, d, z, _ = ob
        s = np.array([self.noise.sn] * 3 + [self.noise.sd(z)])
        r = plane_residual(m, d, L[a][0], L[a][1], P[kf]) / s
        return float(r @ r)


def solve_ours(kfs, poses, odo, planes, obs, closure_idx, noise, s_odo_t, s_odo_r, k, huber_k=1.345,
               chi2=CHI2_3, max_dt=1.0, max_dr=30.0, max_nfev=100):
    """PLAN §2.5 "ours". closure_idx: indices into obs of the closure factors of keyframe k (tier 1 + tier 2),
    tested without the kernel after the solve; rejected ones are removed and the graph is solved once more
    from the initial values. Returns dict(status, poses, planes, dropped, ...)."""
    g = Graph(kfs, poses, odo, planes, obs, [], noise, s_odo_t, s_odo_r)
    sol = g.solve(huber_k=huber_k, max_nfev=max_nfev)
    out = dict(n_nodes=len(kfs), n_odo=len(odo), n_plane=len(obs), n_closure=len(closure_idx), dropped=[],
               resolved=False)
    if not sol["ok"]:
        out.update(status="aborted", reason=f"solver: {sol.get('error', sol['status'])}")
        return out
    bad = [i for i in closure_idx if g.factor_e2(sol["x"], obs[i]) > chi2]
    if bad:
        keep = [o for i, o in enumerate(obs) if i not in set(bad)]
        g = Graph(kfs, poses, odo, planes, keep, [], noise, s_odo_t, s_odo_r)
        sol = g.solve(huber_k=huber_k, max_nfev=max_nfev)
        out["resolved"] = True
        out["dropped"] = bad
        if not sol["ok"]:
            out.update(status="aborted", reason=f"solver (2nd): {sol.get('error', sol['status'])}")
            return out
    P, L = g.unpack(sol["x"])
    dt, dr = pose_change(poses[k], P[k])
    out.update(max_kf_shift_m=max(pose_change(poses[q], P[q])[0] for q in kfs),
               max_kf_rot_deg=max(pose_change(poses[q], P[q])[1] for q in kfs), k_shift_m=dt, k_rot_deg=dr,
               nfev=sol["nfev"])
    if dt > max_dt or dr > max_dr:
        out.update(status="aborted", reason=f"keyframe {k} moved {dt:.3f} m / {dr:.1f} deg")
        return out
    out.update(status="applied", poses=P, planes=L)
    return out


def relpose_from_planes(pk, pj):
    """T_jk (x_j = R x_k + t) from >= 2 non-parallel plane pairs: R by Kabsch n_k -> n_j, t least squares from
    n_j . t = d_k - d_j (minimum-norm solution). pk, pj: [(n, d)] in the k and j cameras."""
    Nk = np.stack([p[0] for p in pk])
    Nj = np.stack([p[0] for p in pj])
    H = Nk.T @ Nj
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    b = np.array([pk[i][1] - pj[i][1] for i in range(len(pk))])
    t = np.linalg.lstsq(Nj, b, rcond=None)[0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def icp_refine(T_jk, src_pts, dst_pts, dst_normals, max_dist=0.05, iters=30):
    """Open3D point-to-plane ICP, source points in camera k, target points (with plane normals) in camera j."""
    import open3d as o3d
    s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(src_pts, dtype=np.float64)))
    d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(dst_pts, dtype=np.float64)))
    d.normals = o3d.utility.Vector3dVector(np.asarray(dst_normals, dtype=np.float64))
    r = o3d.pipelines.registration.registration_icp(
        s, d, max_dist, T_jk, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
    return np.asarray(r.transformation), float(r.fitness), float(r.inlier_rmse)


def solve_sgs(kfs, poses, odo, edges, s_odo_t, s_odo_r, k, max_dt=1.0, max_dr=30.0, max_nfev=100):
    """PLAN §2.5 "sgs": prior + odometry + every sgs closure edge (sigma_odo, no kernel, no landmarks)."""
    g = Graph(kfs, poses, odo, {}, [], edges, None, s_odo_t, s_odo_r)
    sol = g.solve(max_nfev=max_nfev)
    out = dict(n_nodes=len(kfs), n_odo=len(odo), n_plane=0, n_closure=len(edges), dropped=[], resolved=False)
    if not sol["ok"]:
        out.update(status="aborted", reason=f"solver: {sol.get('error', sol['status'])}")
        return out
    P, _ = g.unpack(sol["x"])
    dt, dr = pose_change(poses[k], P[k])
    out.update(max_kf_shift_m=max(pose_change(poses[q], P[q])[0] for q in kfs),
               max_kf_rot_deg=max(pose_change(poses[q], P[q])[1] for q in kfs), k_shift_m=dt, k_rot_deg=dr,
               nfev=sol["nfev"])
    if dt > max_dt or dr > max_dr:
        out.update(status="aborted", reason=f"keyframe {k} moved {dt:.3f} m / {dr:.1f} deg")
        return out
    out.update(status="applied", poses=P, planes={})
    return out


# ============================================================================ PLAN MONOGS-PW2 §2.4 (appended)
# Vectorised residuals (same maths as Graph.residuals) and IRLS Huber on the norm of each plane factor
# (odometry without kernel). Graph / solve_ours above stay as in R1 (PGO.robust: scipy_huber).


def _skew_b(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1), np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def _coef(a):
    small = a < 1e-6
    a_ = np.where(small, 1.0, a)
    A = np.where(small, 1 - a ** 2 / 6, np.sin(a_) / a_)
    B = np.where(small, 0.5 - a ** 2 / 24, (1 - np.cos(a_)) / a_ ** 2)
    C = np.where(small, 1.0 / 6 - a ** 2 / 120, (a_ - np.sin(a_)) / a_ ** 3)
    return A, B, C


def se3_exp_b(xi):
    """Batched SE3_exp (MonoGS order [rho, theta]) -> R (K,3,3), t (K,3)."""
    th = xi[:, 3:]
    a = np.linalg.norm(th, axis=1)
    W = _skew_b(th)
    W2 = W @ W
    A, B, C = _coef(a)
    I = np.eye(3)[None]
    R = I + A[:, None, None] * W + B[:, None, None] * W2
    V = I + B[:, None, None] * W + C[:, None, None] * W2
    return R, np.einsum("kij,kj->ki", V, xi[:, :3])


def se3_log_b(R, t):
    """Batched se3_log of (R, t) -> (K, 6) [rho, theta]; angles near pi fall back to the scalar version."""
    c = np.clip((np.einsum("kii->k", R) - 1) / 2, -1.0, 1.0)
    a = np.arccos(c)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    small = a < 1e-6
    s = np.where(small, 1.0, np.sin(np.where(small, 1.0, a)))
    th = np.where(small[:, None], 0.5 * w, (a / (2 * s))[:, None] * w)
    far = np.pi - a < 1e-4
    for i in np.nonzero(far)[0]:
        th[i] = so3_log(R[i])
    W = _skew_b(th)
    A, B, C = _coef(np.linalg.norm(th, axis=1))
    V = np.eye(3)[None] + B[:, None, None] * W + C[:, None, None] * (W @ W)
    rho = np.linalg.solve(V, t[..., None])[..., 0]
    return np.concatenate([rho, th], 1)


class GraphVec(Graph):
    """Same graph and parametrisation as Graph, residuals computed with numpy batches; plane-factor rows can be
    scaled by per-factor IRLS weights (sqrt(w))."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        kix = {k: i for i, k in enumerate(self.kfs)}
        self.kix = kix
        self.var_mask = np.array([k in self.pi for k in self.kfs])
        self.var_pos = np.array([self.pi.get(k, -1) for k in self.kfs])
        R0 = np.stack([self.P0[k][:3, :3] for k in self.kfs])
        t0 = np.stack([self.P0[k][:3, 3] for k in self.kfs])
        self.R0, self.t0 = R0, t0
        ed = list(self.odo) + list(self.edges)
        self.n_odo_rows = len(ed)
        self.oi = np.array([kix[e[0]] for e in ed], dtype=int)
        self.oj = np.array([kix[e[1]] for e in ed], dtype=int)
        Minv = np.stack([inv(e[2]) for e in ed]) if ed else np.zeros((0, 4, 4))
        self.MRi, self.Mti = Minv[:, :3, :3], Minv[:, :3, 3]
        self.w6 = np.array([self.st] * 3 + [self.sr] * 3)
        self._set_obs(self.obs)
        self.n0 = np.stack([self.L0[a][0] for a in self.aids]) if self.aids else np.zeros((0, 3))
        self.d0 = np.array([self.L0[a][1] for a in self.aids])
        self.Bs = np.stack([self.B[a] for a in self.aids]) if self.aids else np.zeros((0, 3, 2))
        self.irls_w = np.ones(len(self.obs))

    def _set_obs(self, obs):
        self.obs = obs
        self.pk = np.array([self.kix[o[1]] for o in obs], dtype=int)
        self.pa = np.array([self.ai[o[0]] for o in obs], dtype=int)
        self.pm = np.stack([np.asarray(o[2], dtype=np.float64) for o in obs]) if obs else np.zeros((0, 3))
        self.pd = np.array([float(o[3]) for o in obs])
        if obs and self.noise is not None:
            self.ps = np.stack([np.array([self.noise.sn] * 3 + [self.noise.sd(o[4])]) for o in obs])
        else:
            self.ps = np.ones((len(obs), 4))

    def poses(self, x):
        R, t = self.R0.copy(), self.t0.copy()
        if self.var:
            xi = x[:self.n6].reshape(-1, 6)
            Re, te = se3_exp_b(xi[self.var_pos[self.var_mask]])
            R[self.var_mask] = Re @ self.R0[self.var_mask]
            t[self.var_mask] = np.einsum("kij,kj->ki", Re, self.t0[self.var_mask]) + te
        return R, t

    def planes(self, x):
        if not self.aids:
            return self.n0, self.d0
        y = x[self.n6:].reshape(-1, 3)
        n = self.n0 + np.einsum("aij,aj->ai", self.Bs, y[:, :2])
        return n / np.linalg.norm(n, axis=1, keepdims=True), self.d0 + y[:, 2]

    def plane_res(self, x):
        R, t = self.poses(x)
        n, d = self.planes(x)
        Rn = np.einsum("kij,kj->ki", R[self.pk], n[self.pa])
        r = np.concatenate([self.pm - Rn, (self.pd - (d[self.pa] - np.einsum("ki,ki->k", Rn, t[self.pk])))[:, None]], 1)
        return r / self.ps

    def odo_res(self, x):
        if self.n_odo_rows == 0:
            return np.zeros((0, 6))
        R, t = self.poses(x)
        # E = P_j P_i^-1 M^-1
        Ri, ti = R[self.oi], t[self.oi]
        Rj, tj = R[self.oj], t[self.oj]
        Rij = Rj @ np.transpose(Ri, (0, 2, 1))
        tij = tj - np.einsum("kij,kj->ki", Rij, ti)
        RE = Rij @ self.MRi
        tE = np.einsum("kij,kj->ki", Rij, self.Mti) + tij
        return se3_log_b(RE, tE) / self.w6

    def residuals(self, x, obs=None):
        if obs is not None and obs is not self.obs:
            self._set_obs(obs)
            self.irls_w = np.ones(len(obs))
        r_o = self.odo_res(x).ravel()
        r_p = (self.plane_res(x) * np.sqrt(self.irls_w)[:, None]).ravel() if len(self.obs) else np.zeros(0)
        return np.concatenate([r_o, r_p])

    def factor_e2_all(self, x):
        r = self.plane_res(x)
        return np.einsum("ki,ki->k", r, r)


def solve_ours_irls(kfs, poses, odo, planes, obs, closure_idx, noise, s_odo_t, s_odo_r, k, huber_k=1.345,
                    chi2=CHI2_3, max_dt=1.0, max_dr=30.0, max_nfev=100, outer=3):
    """PLAN MONOGS-PW2 §2.4: IRLS Huber on the norm of each plane factor (w = min(1, k/||r||), rows scaled by
    sqrt(w), least_squares with loss linear), 3 outer iterations, odometry without kernel; chi2 test of the closure
    factors without kernel; rejected ones removed and the graph solved again from the initial values."""
    import time as _t
    t0 = _t.perf_counter()

    def run(obs_):
        g = GraphVec(kfs, poses, odo, planes, obs_, [], noise, s_odo_t, s_odo_r)
        x = np.zeros(g.n6 + 3 * len(g.aids))
        S = g.sparsity(obs_)
        res = None
        for _ in range(outer):
            if len(obs_):
                nr = np.sqrt(g.factor_e2_all(x))
                g.irls_w = np.minimum(1.0, huber_k / np.maximum(nr, 1e-12))
            try:
                res = least_squares(g.residuals, x, jac_sparsity=S, method="trf", max_nfev=max_nfev, x_scale="jac")
            except Exception as e:  # noqa: BLE001
                return g, None, str(e)
            x = res.x
        return g, res, None

    out = dict(n_nodes=len(kfs), n_odo=len(odo), n_plane=len(obs), n_closure=len(closure_idx), dropped=[], resolved=False)
    g, res, err = run(obs)
    if res is None or not np.all(np.isfinite(res.x)):
        out.update(status="aborted", reason=f"solver: {err}")
        return out
    e2 = g.factor_e2_all(res.x) if len(obs) else np.zeros(0)
    w_first = g.irls_w.copy()
    bad = [i for i in closure_idx if e2[i] > chi2]
    out["closure_e2"] = {int(i): float(e2[i]) for i in closure_idx}
    out["closure_w"] = {int(i): float(w_first[i]) for i in closure_idx}
    if bad:
        keep_idx = [i for i in range(len(obs)) if i not in set(bad)]
        g, res, err = run([obs[i] for i in keep_idx])
        out["resolved"] = True
        out["dropped"] = bad
        if res is None or not np.all(np.isfinite(res.x)):
            out.update(status="aborted", reason=f"solver (2nd): {err}")
            return out
    R, t = g.poses(res.x)
    n, d = g.planes(res.x)
    P = {}
    for i, q in enumerate(kfs):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R[i], t[i]
        P[q] = T
    L = {a: (n[i], float(d[i])) for i, a in enumerate(g.aids)}
    dt, dr = pose_change(poses[k], P[k])
    out.update(max_kf_shift_m=max(pose_change(poses[q], P[q])[0] for q in kfs),
               max_kf_rot_deg=max(pose_change(poses[q], P[q])[1] for q in kfs), k_shift_m=dt, k_rot_deg=dr,
               nfev=int(res.nfev), solve_s=_t.perf_counter() - t0)
    if dt > max_dt or dr > max_dr:
        out.update(status="aborted", reason=f"keyframe {k} moved {dt:.3f} m / {dr:.1f} deg")
        return out
    out.update(status="applied", poses=P, planes=L)
    return out
