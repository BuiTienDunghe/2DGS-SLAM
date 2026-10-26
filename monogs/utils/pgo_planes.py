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
