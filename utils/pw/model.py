"""PLAN A2G-P1 §3.2–3.4: one error model, one residual, one residual covariance.

residual r(a|b) = [B_b^T n_a ; n_b^T (p_a - o_b)]        (a, b in the same frame; anchors use p_a = o_a)
J_a  = [[B_b^T B_a, 0], [0, n_b^T n_a]]                   (x_a in (B_a, p_a))
J_b  = [[-(n_b^T n_a) I2, 0], [l^T, -1]],  l = B_b^T (p_a - o_b)
J_xi = rows k=1,2: [0, (b_k x n_a)^T];  distance: [-n_c^T, (n_c x p_a)^T]   (b = anchor taken into the camera)
H_EKF = -J_b.  Shift of the reference point o -> o' (l' = B^T (o' - o)): A = [[I2, 0], [-l'^T, 1]].
"""
from collections import deque
from functools import lru_cache

import numpy as np
from scipy.stats import chi2 as _chi2


@lru_cache(maxsize=None)
def chi2(dof, alpha=0.99):
    if dof <= 0:
        return float("inf")
    return float(_chi2.ppf(alpha, dof))


CHI2_3 = chi2(3)
CHI2_2 = chi2(2)


def nis(r, S):
    r = np.asarray(r, dtype=np.float64)
    return float(r @ np.linalg.solve(S, r))


def residual(n_a, p_a, n_b, o_b, B_b):
    return np.concatenate([B_b.T @ n_a, [float(n_b @ (p_a - o_b))]])


def jac_a(n_a, B_a, n_b, B_b):
    J = np.zeros((3, 3))
    J[:2, :2] = B_b.T @ B_a
    J[2, 2] = float(n_b @ n_a)
    return J


def jac_b(n_a, p_a, n_b, o_b, B_b):
    J = np.zeros((3, 3))
    J[:2, :2] = -float(n_b @ n_a) * np.eye(2)
    J[2, :2] = B_b.T @ (p_a - o_b)
    J[2, 2] = -1.0
    return J


def jac_xi(n_a, p_a, n_c, B_c):
    J = np.zeros((3, 6))
    J[0, 3:] = np.cross(B_c[:, 0], n_a)
    J[1, 3:] = np.cross(B_c[:, 1], n_a)
    J[2, :3] = -n_c
    J[2, 3:] = np.cross(n_c, p_a)
    return J


def shift_matrix(l_new):
    A = np.eye(3)
    A[2, :2] = -np.asarray(l_new, dtype=np.float64)
    return A


KINDS = ("seg_seg", "seg_anchor_pred", "seg_anchor_opt", "jc", "anchor_anchor", "keep_u")


def residual_cov(kind, Ja=None, Ra=None, Jb=None, Pb=None, Jx=None, X=None, Delta=None):
    """S for one comparison (PLAN §3.4 table). X is the 6x6 pose covariance entering through J_xi
    (K2: Q_pred + D_j or Cov_post + D_j; K3: D_j; K5: C(t_j, t_k)). Delta replaces J_xi X J_xi^T by
    J_b Delta J_b^T (d_mode delta). K4 is built by jc_sigma, K6 by keep_u."""
    assert kind in KINDS and kind not in ("jc", "keep_u")
    S = np.zeros((3, 3))
    if Ja is not None:
        S += Ja @ Ra @ Ja.T
    if Jb is not None:
        S += Jb @ Pb @ Jb.T
    if kind == "seg_seg":
        return 0.5 * (S + S.T)
    if Delta is not None:
        S += Jb @ Delta @ Jb.T
    elif Jx is not None and X is not None:
        S += Jx @ X @ Jx.T
    return 0.5 * (S + S.T)


def ang_nis(r, S):
    return nis(r[:2], S[:2, :2])


# ------------------------------------------------------------------ noise sources


class SensorNoise:
    """(a) segment measurement noise R_s in (B_s, p_s)."""

    def __init__(self, a=0.0, b=1.425e-3, sigma_theta0_deg=0.5, sigma_d0_m=0.005):
        self.a, self.b = float(a), float(b)
        self.th0 = float(np.deg2rad(sigma_theta0_deg or 0.0))
        self.d0 = float(sigma_d0_m or 0.0)

    def sigma_z(self, z):
        return self.a + self.b * np.square(z)

    def R(self, N_s, lam1, lam2, zbar, nv):
        s2 = float(self.sigma_z(zbar)) ** 2 * float(nv) ** 2
        return np.diag([s2 / (N_s * lam1) + self.th0 ** 2, s2 / (N_s * lam2) + self.th0 ** 2, s2 / N_s + self.d0 ** 2])


class PredNoise:
    """(c) prediction error of the constant-velocity gate T_gate,k = Exp(xi_{k-1}) T_{k-1} (PLAN v2 §3.3):
    eps_k = Log(T_opt,k T_gate,k^-1); Q_pred = full 6x6 mean of eps eps^T over the last W frames (+1e-10 I)."""

    def __init__(self, window=30, sigma_rho_m=0.01, sigma_theta_deg=0.5):
        self.buf = deque(maxlen=int(window))
        self.Q0 = np.diag([float(sigma_rho_m) ** 2] * 3 + [float(np.deg2rad(sigma_theta_deg)) ** 2] * 3)

    def push(self, eps):
        self.buf.append(np.asarray(eps, dtype=np.float64))

    def Q(self):
        if not self.buf:
            return self.Q0.copy()
        X = np.stack(self.buf)
        return X.T @ X / len(X) + 1e-10 * np.eye(6)

    def sigmas(self):
        q = np.diag(self.Q())
        return float(np.sqrt(q[:3].mean())), float(np.sqrt(q[3:].mean()))


class OffsetEstimator:
    """Measured anchor-map offset b_hat (PLAN v2 §3.2), in the anchor's (B_j, o_j) coordinates.
    Samples z = -J_b^-1 r (apparent offset of the original anchor w.r.t. the current map/pose, at T_opt) with
    R_z = J_b^-1 (J_a R_s J_a^T) J_b^-T are accepted when (z - b)^T (R_z + V(k))^-1 (z - b) <= chi2_3,
    V(k) = V + Gamma (k - tau). b_hat = weighted mean over accepted samples of the last W frames,
    V = f (sum W_i)^-1 with f = max(1, chi2 / (3 (n - 1))) for n >= 3."""

    def __init__(self, P, k0, window=30):
        self.b = np.zeros(3)
        self.V = np.array(P, dtype=np.float64)
        self.tau = int(k0)
        self.window = int(window)
        self.buf = deque()
        self.f = 1.0
        self.n_accept = 0
        self.n_reject = 0

    def cov_at(self, k, Gamma):
        return self.V + np.asarray(Gamma) * max(0, int(k) - self.tau)

    def test(self, z, Rz, k, Gamma):
        d = np.asarray(z) - self.b
        m = float(d @ np.linalg.solve(Rz + self.cov_at(k, Gamma), d))
        return m, d

    def sample(self, z, Rz, k, Gamma, min_gap=5):
        """Returns (accepted, mahalanobis, gamma_obs or None)."""
        m, d = self.test(z, Rz, k, Gamma)
        if m > CHI2_3:
            self.n_reject += 1
            return False, m, None
        gap = int(k) - self.tau
        gamma_obs = None
        if gap >= min_gap:
            gamma_obs = np.diag(np.outer(d, d) - Rz - self.V) / gap     # unclipped; the running mean is clipped
        self.buf.append((int(k), np.asarray(z, dtype=np.float64), np.asarray(Rz, dtype=np.float64)))
        while self.buf and self.buf[0][0] <= int(k) - self.window:
            self.buf.popleft()
        Ws = [np.linalg.inv(R) for _, _, R in self.buf]
        Wsum = sum(Ws)
        Vm = np.linalg.inv(Wsum)
        b = Vm @ sum(W @ zz for W, (_, zz, _) in zip(Ws, self.buf))
        n = len(self.buf)
        f = 1.0
        if n >= 3:
            chi = sum(float((zz - b) @ W @ (zz - b)) for W, (_, zz, _) in zip(Ws, self.buf))
            f = max(1.0, chi / (3 * (n - 1)))
        self.b, self.f = b, f
        self.V = 0.5 * (f * Vm + (f * Vm).T)
        self.tau = int(k)
        self.n_accept += 1
        return True, m, gamma_obs

    def y(self, n, o, B, oplus):
        """Offset-corrected anchor y_j = x_j (+) b_hat (decisions only)."""
        return oplus(n, o, B, self.b)
