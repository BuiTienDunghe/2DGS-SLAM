"""PLAN A2G-P1 core §2: the uncertainty API (reused by later modules). Knows nothing about tracking: no anchor
state, no import of world or association (I11).

Every plane-world decision is nis(r, S) <= chi2(dof) with S = residual_cov(...).

residual r(a|b) = [B_b^T n_a ; n_b^T (p_a - o_b)]
J_a = [[B_b^T B_a, 0], [0, n_b^T n_a]];  J_b = [[-(n_b^T n_a) I2, 0], [l^T, -1]], l = B_b^T (p_a - o_b)
J_xi rows k=1,2: [0, (b_k x n_a)^T]; distance: [-n_c^T, (n_c x p_a)^T]   (b = plane taken into the camera)
"""
from collections import deque
from functools import lru_cache

import numpy as np
from scipy.stats import chi2 as _chi2
from scipy.stats import ncx2 as _ncx2

ALPHA = 0.99


@lru_cache(maxsize=None)
def chi2(dof, alpha=ALPHA):
    if dof <= 0:
        return float("inf")
    return float(_chi2.ppf(alpha, dof))


CHI2_3 = chi2(3)
CHI2_2 = chi2(2)


def nis(r, S):
    r = np.asarray(r, dtype=np.float64)
    return float(r @ np.linalg.solve(S, r))


def ang_nis(r, S):
    return nis(r[:2], S[:2, :2])


def resolvable(r_true, S, power=0.95, alpha=ALPHA):
    """Detection power of the chi2_3 test for a true residual r_true (non-central chi2 with
    lambda = r^T S^-1 r): True when P(chi2_3(lambda) > chi2_3(alpha)) >= power."""
    lam = nis(r_true, S)
    return bool(_ncx2.sf(chi2(3, alpha), 3, lam) >= power), lam


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


def residual_cov(Ja, Ra, Jb, Pb, Jx=None, X=None):
    """The one covariance formula: S = J_a R_a J_a^T + J_b P_b J_b^T (+ J_xi X J_xi^T)."""
    S = Ja @ Ra @ Ja.T + Jb @ Pb @ Jb.T
    if Jx is not None and X is not None:
        S = S + Jx @ X @ Jx.T
    return 0.5 * (S + S.T)


# ------------------------------------------------------------------ sensor: segment measurement noise


class SensorNoise:
    """sigma_z(z) = a + b z^2 (E1c) and the systematic floors sigma_theta0, sigma_d0 (calibrated once, locked)."""

    def __init__(self, a=0.0, b=1.425e-3, sigma_theta0_deg=0.5, sigma_d0_m=0.005):
        self.a, self.b = float(a), float(b)
        self.th0 = float(np.deg2rad(sigma_theta0_deg or 0.0))
        self.d0 = float(sigma_d0_m or 0.0)

    def sigma_z(self, z):
        return self.a + self.b * np.square(z)

    def R_old(self, N_s, lam1, lam2, zbar, nv):
        """v1/v2 formula (centroid depth, unweighted cell spread); kept for T2 comparison only."""
        s2 = float(self.sigma_z(zbar)) ** 2 * float(nv) ** 2
        return np.diag([s2 / (N_s * lam1) + self.th0 ** 2, s2 / (N_s * lam2) + self.th0 ** 2, s2 / N_s + self.d0 ** 2])


def cell_rows(mu, n, p, B):
    """h_c for every cell centre: offset of the cell w.r.t. the plane perturbed by x = [dth1, dth2, dd] in (B, p),
    same (+) convention as J_a: n' = normalize(n + B dth), p' = p + dd n. Signed distance of the fixed cell centre to
    the perturbed plane is n'.(mu - p') = u^T dth - dd (first order), u = B^T (mu - p)."""
    u = (mu - p) @ B
    return np.concatenate([u, -np.ones((len(mu), 1))], 1)


def segment_cov(mu, n, p, B, sensor):
    """R_s (3x3 in (B, p)) from per-cell measurements (PLAN core §3.1): each cell centre mu_c is one measurement of
    the plane with sigma_c = sigma_z(mu_c,z) |n^T v_c|, v_c = mu_c / mu_c,z; I = sum h_c h_c^T / sigma_c^2;
    R_s = I^-1 + diag(sigma_theta0^2, sigma_theta0^2, sigma_d0^2)."""
    mu = np.atleast_2d(mu)
    H = cell_rows(mu, n, p, B)
    v = mu / mu[:, 2:3]
    sig = sensor.sigma_z(mu[:, 2]) * np.abs(v @ n)
    sig = np.maximum(sig, 1e-9)
    I = (H / sig[:, None] ** 2).T @ H
    w, V = np.linalg.eigh(0.5 * (I + I.T))
    w = np.maximum(w, 1e-12 * max(float(w.max()), 1e-30))
    Rinv = (V / w) @ V.T
    return 0.5 * (Rinv + Rinv.T) + np.diag([sensor.th0 ** 2, sensor.th0 ** 2, sensor.d0 ** 2])


# ------------------------------------------------------------------ measured: anchor-map offset, prediction noise


class OffsetEstimator:
    """Measured anchor-map offset b_hat (PLAN core §3.2) in the anchor's (B_j, o_j) coordinates. Stationary: no
    growth model. Samples z = -J_b^-1 r(s | original anchor) at T_opt with R_z = J_b^-1 J_a R_s J_a^T J_b^-T are
    accepted when (z - b)^T (R_z + V)^-1 (z - b) <= chi2_3. b_hat, V from the accepted samples of the last W_b frames:
    b = (sum R_z^-1)^-1 sum R_z^-1 z, V = f (sum R_z^-1)^-1, f = max(1, chi / (3 (n - 1))) for n >= 3."""

    def __init__(self, P, k0, window=30):
        self.b = np.zeros(3)
        self.V = np.array(P, dtype=np.float64)
        self.tau = int(k0)
        self.window = int(window)
        self.buf = deque()
        self.f = 1.0
        self.n_accept = 0
        self.n_reject = 0

    @property
    def n(self):
        return len(self.buf)

    def test(self, z, Rz):
        d = np.asarray(z) - self.b
        return float(d @ np.linalg.solve(Rz + self.V, d))

    def sample(self, z, Rz, k):
        """Returns (accepted, mahalanobis)."""
        m = self.test(z, Rz)
        if m > CHI2_3:
            self.n_reject += 1
            return False, m
        self.buf.append((int(k), np.asarray(z, dtype=np.float64), np.asarray(Rz, dtype=np.float64)))
        while self.buf and self.buf[0][0] <= int(k) - self.window:
            self.buf.popleft()
        Ws = [np.linalg.inv(R) for _, _, R in self.buf]
        Vm = np.linalg.inv(sum(Ws))
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
        return True, m

    def y(self, n, o, B, oplus):
        """Offset-corrected anchor y_j = x_j (+) b_hat (decisions only)."""
        return oplus(n, o, B, self.b)


class PredNoise:
    """Prediction noise of the constant-velocity gate: eps_k = Log(T_opt,k T_gate,k^-1); Q_pred = mean eps eps^T over
    the last W frames (+1e-10 I); first frame diag((1 cm)^2 I3, (0.5 deg)^2 I3)."""

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
