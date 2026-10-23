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


class DriftModel:
    """(b) absolute pose error e(t) in xi = [rho, theta] coordinates; isotropic, path-length driven."""

    def __init__(self, form="linear", kappa_t=0.0, kappa_r=0.0, kappa_t_rw=0.0, kappa_r_rw=0.0):
        assert form in ("linear", "rw", "mixed")
        self.form = form
        self.kt, self.kr = float(kappa_t or 0.0), float(kappa_r or 0.0)
        self.kt_rw, self.kr_rw = float(kappa_t_rw or 0.0), float(kappa_r_rw or 0.0)
        if form == "rw":           # rw alone: the rw coefficients are kappa_t, kappa_r
            self.kt_rw, self.kr_rw, self.kt, self.kr = self.kt, self.kr, 0.0, 0.0
        elif form == "linear":
            self.kt_rw = self.kr_rw = 0.0

    @staticmethod
    def _blk(vt, vr):
        return np.diag([vt] * 3 + [vr] * 3)

    def C(self, sa, pa, sb, pb):
        """Cov(e(t_a) - e(t_b))."""
        ds, dp = abs(sb - sa), abs(pb - pa)
        return self._blk((self.kt * ds) ** 2 + self.kt_rw ** 2 * ds, (self.kr * dp) ** 2 + self.kr_rw ** 2 * dp)

    def cov_e(self, sa, pa, sb, pb):
        """Cov(e(t_a), e(t_b))."""
        return self._blk(self.kt ** 2 * sa * sb + self.kt_rw ** 2 * min(sa, sb),
                         self.kr ** 2 * pa * pb + self.kr_rw ** 2 * min(pa, pb))

    def cov_eps(self, sk, pk, si, pi_, sl, pl):
        """Cov(eps_i, eps_l), eps_j = e(k) - e(t_j)."""
        return (self.cov_e(sk, pk, sk, pk) - self.cov_e(sk, pk, sl, pl) - self.cov_e(si, pi_, sk, pk)
                + self.cov_e(si, pi_, sl, pl))


class PredNoise:
    """(c) prediction error: Q_pred = diag(s_rho^2 I3, s_theta^2 I3) from the last W xi_k = Log(T_opt T_pred^-1)."""

    def __init__(self, window=30, sigma_rho_m=0.01, sigma_theta_deg=0.5):
        self.buf = deque(maxlen=int(window))
        self.s0 = (float(sigma_rho_m), float(np.deg2rad(sigma_theta_deg)))

    def push(self, xi):
        self.buf.append(np.asarray(xi, dtype=np.float64))

    def sigmas(self):
        if not self.buf:
            return self.s0
        X = np.stack(self.buf)
        return (float(np.sqrt(np.mean(np.sum(X[:, :3] ** 2, 1)) / 3)),
                float(np.sqrt(np.mean(np.sum(X[:, 3:] ** 2, 1)) / 3)))

    def Q(self):
        sr, st = self.sigmas()
        return np.diag([sr ** 2] * 3 + [st ** 2] * 3)
