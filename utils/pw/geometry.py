"""PLAN A2G-P1 §2, §3.7, §3.9: plane geometry, SE(3) helpers, support grid, visibility.

Conventions (PLAN §1): T_cw world->camera, x_c = R x_w + t; plane n.x + d = 0 (d = -n.o); left pose update
T <- Exp(xi) T with xi = [rho (translation), theta (rotation)], same Exp as gaussian_splatting SE3_exp.
Everything is float64 numpy on the CPU. This module never reads ground truth (I12).
"""
import numpy as np

# ------------------------------------------------------------------ SO(3) / SE(3)


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(theta):
    W = skew(theta)
    W2 = W @ W
    a = float(np.linalg.norm(theta))
    if a < 1e-5:
        return np.eye(3) + W + 0.5 * W2
    return np.eye(3) + (np.sin(a) / a) * W + ((1 - np.cos(a)) / a ** 2) * W2


def _V(theta):
    W = skew(theta)
    W2 = W @ W
    a = float(np.linalg.norm(theta))
    if a < 1e-5:
        return np.eye(3) + 0.5 * W + (1.0 / 6.0) * W2
    return np.eye(3) + W * ((1.0 - np.cos(a)) / a ** 2) + W2 * ((a - np.sin(a)) / a ** 3)


def se3_exp(xi):
    xi = np.asarray(xi, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = so3_exp(xi[3:])
    T[:3, 3] = _V(xi[3:]) @ xi[:3]
    return T


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    a = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a < 1e-7:
        return 0.5 * w
    if np.pi - a < 1e-6:                      # near pi: axis from the symmetric part
        B = (R + np.eye(3)) / 2
        ax = np.sqrt(np.clip(np.diag(B), 0, None))
        k = int(np.argmax(ax))
        ax = B[:, k] / np.sqrt(B[k, k])
        return a * ax / np.linalg.norm(ax)
    return a / (2 * np.sin(a)) * w


def se3_log(T):
    th = so3_log(T[:3, :3])
    rho = np.linalg.solve(_V(th), T[:3, 3])
    return np.concatenate([rho, th])


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def centre(T):
    return -T[:3, :3].T @ T[:3, 3]


def rot_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


# ------------------------------------------------------------------ plane parametrisation


def basis(n):
    """B(n): 3x2, orthonormal, perpendicular to n; deterministic (cross with the least parallel axis)."""
    n = np.asarray(n, dtype=np.float64)
    e = np.zeros(3)
    e[int(np.argmin(np.abs(n)))] = 1.0
    b1 = np.cross(n, e)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1)
    return np.stack([b1, b2], 1)


def gram_schmidt(B, n):
    b1 = B[:, 0] - (B[:, 0] @ n) * n
    b1 /= np.linalg.norm(b1)
    b2 = B[:, 1] - (B[:, 1] @ n) * n - (B[:, 1] @ b1) * b1
    b2 /= np.linalg.norm(b2)
    return np.stack([b1, b2], 1)


def oplus(n, o, B, x):
    """x = [dtheta1, dtheta2, dd]: n <- normalize(n + B dtheta); o <- o + dd n_old; B transported."""
    x = np.asarray(x, dtype=np.float64)
    n_new = n + B @ x[:2]
    n_new = n_new / np.linalg.norm(n_new)
    o_new = o + x[2] * n
    B_new = gram_schmidt(B - np.outer(n, x[:2]), n_new)
    return n_new, o_new, B_new


def nd_from_no(n, o):
    return np.asarray(n, dtype=np.float64), -float(np.dot(n, o))


def no_from_nd(n, d):
    n = np.asarray(n, dtype=np.float64)
    return n, -float(d) * n


def to_camera(T, n_w, o_w, B_w):
    R, t = T[:3, :3], T[:3, 3]
    return R @ n_w, R @ o_w + t, R @ B_w


def to_world(T, n_c, p_c, B_c):
    R, t = T[:3, :3], T[:3, 3]
    return R.T @ n_c, R.T @ (p_c - t), R.T @ B_c


def facing(n_c, o_c):
    """Anchor faces the camera: -n_c . o_c > 0 (d_c > 0)."""
    return -float(n_c @ o_c) > 0.0


# ------------------------------------------------------------------ support grid (§3.9)


def grid_coords(X_w, g_o, g_B, h):
    """continuous 2D grid coordinates (in cells) of world points (N,3)."""
    return (np.atleast_2d(X_w) - g_o) @ g_B / h


def grid_index(X_w, g_o, g_B, h):
    return np.floor(grid_coords(X_w, g_o, g_B, h)).astype(np.int64)


def grid_centres(cells, g_o, g_B, h):
    """world centres of grid cells (iterable of (i, j))."""
    c = np.asarray(sorted(cells), dtype=np.float64).reshape(-1, 2)
    return g_o + ((c + 0.5) * h) @ g_B.T


def cells_from_segment(cells_ij, cs, K, n_c, d_c, T, g_o, g_B, h):
    """Grid cells covered by the CAPE cells of a segment: 4 pixel corners of each cell -> ray/plane intersection
    on the segment plane (camera) -> world -> grid; every grid cell inside the bounding box of the quad."""
    cells_ij = np.asarray(cells_ij, dtype=np.int64).reshape(-1, 2)
    if cells_ij.shape[0] == 0:
        return set()
    Kinv = np.linalg.inv(K)
    i, j = cells_ij[:, 0], cells_ij[:, 1]
    corners = []
    for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1)):
        u = (j + dj) * cs
        v = (i + di) * cs
        corners.append(np.stack([u, v, np.ones_like(u)], 1).astype(np.float64))
    out = set()
    R, t = T[:3, :3], T[:3, 3]
    rays = np.stack([c @ Kinv.T for c in corners], 1)                 # (N, 4, 3)
    den = rays @ n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -d_c / den
    ok = np.all(np.isfinite(lam) & (lam > 0), axis=1)
    X_c = rays * lam[..., None]
    X_w = (X_c - t) @ R                                                 # R^T (x - t)
    g = (X_w - g_o) @ g_B / h                                          # (N, 4, 2)
    lo = np.floor(g.min(1)).astype(np.int64)
    hi = np.floor(g.max(1)).astype(np.int64)
    for k in np.nonzero(ok)[0]:
        for a in range(lo[k, 0], hi[k, 0] + 1):
            for b in range(lo[k, 1], hi[k, 1] + 1):
                out.add((int(a), int(b)))
    return out


def overlap_or_adjacent(idx, G):
    """idx (N,2) grid indices: any equal to or 8-adjacent to a cell of G."""
    if not G or len(idx) == 0:
        return False
    for a, b in np.asarray(idx).reshape(-1, 2).tolist():
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                if (a + da, b + db) in G:
                    return True
    return False


def n_visible_cells(cells, g_o, g_B, h, T, K, depth, sigma_z, sig_rho, H, W):
    """§3.7: grid-cell centres projected with T (z > 0, inside the image); a cell is seen when the measured depth
    z_m > 0 and z_m >= z_proj - 3 sigma_z(z_proj) - 3 sig_rho (nothing closer occludes it)."""
    if not cells:
        return 0
    X = grid_centres(cells, g_o, g_B, h)
    Xc = X @ T[:3, :3].T + T[:3, 3]
    z = Xc[:, 2]
    m = z > 1e-6
    if not m.any():
        return 0
    Xc, z = Xc[m], z[m]
    u = np.floor(K[0, 0] * Xc[:, 0] / z + K[0, 2]).astype(np.int64)
    v = np.floor(K[1, 1] * Xc[:, 1] / z + K[1, 2]).astype(np.int64)
    ins = (u >= 0) & (v >= 0) & (u < W) & (v < H)
    if not ins.any():
        return 0
    zm = depth[v[ins], u[ins]]
    zp = z[ins]
    seen = (zm > 0) & (zm >= zp - 3 * sigma_z(zp) - 3 * sig_rho)
    return int(seen.sum())
