"""PLAN E1 §6 unit tests T1–T6 on synthetic data (no dataset, CPU only)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gaussian_splatting"))

from utils.plane_anchor import (  # noqa: E402
    Anchor,
    PlaneAnchorMap,
    PlaneExtractor,
    PlaneMeas,
    plane_c2w,
    plane_w2c,
)
from gaussian_splatting.utils.camera_utils import SE3_exp  # noqa: E402

H, W = 480, 640
K = np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])
CFG = dict(
    lambda_plane=1.0, sigma_normal=0.02, sigma_dist=0.01, cell_size=20, c_noise=3.0,
    merge_angle_deg=10, merge_dist_k=3.0, min_plane_pixels=3200, max_plane_depth=4.0,
    match_max_angle_deg=10, match_max_dist_m=0.10, match_min_overlap=0.3,
    min_singular_value=0.3, solve_max_trans_m=0.20, solve_max_rot_deg=15, activate_after_kf=2,
)
ROT_LR, TRANS_LR = 0.003, 0.0015      # configs/tum/base_config.yaml Training.lr


def rot(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.deg2rad(deg)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rand_rot(rng, max_deg):
    return rot(rng.normal(size=3), rng.uniform(-max_deg, max_deg))


def raycast(planes_w, T_cw, noise=0.0, rng=None):
    """Depth (z) of the nearest plane hit per pixel; 0 where no plane is hit in front."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1)
    z = np.full((H, W), np.inf)
    for n_w, d_w in planes_w:
        n_c, d_c = plane_w2c(n_w, d_w, T_cw)
        den = r @ n_c
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -d_c / den
        s[(den >= 0) | (s <= 0)] = np.inf
        z = np.minimum(z, s)
    z[~np.isfinite(z)] = 0.0
    if noise > 0:
        z = np.where(z > 0, z + rng.normal(0, noise, z.shape), 0.0)
    return torch.from_numpy(z)


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))


def rot_err_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))


def centre(T):
    return -T[:3, :3].T @ T[:3, 3]


def meas(n, d, n_px=10000):
    return PlaneMeas(n=torch.as_tensor(n, dtype=torch.float64), d=float(d),
                     mask_cells=torch.zeros(H // 20, W // 20, dtype=torch.bool), n_px=int(n_px), mse=0.0)


def anchor_map_with(anchors_w):
    m = PlaneAnchorMap(CFG, K, H, W)
    for i, (n_w, d_w) in enumerate(anchors_w):
        m.anchors[i] = Anchor(id=i, n_w=torch.as_tensor(n_w, dtype=torch.float64), d_w=float(d_w),
                              samples_w=torch.zeros(0, 3, dtype=torch.float64), n_obs_kf=2, active=True,
                              created_kf=0, weight=1.0)
    m.next_id = len(anchors_w)
    return m


def random_scene(rng, k, T_gt, min_sv=0.5):
    """k anchor planes, well conditioned (σ_min of measured normals >= min_sv), d_c > 0 under T_gt."""
    while True:
        out = []
        for _ in range(k):
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
            d = rng.uniform(1.0, 3.0)
            n_c, d_c = plane_w2c(n, d, T_gt)
            if d_c < 0.3 and d_c > -0.3:
                break
            if d_c < 0:
                n, d = -n, -d
            out.append((n, d))
        if len(out) < k:
            continue
        M = np.stack([plane_w2c(n, d, T_gt)[0] for n, d in out])
        if np.linalg.svd(M, compute_uv=False)[2] >= min_sv:
            return out


# ---------------------------------------------------------------------------- T1
def test_T1_extractor_room_corner():
    rng = np.random.default_rng(1)
    Ry = rot([0, 1, 0], 8.0)
    gt = [(Ry @ np.array([0.0, 0.0, -1.0]), 3.0),   # back wall, 3 m
          (Ry @ np.array([1.0, 0.0, 0.0]), 1.2),    # left wall
          (Ry @ np.array([0.0, -1.0, 0.0]), 0.8)]   # floor
    depth = raycast(gt, np.eye(4), noise=0.002, rng=rng)
    zs = depth[depth > 0]
    assert 1.5 <= float(zs.min()) and float(zs.max()) <= 3.6
    planes = PlaneExtractor(CFG)(depth, K)
    print(f"T1: {len(planes)} planes, n_px={[p.n_px for p in planes]}")
    assert len(planes) == 3
    for n_g, d_g in gt:
        p = max(planes, key=lambda q: float(q.n.numpy() @ n_g))
        e_n, e_d = angle_deg(p.n.numpy(), n_g), abs(p.d - d_g)
        print(f"T1: normal err {e_n:.3f} deg, d err {e_d * 100:.3f} cm")
        assert e_n < 1.0 and e_d < 0.01


# ---------------------------------------------------------------------------- T2
def test_T2_plane_transform_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(20):
        T = make_T(rand_rot(rng, 180), rng.uniform(-2, 2, 3))
        n_w = rng.normal(size=3)
        n_w /= np.linalg.norm(n_w)
        d_w = rng.uniform(-3, 3)
        # three points on the world plane
        a = np.cross(n_w, [1.0, 0, 0] if abs(n_w[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n_w, a)
        X_w = np.stack([-d_w * n_w + s * a + q * b for s, q in rng.uniform(-2, 2, (3, 2))])
        X_c = X_w @ T[:3, :3].T + T[:3, 3]
        n_c, d_c = plane_w2c(n_w, d_w, T)
        assert np.max(np.abs(X_c @ n_c + d_c)) < 1e-6
        assert abs(np.linalg.norm(n_c) - 1) < 1e-9
        n_b, d_b = plane_c2w(n_c, d_c, T)
        assert np.max(np.abs(X_w @ n_b + d_b)) < 1e-6
        assert np.allclose(n_b, n_w, atol=1e-9) and abs(d_b - d_w) < 1e-9


# ---------------------------------------------------------------------------- T3
def _solve_trial(rng, k, noise):
    T_gt = make_T(rand_rot(rng, 30), rng.uniform(-0.5, 0.5, 3))
    scene = random_scene(rng, k, T_gt)
    m = anchor_map_with(scene)
    planes = []
    for n_w, d_w in scene:
        n_c, d_c = plane_w2c(n_w, d_w, T_gt)
        if noise:
            ax = np.cross(n_c, rng.normal(size=3))
            n_c = rot(ax, 0.5) @ n_c
            d_c = d_c + rng.normal(0, 0.005)
        planes.append(meas(n_c, d_c, rng.integers(5000, 20000)))
    pairs = [(i, i) for i in range(k)]
    T_prior = make_T(rot(rng.normal(size=3), 1.0) @ T_gt[:3, :3], T_gt[:3, 3] + rng.normal(0, 0.01, 3))
    T = m.solve_pose(planes, pairs, torch.from_numpy(T_prior))
    assert T is not None, m.last_solve_status
    T = T.numpy()
    return rot_err_deg(T[:3, :3], T_gt[:3, :3]), np.linalg.norm(centre(T) - centre(T_gt))


def test_T3_solve_pose():
    rng = np.random.default_rng(3)
    clean, noisy = [], []
    for trial in range(20):
        k = 3 + trial % 3
        clean.append(_solve_trial(rng, k, noise=False))
        noisy.append(_solve_trial(rng, k, noise=True))
    clean, noisy = np.array(clean), np.array(noisy)
    n_bad = int(((noisy[:, 0] >= 1.0) | (noisy[:, 1] >= 0.01)).sum())
    print(f"T3: clean worst {clean[:, 0].max():.2e} deg / {clean[:, 1].max():.2e} m; "
          f"noisy rot median {np.median(noisy[:, 0]):.3f} max {noisy[:, 0].max():.3f} deg, "
          f"centre median {100 * np.median(noisy[:, 1]):.3f} max {100 * noisy[:, 1].max():.3f} cm, "
          f"trials over limit {n_bad}/20")
    assert np.deg2rad(clean[:, 0].max()) < 1e-5 and clean[:, 1].max() < 1e-5
    assert n_bad == 0
    # only two planes -> None
    T_gt = make_T(np.eye(3), np.zeros(3))
    scene = [(np.array([0.0, -1.0, 0.0]), 1.0), (np.array([1.0, 0.0, 0.0]), 1.5)]
    m = anchor_map_with(scene)
    planes = [meas(*plane_w2c(n, d, T_gt)) for n, d in scene]
    assert m.solve_pose(planes, [(0, 0), (1, 1)], torch.from_numpy(T_gt)) is None


# ---------------------------------------------------------------------------- T4
def test_T4_gradcheck_plane_loss():
    rng = np.random.default_rng(4)
    T_gt = make_T(rand_rot(rng, 20), rng.uniform(-0.5, 0.5, 3))
    scene = random_scene(rng, 4, T_gt)
    m = anchor_map_with(scene)
    planes = [meas(*plane_w2c(n, d, T_gt), rng.integers(5000, 20000)) for n, d in scene]
    pairs = [(i, i) for i in range(4)]
    T0 = torch.from_numpy(make_T(rot([1, 2, 3], 2.0) @ T_gt[:3, :3], T_gt[:3, 3] + 0.03))
    tau = torch.tensor(rng.normal(0, 1e-2, 6), dtype=torch.float64, requires_grad=True)

    def f(x):
        return m.plane_loss(SE3_exp(x) @ T0, planes, pairs)

    assert torch.autograd.gradcheck(f, (tau,), eps=1e-6, atol=1e-5, rtol=1e-4)
    g = torch.autograd.grad(f(tau), tau)[0]
    assert torch.all(torch.isfinite(g)) and g[:3].abs().sum() > 0 and g[3:].abs().sum() > 0


# ---------------------------------------------------------------------------- T5
class _Cam:
    """Minimal stand-in for Camera: same parameters and update_pose rule as the repo."""

    def __init__(self, T):
        self.T = T.clone()
        self.cam_rot_delta = torch.nn.Parameter(torch.zeros(3, dtype=T.dtype))
        self.cam_trans_delta = torch.nn.Parameter(torch.zeros(3, dtype=T.dtype))

    def update_pose(self, converged_threshold=1e-4):
        tau = torch.cat([self.cam_trans_delta, self.cam_rot_delta], axis=0)
        new_w2c = SE3_exp(tau) @ self.T
        converged = (tau ** 2).sum() < (converged_threshold ** 2)
        self.T = new_w2c.detach().clone()
        self.cam_rot_delta.data.fill_(0)
        self.cam_trans_delta.data.fill_(0)
        return converged


def test_T5_plane_loss_only_converges():
    rng = np.random.default_rng(5)
    T_gt = make_T(rot([0, 1, 0], 10.0), np.array([0.1, -0.05, 0.2]))
    scene = [(np.array([0.0, -1.0, 0.0]), 1.0),                       # floor
             (rot([0, 1, 0], 20.0) @ np.array([1.0, 0.0, 0.0]), 1.5),  # wall
             (rot([0, 1, 0], -15.0) @ np.array([0.0, 0.0, -1.0]), 3.0)]  # wall
    m = anchor_map_with(scene)
    planes = [meas(*plane_w2c(n, d, T_gt), 10000) for n, d in scene]
    pairs = [(i, i) for i in range(3)]
    ax = rng.normal(size=3)
    dirn = rng.normal(size=3)
    T_init = make_T(rot(ax, 3.0) @ T_gt[:3, :3], T_gt[:3, 3] + 0.05 * dirn / np.linalg.norm(dirn))
    cam = _Cam(torch.from_numpy(T_init).float())
    opt = torch.optim.AdamW([{"params": [cam.cam_rot_delta], "lr": ROT_LR},
                             {"params": [cam.cam_trans_delta], "lr": TRANS_LR}], weight_decay=0.01)
    ctx = m.make_loss_ctx(planes, pairs, "cpu", torch.float32)
    iters = 0
    for i in range(120):
        tau = torch.cat([cam.cam_trans_delta, cam.cam_rot_delta], 0)
        loss = m.plane_loss_ctx(SE3_exp(tau) @ cam.T, ctx)
        loss.backward()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            iters = i + 1
            if cam.update_pose():
                break
    T = cam.T.double().numpy()
    er, et = rot_err_deg(T[:3, :3], T_gt[:3, :3]), np.linalg.norm(centre(T) - centre(T_gt))
    print(f"T5: iters {iters}, rot err {er:.4f} deg, centre err {et * 1000:.3f} mm, final loss {float(loss):.3e}")
    assert et < 1e-3 and er < 0.1


# ---------------------------------------------------------------------------- T6
def test_T6_corridor_match_and_update():
    rng = np.random.default_rng(6)
    world = [(np.array([0.0, -1.0, 0.0]), 1.0),    # floor   y = 1
             (np.array([0.0, 1.0, 0.0]), 1.2),     # ceiling y = -1.2
             (np.array([1.0, 0.0, 0.0]), 1.0),     # left    x = -1
             (np.array([-1.0, 0.0, 0.0]), 1.1),    # right   x = 1.1
             (np.array([0.0, 0.0, -1.0]), 6.5)]    # end wall z = 6.5 (in range only later)
    ext = PlaneExtractor(CFG)
    m = PlaneAnchorMap(CFG, K, H, W)
    seen_ids = {}           # world plane index -> anchor id
    created_at = {}
    for k in range(13):
        T_cw = make_T(rot([0, 1, 0], 3.0 * np.sin(k)), np.zeros(3))
        T_cw[:3, 3] = -T_cw[:3, :3] @ np.array([0.0, 0.0, 0.25 * k])      # camera centre moves along +z
        depth = raycast(world, T_cw, noise=0.001, rng=rng)
        planes = ext(depth, K)
        pairs = m.match(planes, torch.from_numpy(T_cw))
        n_before = set(m.anchors)
        # which world plane each measurement is (a measurement matching none of them is spurious)
        for pi, p in enumerate(planes):
            n_w, d_w = plane_c2w(p.n.numpy(), p.d, T_cw)
            wi = int(np.argmax([n_w @ n for n, _ in world]))
            assert n_w @ world[wi][0] > 0.99 and abs(d_w - world[wi][1]) < 0.05, f"kf {k}: spurious plane {n_w} {d_w}"
            aid = dict(pairs).get(pi)
            if wi in seen_ids:
                assert aid == seen_ids[wi], f"kf {k}: world plane {wi} matched {aid}, expected {seen_ids[wi]}"
        m.update_at_keyframe(k, planes, pairs, torch.from_numpy(T_cw))
        for pi, p in enumerate(planes):
            n_w, _ = plane_c2w(p.n.numpy(), p.d, T_cw)
            wi = int(np.argmax([n_w @ n for n, _ in world]))
            if wi not in seen_ids:
                new = [a for a in m.anchors if a not in n_before and float(m.anchors[a].n_w.numpy() @ world[wi][0]) > 0.99]
                assert len(new) == 1
                seen_ids[wi] = new[0]
                created_at[wi] = k
        # activation rule
        for wi, aid in seen_ids.items():
            a = m.anchors[aid]
            if created_at[wi] == 0:
                assert a.active
            else:
                assert a.active == (k > created_at[wi]), (k, wi, a.n_obs_kf)
    print(f"T6: anchors {len(m.anchors)}, world->anchor {seen_ids}, created {created_at}")
    assert len(m.anchors) == len(seen_ids)                    # no duplicate anchors
    assert set(seen_ids) == set(range(5)) and created_at[4] > 0
    for wi, aid in seen_ids.items():
        a = m.anchors[aid]
        assert angle_deg(a.n_w.numpy(), world[wi][0]) < 1.0 and abs(a.d_w - world[wi][1]) < 0.01
