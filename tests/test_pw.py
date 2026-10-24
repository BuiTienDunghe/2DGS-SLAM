"""PLAN A2G-P1 v2 §5: tests Q1, Q2, Q4–Q7, Q11, Q12 (CPU unless stated). python tests/run_tests.py tests/test_pw.py <filter>"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy.stats import kstest, chi2 as sp_chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gaussian_splatting"))

from gaussian_splatting.utils.camera_utils import SE3_exp  # noqa: E402
from utils.plane_anchor import PlaneExtractor, PlaneMeas  # noqa: E402
from utils.pw import geometry as G  # noqa: E402
from utils.pw import model as Mo  # noqa: E402
from utils.pw.association import (ORIG, ambiguity, build_jc, ic_and_assign, init_from_jc, jc_stats,  # noqa: E402
                                  joint_compat, pair_S, pair_terms)
from utils.pw.segments import Seg, build_segments, fit_cells, merge_in_frame, seg_seg_nis  # noqa: E402

H, W = 480, 640
K = np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])
CS = 20
PT_CFG = dict(cell_size=20, c_noise=3.0, merge_angle_deg=10, merge_dist_k=3.0, min_plane_pixels=3200,
              max_plane_depth=4.0)
SENSOR = Mo.SensorNoise(0.0, 1.425e-3, 0.0, 0.0)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rand_unit(rng):
    return unit(rng.normal(size=3))


def rot(axis, deg):
    return G.so3_exp(unit(axis) * np.deg2rad(deg))


@dataclass
class A:                                 # minimal anchor (same attribute names as AnchorP1)
    id: int
    n: np.ndarray
    o: np.ndarray
    B: np.ndarray
    P: np.ndarray
    state: str = "active"
    conflict_with: Optional[int] = None
    primary_id: int = -1
    grid_o: np.ndarray = None
    grid_B: np.ndarray = None
    cells: set = field(default_factory=set)
    est: object = None


def mk_seg(n, p, R, B=None, mu=None):
    n = unit(n)
    B = G.basis(n) if B is None else B
    mu = np.atleast_2d(p) if mu is None else mu
    return Seg(cells_ij=np.zeros((1, 2), int), cnt=np.ones(1), s1=np.zeros((1, 3)), s2=np.zeros((1, 3, 3)),
               n=n, p=np.asarray(p, float), B=B, R=R, N_s=1, src=[0],
               meas=PlaneMeas(n=torch.from_numpy(n.copy()), d=-float(n @ p), mask_cells=None, n_px=1000, mse=0.0),
               mu=mu)


def perturb(n, p, B, x):
    return G.oplus(n, p, B, x)


# ============================================================================ Q1


def _fd(f, x0, h=1e-6):
    cols = []
    for i in range(len(x0)):
        e = np.zeros(len(x0))
        e[i] = h
        cols.append((f(x0 + e) - f(x0 - e)) / (2 * h))
    return np.stack(cols, 1)


def _rel(Jn, Ja):
    return np.linalg.norm(Jn - Ja) / max(np.linalg.norm(Ja), 1.0)


def test_Q1_jacobians():
    rng = np.random.default_rng(11)
    worst = 0.0
    for it in range(500):
        n_b = rand_unit(rng)
        n_a = unit(n_b + rng.normal(size=3) * 0.04) if it % 2 == 0 else rand_unit(rng)
        if n_a @ n_b < 0 and it % 2 == 1:
            n_a = -n_a
        B_a, B_b = G.basis(n_a), G.basis(n_b)
        p_a, o_b = rng.normal(size=3), rng.normal(size=3)
        # J_a
        fa = lambda x: Mo.residual(*perturb(n_a, p_a, B_a, x)[:2], n_b, o_b, B_b)
        worst = max(worst, _rel(_fd(fa, np.zeros(3)), Mo.jac_a(n_a, B_a, n_b, B_b)))
        # J_b
        def fb(x):
            nb, ob, Bb = perturb(n_b, o_b, B_b, x)
            return Mo.residual(n_a, p_a, nb, ob, Bb)
        worst = max(worst, _rel(_fd(fb, np.zeros(3)), Mo.jac_b(n_a, p_a, n_b, o_b, B_b)))
        # J_xi with the repo's SE3_exp
        Rw = G.so3_exp(rng.normal(size=3))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rw, rng.normal(size=3)
        n_j, o_j, B_j = n_b, o_b, B_b
        def fx(xi):
            Tp = SE3_exp(torch.from_numpy(xi)).numpy() @ T
            nc, oc, Bc = G.to_camera(Tp, n_j, o_j, B_j)
            return Mo.residual(n_a, p_a, nc, oc, Bc)
        nc, oc, Bc = G.to_camera(T, n_j, o_j, B_j)
        worst = max(worst, _rel(_fd(fx, np.zeros(6)), Mo.jac_xi(n_a, p_a, nc, Bc)))
        # A (shift of the reference point)
        o2 = o_b + B_b @ rng.normal(size=2)
        l2 = B_b.T @ (o2 - o_b)
        def fs(x):
            nn, oo, _ = perturb(n_b, o_b, B_b, x)
            th = B_b.T @ nn / (n_b @ nn)
            dd = float(nn @ (oo - o2)) / float(nn @ n_b)
            return np.concatenate([th, [dd]])
        worst = max(worst, _rel(_fd(fs, np.zeros(3)), Mo.shift_matrix(l2)))
        # numpy se3_exp == repo SE3_exp
        xi = rng.normal(size=6) * 0.3
        assert np.allclose(G.se3_exp(xi), SE3_exp(torch.from_numpy(xi)).numpy(), atol=1e-12)
        assert np.allclose(G.se3_log(G.se3_exp(xi)), xi, atol=1e-9)
    print(f"Q1: worst relative error {worst:.2e}")
    assert worst < 1e-6
    # z = -J_b^-1 r: if the map plane is the anchor moved by dx, the apparent offset z equals dx (first order)
    worst_z = 0.0
    for it in range(500):
        n_b = rand_unit(rng)
        B_b, o_b = G.basis(n_b), rng.normal(size=3)
        dx = rng.normal(size=3) * 1e-7
        n2, o2, B2 = G.oplus(n_b, o_b, B_b, dx)
        p_a = o2 + B2 @ rng.normal(size=2)                 # a point of the moved plane
        n_a = n2
        if it % 2:
            n_a = unit(n2 + rng.normal(size=3) * 1e-9)
        r = Mo.residual(n_a, p_a, n_b, o_b, B_b)
        z = -np.linalg.solve(Mo.jac_b(n_a, p_a, n_b, o_b, B_b), r)
        worst_z = max(worst_z, np.linalg.norm(z - dx) / np.linalg.norm(dx))
    print(f"Q1: z = -J_b^-1 r vs dx, worst relative error {worst_z:.2e}")
    assert worst_z < 1e-6


# ============================================================================ Q2

R_SEG = np.diag([np.deg2rad(0.3) ** 2, np.deg2rad(0.3) ** 2, 0.003 ** 2])
Q_GATE = np.diag([0.006 ** 2] * 3 + [np.deg2rad(0.16) ** 2] * 3)
GAMMA0 = np.zeros((3, 3))          # constant offset: no growth (Gamma is exercised in Q11)


def _noisy(n, o, B, Rm, rng):
    return G.oplus(n, o, B, rng.multivariate_normal(np.zeros(3), Rm))


def bias_sample(seg, a, T=np.eye(4)):
    """same formula as PlaneWorld._bias_sample: z = -J_b^-1 r(s | original anchor), R_z = J_b^-1 J_a R J_a^T J_b^-T"""
    r, Ja, Jb, _ = pair_terms(seg, ORIG(a), T)
    Jbi = np.linalg.inv(Jb)
    Rz = Jbi @ (Ja @ seg.R @ Ja.T) @ Jbi.T
    return -Jbi @ r, 0.5 * (Rz + Rz.T)


def _anchor_with_offset(n_t, o_t, rng, aid=0, n_samples=10):
    """anchor = true plane (+) a constant random offset b ~ N(0, P_j) (the prior of the offset estimator: V = P_j at
    creation); b_hat from n_samples segments of the true plane (camera = world). Returns the anchor (est filled)."""
    Bt = G.basis(n_t)
    an, ao, aB = _noisy(n_t, o_t, Bt, R_SEG, rng)
    a = A(aid, an, ao, aB, R_SEG.copy())
    a.est = Mo.OffsetEstimator(a.P, 0, 30)
    for k in range(1, n_samples + 1):
        sn, sp, sB = _noisy(n_t, o_t + Bt @ rng.normal(size=2) * 0.3, Bt, R_SEG, rng)
        z, Rz = bias_sample(mk_seg(sn, sp, R_SEG, sB), a)
        a.est.sample(z, Rz, k, GAMMA0)
    return a


def test_Q2_offset_montecarlo():
    rng = np.random.default_rng(22)
    N = 5000
    n_t, o_t = unit([0.1, -0.05, -1.0]), np.array([0.2, -0.1, 2.5])
    Bt = G.basis(n_t)
    ic, mg = [], []
    for i in range(N):
        a = _anchor_with_offset(n_t, o_t, rng)
        Tg = G.se3_exp(rng.multivariate_normal(np.zeros(6), Q_GATE))
        # a new segment of the true plane seen from the true camera (identity); the gate pose carries the error
        sn, sp, sB = _noisy(n_t, o_t + Bt @ rng.normal(size=2) * 0.3, Bt, R_SEG, rng)
        seg_w = mk_seg(sn, sp, R_SEG, sB)
        n_c, p_c, B_c = G.to_camera(Tg, seg_w.n, seg_w.p, seg_w.B)      # what the camera at Tg would measure ...
        seg = mk_seg(sn, sp, R_SEG, sB)                                 # ... is the true-camera measurement
        r, S, *_ = pair_S(seg, a.est.y(a.n, a.o, a.B, G.oplus), a.est.cov_at(11, GAMMA0), Tg, Q_GATE)
        ic.append(Mo.nis(r, S))
        # two true duplicates of the same plane, each with its own offset and samples
        a1 = _anchor_with_offset(n_t, o_t, rng, 1)
        a2 = _anchor_with_offset(n_t, o_t, rng, 2)
        y1, y2 = a1.est.y(a1.n, a1.o, a1.B, G.oplus), a2.est.y(a2.n, a2.o, a2.B, G.oplus)
        rr = Mo.residual(y2[0], y2[1], y1[0], y1[1], y1[2])
        Ja, Jb = Mo.jac_a(y2[0], y2[2], y1[0], y1[2]), Mo.jac_b(y2[0], y2[1], y1[0], y1[1], y1[2])
        mg.append(Mo.nis(rr, Ja @ a2.est.V @ Ja.T + Jb @ a1.est.V @ Jb.T))
    for name, v in (("IC (y_j, V, Q_pred)", np.array(ic)), ("merge of two true duplicates", np.array(mg))):
        m, rej = float(v.mean()), float(np.mean(v > Mo.CHI2_3))
        print(f"Q2 {name}: mean NIS {m:.3f} (dof 3), false reject {100 * rej:.2f} %")
        assert abs(m - 3) <= 0.15, (name, m)
        assert abs(rej - 0.01) <= 0.005, (name, rej)

# ============================================================================ Q4


def _orig_extractor_cls():
    import importlib.util
    import subprocess
    src = subprocess.run(["git", "-C", ROOT, "show", "c80ff00:utils/plane_anchor.py"], capture_output=True,
                         text=True, check=True).stdout
    path = "/tmp/plane_anchor_e1c_orig.py"
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("pa_orig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PlaneExtractor


def raycast(planes_w, T_cw, noise=0.0, rng=None):
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1)
    z = np.full((H, W), np.inf)
    for n_w, d_w in planes_w:
        n_c = T_cw[:3, :3] @ n_w
        d_c = d_w - n_c @ T_cw[:3, 3]
        den = r @ n_c
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -d_c / den
        s[(den >= 0) | (s <= 0)] = np.inf
        z = np.minimum(z, s)
    z[~np.isfinite(z)] = 0.0
    if noise > 0:
        z = np.where(z > 0, z + rng.normal(0, noise, z.shape), 0.0)
    return torch.from_numpy(z)


def _compare_extract(depth, Korig):
    old = Korig(PT_CFG)(depth, K)
    new = PlaneExtractor(PT_CFG)(depth, K)
    assert len(old) == len(new)
    for a, b in zip(old, new):
        assert torch.equal(a.n, b.n) and a.d == b.d and torch.equal(a.mask_cells, b.mask_cells)
        assert a.n_px == b.n_px and a.mse == b.mse and np.array_equal(a.pts_c, b.pts_c)
        c = b.cells
        assert int(c["cnt"].sum()) == b.n_px
        sg = fit_cells(np.asarray(c["ij"]), np.asarray(c["cnt"], float), c["s1"], c["s2"], SENSOR, [0], meas=b)
        assert np.allclose(sg.n, b.n.numpy(), atol=1e-10) and abs(sg.d - b.d) < 1e-10
    return len(new)


def test_Q4_extractor_unchanged():
    Korig = _orig_extractor_cls()
    rng = np.random.default_rng(4)
    Ry = rot([0, 1, 0], 8.0)
    gt = [(Ry @ np.array([0.0, 0.0, -1.0]), 3.0), (Ry @ np.array([1.0, 0.0, 0.0]), 1.2),
          (Ry @ np.array([0.0, -1.0, 0.0]), 0.8)]
    n = _compare_extract(raycast(gt, np.eye(4), 0.002, rng), Korig)
    print(f"Q4: synthetic corner {n} planes identical")
    seq = os.path.join(ROOT, "datasets/tum/rgbd_dataset_freiburg3_structure_notexture_far")
    if os.path.isdir(seq):
        import cv2
        files = sorted(os.listdir(os.path.join(seq, "depth")))[:20:2]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tot = 0
        for f in files:
            d = cv2.imread(os.path.join(seq, "depth", f), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
            tot += _compare_extract(torch.from_numpy(d).to(dev), Korig)
        print(f"Q4: {len(files)} real snt_far frames ({dev}), {tot} planes identical")


# ============================================================================ Q5


def _synth_cells(n_c, d_c, cells, rng, noise=0.002):
    """Per-cell sums of the pixels of the given CAPE cells hitting the plane n_c.x + d_c = 0."""
    ij, cnt, s1, s2, pts = [], [], [], [], []
    for (ci, cj) in cells:
        v, u = np.mgrid[ci * CS:(ci + 1) * CS, cj * CS:(cj + 1) * CS].astype(float)
        r = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1).reshape(-1, 3)
        lam = -d_c / (r @ n_c)
        z = lam + rng.normal(0, noise, lam.shape)
        X = r * z[:, None]
        ij.append((ci, cj))
        cnt.append(len(X))
        s1.append(X.sum(0))
        s2.append(X.T @ X)
        pts.append(X)
    return np.array(ij), np.array(cnt, float), np.array(s1), np.array(s2), np.concatenate(pts)


def _synth_seg(n_c, d_c, cells, rng, idx):
    ij, cnt, s1, s2, pts = _synth_cells(unit(n_c), d_c, cells, rng)
    mask = torch.zeros(H // CS, W // CS, dtype=torch.bool)
    for a, b in cells:
        mask[a, b] = True
    sg = fit_cells(ij, cnt, s1, s2, SENSOR, [idx])
    sg.meas = PlaneMeas(n=torch.from_numpy(sg.n.copy()), d=sg.d, mask_cells=mask, n_px=int(cnt.sum()), mse=sg.mse,
                        pts_c=pts[::max(1, len(pts) // 500)][:500])
    sg.meas.cells = dict(ij=ij, cnt=cnt, s1=s1, s2=s2)
    return sg


def test_Q5_merge_in_frame():
    rng = np.random.default_rng(5)
    # (a) wall z = 2 m split by a bar (depth 1 m) over columns 14..17 -> left and right parts, not adjacent
    left = [(i, j) for i in range(24) for j in range(0, 14)]
    right = [(i, j) for i in range(24) for j in range(18, 32)]
    n_w = np.array([0.0, 0.0, -1.0])
    segs = [_synth_seg(n_w, 2.0, left, rng, 0), _synth_seg(n_w, 2.0, right, rng, 1)]
    planes = [s.meas for s in segs]
    out = merge_in_frame(segs, planes, SENSOR)
    assert len(out) == 1 and out[0].src == [0, 1], [o.src for o in out]
    ij = np.concatenate([s.cells_ij for s in segs])
    ref = fit_cells(ij, np.concatenate([s.cnt for s in segs]), np.concatenate([s.s1 for s in segs]),
                    np.concatenate([s.s2 for s in segs]), SENSOR, [0, 1])
    assert np.abs(ref.n - out[0].n).max() < 1e-12 and np.abs(ref.p - out[0].p).max() < 1e-12
    # (b) two parallel planes 5 cm apart at 2 m, one per image half
    segs = [_synth_seg(n_w, 2.0, left, rng, 0), _synth_seg(n_w, 2.05, right, rng, 1)]
    out = merge_in_frame(segs, [s.meas for s in segs], SENSOR)
    assert len(out) == 2
    # (c) bridging: A ~ B, B ~ C pass, A !~ C -> A and C never end in one group
    # three neighbouring strips (short lever arms, so the offset is not absorbed by the tilt uncertainty)
    # A and C: large precise blocks; B: small noisy patch between them (bridges through its large R_B)
    cellsA = [(i, j) for i in range(2, 22) for j in range(0, 13)]
    cellsB = [(i, j) for i in range(10, 13) for j in range(15, 17)]
    cellsC = [(i, j) for i in range(2, 22) for j in range(19, 32)]
    for delta in np.linspace(0.04, 0.0005, 160):
        sA = _synth_seg(n_w, 2.0, cellsA, np.random.default_rng(50), 0)
        sB = _synth_seg(n_w, 2.0 + delta, cellsB, np.random.default_rng(51), 1)
        sC = _synth_seg(n_w, 2.0 + 2 * delta, cellsC, np.random.default_rng(52), 2)
        p_ab = seg_seg_nis(sA, sB) <= Mo.CHI2_3 and seg_seg_nis(sB, sA) <= Mo.CHI2_3
        p_bc = seg_seg_nis(sB, sC) <= Mo.CHI2_3 and seg_seg_nis(sC, sB) <= Mo.CHI2_3
        p_ac = seg_seg_nis(sA, sC) <= Mo.CHI2_3 and seg_seg_nis(sC, sA) <= Mo.CHI2_3
        if p_ab and p_bc and not p_ac:
            out = merge_in_frame([sA, sB, sC], [sA.meas, sB.meas, sC.meas], SENSOR)
            groups = [o.src for o in out]
            print(f"Q5c: delta {delta * 100:.2f} cm, groups {groups}")
            assert not any(0 in g and 2 in g for g in groups)
            break
    else:
        raise AssertionError("Q5c: no bridging configuration found")
    print("Q5: occluded wall merged, parallel planes kept apart, bridge split")


# ============================================================================ Q6


def _scene4():
    return [(unit([0.05, 0.0, -1.0]), 3.0), (unit([1.0, 0.0, 0.05]), 1.2), (unit([0.0, -1.0, 0.0]), 0.9),
            (unit([-0.6, 0.3, -0.74]), 1.4)]


def _anchors_from(planes, P, ids=None):
    out = []
    for j, (n, d) in enumerate(planes):
        o = -d * n
        Bn = G.basis(n)
        a = A(j if ids is None else ids[j], n, o, Bn, P.copy(), grid_o=o, grid_B=Bn)
        a.est = Mo.OffsetEstimator(a.P, 0, 30)
        out.append(a)
    return out


def GY(a):
    return a.est.y(a.n, a.o, a.B, G.oplus)


def VK(k=1, Gm=np.zeros((3, 3))):
    return lambda a: a.est.cov_at(k, Gm)


def test_Q6_association():
    rng = np.random.default_rng(6)
    planes = _scene4()
    anchors = _anchors_from(planes, R_SEG)
    ok = 0
    for trial in range(300):
        Tp = G.se3_exp(rng.multivariate_normal(np.zeros(6), Q_GATE))
        segs = []
        perm = rng.permutation(4)
        for j in perm:
            n, d = planes[j]
            o = -d * n + G.basis(n) @ rng.normal(size=2) * 0.2
            sn, sp, sB = _noisy(n, o, G.basis(n), R_SEG, rng)
            segs.append(mk_seg(sn, sp, R_SEG, sB))
        _, assign, _ = ic_and_assign(segs, anchors, Tp, GY, VK(), Q_GATE)
        ok += sorted(assign) == sorted((i, int(perm[i])) for i in range(4))
    print(f"Q6a: correct assignment {ok}/300")
    assert ok >= 297
    # (b) j and k (parallel, k 4 cm in front of j) both with a measured offset of 4 cm (common map shift);
    #     the segment of K seen where K now is -> assigned to k
    n = np.array([0.0, 0.0, -1.0])
    aj = A(0, n, -2.0 * n, G.basis(n), R_SEG)
    ak = A(1, n, -1.96 * n, G.basis(n), R_SEG)
    for a in (aj, ak):
        a.est = Mo.OffsetEstimator(R_SEG / 10, 0, 30)
        a.est.b = np.array([0.0, 0.0, 0.04])
    seg = mk_seg(n, -1.92 * n + np.array([0.1, 0.0, 0.0]), R_SEG)
    cand, assign, info = ic_and_assign([seg], [aj, ak], np.eye(4), GY, VK(), Q_GATE)
    print(f"Q6b: NIS j {info[(0, 0)][0]:.1f}, NIS k {info[(0, 1)][0]:.2f}, assign {assign}")
    assert assign == [(0, 1)]
    # (c) back-facing anchor never a candidate
    back = _anchors_from([(-n, -2.0)], R_SEG, ids=[5])[0]
    cand, assign, _ = ic_and_assign([mk_seg(n, -2.0 * n, R_SEG)], [back], np.eye(4), GY, VK(), Q_GATE)
    assert not cand and not assign
    # (d) conflict(j) never gets a segment that j passes IC
    j = _anchors_from([(n, 2.0)], R_SEG, ids=[0])[0]
    c = _anchors_from([(n, 2.0)], R_SEG, ids=[1])[0]
    c.state, c.conflict_with = "conflict", 0
    s2 = [mk_seg(n, -2.0 * n, R_SEG)]
    _, assign, _ = ic_and_assign(s2, [c, j], np.eye(4), GY, VK(), Q_GATE)
    assert assign == [(0, 0)]
    print("Q6: Hungarian with ln det S on offset-corrected anchors, back-facing and conflict rules OK")


# ============================================================================ Q7


def _jc_case(rng, planes, Tt, offset=None, noise=True):
    anchors = {a.id: a for a in _anchors_from(planes, R_SEG)}
    segs = []
    for j, (n, d) in enumerate(planes):
        o = -d * n + G.basis(n) @ rng.normal(size=2) * 0.2
        nc, oc, Bc = G.to_camera(Tt, n, o, G.basis(n))
        if noise:
            nc, oc, Bc = _noisy(nc, oc, Bc, R_SEG, rng)
        if offset is not None and j in offset:
            oc = oc + offset[j] * nc
        segs.append(mk_seg(nc, oc, R_SEG, Bc))
    return anchors, segs


def test_Q7_joint_compat_and_init():
    rng = np.random.default_rng(7)
    Q = np.diag([0.01 ** 2] * 3 + [np.deg2rad(0.5) ** 2] * 3)
    planes5 = _scene4() + [(unit([0.5, 0.5, -0.7]), 2.0)]
    pairs = [(j, j) for j in range(5)]
    # (a) one pair wrong by 15 cm
    anchors, segs = _jc_case(rng, planes5, np.eye(4), offset={2: 0.15})
    kept, st, dropped = joint_compat(pairs, segs, anchors, np.eye(4), Q, GY, VK())
    print(f"Q7a: dropped {dropped}")
    assert dropped == [(2, 2)] and len(kept) == 4
    # (b) coherent 30 cm aliasing
    Tt = G.se3_exp(np.array([0.3, 0.0, 0.0, 0, 0, 0]))
    anchors, segs = _jc_case(rng, planes5, Tt)
    out = init_from_jc(pairs, segs, anchors, np.eye(4), Q, GY, VK())
    print(f"Q7b: {out['source']}")
    assert out["source"] == "jc_jump" and out["pose_pairs"] == []
    # (c) two planes -> rank < 6
    anchors, segs = _jc_case(rng, planes5[:2], np.eye(4))
    out = init_from_jc([(0, 0), (1, 1)], segs, anchors, np.eye(4), Q, GY, VK())
    assert out["source"] == "rank_lt6" and len(out["pose_pairs"]) == 2
    # (d) correct pairs: pose error after init <= 50 % of the gate's (median of 100 trials)
    ratios = []
    for _ in range(100):
        Tt = G.se3_exp(rng.multivariate_normal(np.zeros(6), Q))
        anchors, segs = _jc_case(rng, planes5, Tt)
        out = init_from_jc(pairs, segs, anchors, np.eye(4), Q, GY, VK())
        if out["source"] != "plane":
            continue
        e0 = np.linalg.norm(G.centre(Tt))
        ratios.append(np.linalg.norm(G.centre(out["T_init"]) - G.centre(Tt)) / max(e0, 1e-9))
    med = float(np.median(ratios))
    print(f"Q7d: {len(ratios)} plane inits, median error ratio {med:.3f}")
    assert len(ratios) >= 90 and med <= 0.5
    # (e) GN uses the ORIGINAL anchors: every map plane shifted by the same translation t (b_hat measured);
    #     JC on y_j passes at T_gate = I, and GN pulls the pose to the anchors: T_p translation = t
    t = np.array([0.03, -0.02, 0.025])
    anchors, segs = _jc_case(rng, planes5, np.eye(4), noise=False)
    for (s, j) in pairs:                                # map = anchors shifted by t: segment centre p -> p + t
        sg = segs[s]
        segs[s] = mk_seg(sg.n, sg.p + t, R_SEG, sg.B)
        a = anchors[j]
        a.est.b = np.array([0.0, 0.0, float(a.n @ t)])
        a.est.V = R_SEG / 10
    out = init_from_jc(pairs, segs, anchors, np.eye(4), Q, GY, VK())
    tp = out["T_init"][:3, 3]
    print(f"Q7e: source {out['source']}, jump {out['jc']['jump']:.2e}, T_p translation {tp}, map shift {t}")
    assert out["source"] == "plane" and np.linalg.norm(tp - t) < 1e-4


# ============================================================================ Q11 / Q12 (synthetic room, PlaneWorld API)

Q11_KINDS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")


def test_Q11_general():
    import json
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pw_sim import run_case
    rows, fails = [], []
    for kind in Q11_KINDS:
        for seed in range(5):
            r = run_case(kind, seed)
            rows.append(r)
            bad = []
            if r["nis_mean"] is None or not (2.0 <= r["nis_mean"] <= 4.0):
                bad.append(f"NIS {r['nis_mean']}")
            if r["pct_correct"] < 99.0:
                bad.append(f"correct {r['pct_correct']:.2f} %")
            if r["false_merges"]:
                bad.append(f"false merges {r['false_merges']}")
            if not r["one_each"]:
                bad.append(f"active per plane {r['active_per_plane']}")
            if r["invariants"]:
                bad.append(f"invariants {r['invariants']}")
            print(f"Q11 {kind} seed {seed}: NIS {r['nis_mean']:.2f} ({r['n_nis']}), correct {r['pct_correct']:.2f} %, "
                  f"merges {r['merges']}, false {r['false_merges']}, conflicts {r['conflicts']}, revisits {r['revisits']}, "
                  f"active/plane {r['active_per_plane']}, Gamma {np.round(np.sqrt(r['gamma']), 5).tolist()}"
                  + (f"  FAIL: {bad}" if bad else ""), flush=True)
            if bad:
                fails.append((kind, seed, bad))
    out = os.environ.get("Q11_OUT", os.path.join(ROOT, "results_exp", "a2g_p1v2", "q11.json"))
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rows, open(out, "w"), indent=1)
    except OSError:
        pass
    assert not fails, fails


def test_Q12_wrong_pair_rejected():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pw_sim import run_case
    r = run_case("P8", 0, inject_wrong_at=200)
    w = r["wrong"]
    print(f"Q12: forced P4 segment -> wall A anchor {w['j']}: rejected {w['rejected']}, b_hat shift "
          f"{w['shift_sigma']:.3f} sigma, false merges {r['false_merges']}")
    assert w["rejected"] and w["shift_sigma"] < 1.0 and r["false_merges"] == 0
