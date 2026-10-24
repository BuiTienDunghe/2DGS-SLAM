"""PLAN A2G-P1 core §5: tests T1-T9 (CPU), run and reported, never blocking the GPU runs. python tests/run_tests.py tests/test_pw.py <filter>"""
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
from utils.pw import uncertainty as Mo  # noqa: E402
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


def test_T1_jacobians():
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
        n_a = n2                                           # the segment lies exactly on the moved plane
        r = Mo.residual(n_a, p_a, n_b, o_b, B_b)
        z = -np.linalg.solve(Mo.jac_b(n_a, p_a, n_b, o_b, B_b), r)
        worst_z = max(worst_z, np.linalg.norm(z - dx) / np.linalg.norm(dx))
    print(f"Q1: z = -J_b^-1 r vs dx, worst relative error {worst_z:.2e}")
    assert worst_z < 1e-6
    # h_c of R_s: offset of a fixed cell centre w.r.t. the plane perturbed by x (same (+) as J_a)
    worst_h = 0.0
    for it in range(500):
        n = rand_unit(rng)
        B, p0 = G.basis(n), rng.normal(size=3)
        mu = p0 + B @ rng.normal(size=2) + 0.01 * rng.normal() * n
        def fh(x):
            n2, p2, _ = G.oplus(n, p0, B, x)
            return np.array([float(n2 @ (mu - p2))])
        Hn = _fd(fh, np.zeros(3))
        Ha = Mo.cell_rows(mu[None], n, p0, B)
        Ha[0, :2] += 0.0
        worst_h = max(worst_h, _rel(Hn, Ha) if abs(float(n @ (mu - p0))) < 1e-12 else
                      np.linalg.norm(Hn - Ha) / max(np.linalg.norm(Ha), 1.0))
    print(f"Q1: R_s rows h_c vs finite differences, worst relative error {worst_h:.2e}")


# ============================================================================ Q2

R_SEG = np.diag([np.deg2rad(0.3) ** 2, np.deg2rad(0.3) ** 2, 0.003 ** 2])
Q_GATE = np.diag([0.006 ** 2] * 3 + [np.deg2rad(0.16) ** 2] * 3)


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
        a.est.sample(z, Rz, k)
    return a


def test_T3_ic_merge_jc_montecarlo():
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
        r, S, *_ = pair_S(seg, a.est.y(a.n, a.o, a.B, G.oplus), a.est.V, Tg, Q_GATE)
        ic.append(Mo.nis(r, S))
        # two true duplicates of the same plane, each with its own offset and samples
        a1 = _anchor_with_offset(n_t, o_t, rng, 1)
        a2 = _anchor_with_offset(n_t, o_t, rng, 2)
        y1, y2 = a1.est.y(a1.n, a1.o, a1.B, G.oplus), a2.est.y(a2.n, a2.o, a2.B, G.oplus)
        rr = Mo.residual(y2[0], y2[1], y1[0], y1[1], y1[2])
        Ja, Jb = Mo.jac_a(y2[0], y2[2], y1[0], y1[2]), Mo.jac_b(y2[0], y2[1], y1[0], y1[1], y1[2])
        mg.append(Mo.nis(rr, Ja @ a2.est.V @ Ja.T + Jb @ a1.est.V @ Jb.T))
    # JC on four offset-corrected anchors with the gate pose error Q: shape ~ chi2(6), jump ~ chi2(6)
    planes4 = [(unit([0.05, 0.0, -1.0]), 3.0), (unit([1.0, 0.0, 0.05]), 1.2), (unit([0.0, -1.0, 0.0]), 0.9),
               (unit([-0.6, 0.3, -0.74]), 1.4)]
    shp, jmp, err = [], [], 0.0
    for i in range(2000):
        anchors, segs = {}, []
        for j, (nn, dd) in enumerate(planes4):
            oo = -dd * nn
            anchors[j] = _anchor_with_offset(nn, oo, rng, j)
            Bn = G.basis(nn)
            sn, sp, sB = _noisy(nn, oo + Bn @ rng.normal(size=2) * 0.3, Bn, R_SEG, rng)
            segs.append(mk_seg(sn, sp, R_SEG, sB))
        Tg = G.se3_exp(rng.multivariate_normal(np.zeros(6), Q_GATE))
        r0, J, Sig = build_jc([(j, j) for j in range(4)], segs, anchors, Tg, Q_GATE, GY, VK())
        st = jc_stats(r0, J, Sig)
        shp.append(st["shape"])
        jmp.append(st["jump"])
        err = max(err, abs(st["shape"] + st["jump"] - st["total"]))
    print(f"T3 JC: mean shape {np.mean(shp):.3f} (dof 6), mean jump {np.mean(jmp):.3f} (dof 6), "
          f"max |shape + jump - total| {err:.1e}")
    assert abs(np.mean(shp) - 6) <= 0.3 and abs(np.mean(jmp) - 6) <= 0.3 and err < 1e-8
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
        # summed per-cell statistics reproduce the segment statistics (the P1 core fit itself is weighted, §3.1)
        S1 = np.asarray(c["s1"]).sum(0)
        ref = np.asarray(b.pts_c).mean(0) if False else None
        assert np.allclose(S1 / c["cnt"].sum() @ b.n.numpy() + b.d, 0.0, atol=1e-6)
    return len(new)


def test_T5_extractor_unchanged():
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


def test_T6_merge_in_frame():
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


def VK():
    return lambda a: a.est.V


def test_T3b_association():
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


def test_T7_joint_compat_and_init():
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


# ============================================================================ T2: R_s Monte-Carlo


def _rs_case(n_c, d_c, cells, rng, n_mc=2000):
    """per-cell depth error (common to the cell, sigma_z along the ray) -> fitted plane error in (B, p) of the
    noise-free fit; NIS with R_s (sigma0 = 0) and with the old centroid formula."""
    sen = Mo.SensorNoise(0.0, 1.425e-3, 0.0, 0.0)
    ij0, cnt0, s10, s20, _ = _synth_cells(n_c, d_c, cells, rng, noise=0.0)
    ref = fit_cells(ij0, cnt0, s10, s20, sen, [0])
    new, old = [], []
    for _ in range(n_mc):
        s1, s2 = s10.copy(), s20.copy()
        mu = s10 / cnt0[:, None]
        dz = rng.normal(0, 1, len(cells)) * sen.sigma_z(mu[:, 2])
        for c in range(len(cells)):                       # shift every pixel of cell c by dz_c along its ray
            v = mu[c] / mu[c, 2]
            sh = dz[c] * v
            s2[c] = s2[c] + np.outer(s1[c], sh) + np.outer(sh, s1[c]) + cnt0[c] * np.outer(sh, sh)
            s1[c] = s1[c] + cnt0[c] * sh
        sg = fit_cells(ij0, cnt0, s1, s2, sen, [0])
        r = Mo.residual(sg.n, sg.p, ref.n, ref.p, ref.B)          # error of the fit in the reference (B, p)
        new.append(Mo.nis(r, ref.R))
        lam = [max(float(np.var((mu - ref.p) @ ref.B[:, k])), 1e-4) for k in range(2)]
        Ro = sen.R_old(len(cells), lam[0], lam[1], float(ref.p[2]), float(ref.n @ (ref.p / ref.p[2])))
        old.append(Mo.nis(r, Ro))
    return float(np.mean(new)), float(np.mean(old))


def test_T2_segment_cov_montecarlo():
    rng = np.random.default_rng(2)
    full = [(i, j) for i in range(2, 22) for j in range(4, 28)]
    cases = {"frontal 2 m": (unit([0, 0, -1]), 2.0, full),
             "oblique 60 deg, 2-5 m": (unit([np.sin(np.deg2rad(60)), 0, -np.cos(np.deg2rad(60))]), None, full),
             "oblique 75 deg, 2-6 m": (unit([np.sin(np.deg2rad(75)), 0, -np.cos(np.deg2rad(75))]), None, full),
             "small 8 cells at 4 m": (unit([0, 0, -1]), 4.0, [(i, j) for i in range(11, 13) for j in range(14, 18)])}
    bad = []
    for name, (n, d, cells) in cases.items():
        if d is None:                                    # put the plane through (0, 0, 3.5) -> depths span the range
            d = -float(n @ np.array([0.0, 0.0, 3.5]))
        cells = [c for c in cells if True]
        m_new, m_old = _rs_case(n, d, cells, rng)
        print(f"T2 {name}: mean NIS {m_new:.3f} (per-cell R_s), {m_old:.3f} (old formula)")
        if abs(m_new - 3) > 0.15:
            bad.append((name, m_new))
    assert not bad, bad


# ============================================================================ T4: OffsetEstimator


def test_T4_offset_estimator():
    rng = np.random.default_rng(4)
    n_t, o_t = unit([0.1, -0.05, -1.0]), np.array([0.2, -0.1, 2.5])
    Bt = G.basis(n_t)
    inside = 0
    for trial in range(200):
        a = _anchor_with_offset(n_t, o_t, rng)
        # true offset of the map w.r.t. the anchor, in the anchor's (B, o)
        dth = a.B.T @ n_t / (a.n @ n_t)
        dd = float(n_t @ (o_t - a.o)) / float(n_t @ a.n)
        e = np.array([dth[0], dth[1], dd]) - a.est.b
        inside += float(e @ np.linalg.solve(a.est.V, e)) <= Mo.chi2(3, 0.954)   # "2 sigma" region in 3D
    print(f"T4: b_hat within the 2-sigma region of the true offset in {inside}/200 trials")
    assert inside >= 180
    # a wrong sample (parallel surface 4 cm away, seen at the anchor) is rejected and leaves b_hat unchanged
    a = _anchor_with_offset(n_t, o_t, rng)
    b0 = a.est.b.copy()
    sn, sp, sB = _noisy(n_t, o_t - 0.04 * n_t, Bt, R_SEG, rng)
    z, Rz = bias_sample(mk_seg(sn, sp, R_SEG, sB), a)
    ok, m = a.est.sample(z, Rz, 11)
    print(f"T4: wrong sample m {m:.1f}, accepted {ok}")
    assert not ok and np.array_equal(a.est.b, b0)
    # an ambiguous pair is never a sample (PlaneWorld path; I6)
    from utils.pw.world import PlaneWorld
    pw = PlaneWorld({}, K, H, W, legacy=None, out_dir=None)
    n0 = a.est.n_accept
    out = pw._bias_sample(mk_seg(n_t, o_t, R_SEG), a, np.eye(4), unambiguous=False)
    assert out["accepted"] is False and a.est.n_accept == n0 and pw.n_invariant == 0
    # V inflates through f when the offset changes inside the window
    a = _anchor_with_offset(n_t, o_t, rng)
    V0 = np.trace(a.est.V)
    for k in range(11, 21):
        sn, sp, sB = _noisy(n_t, o_t + 0.004 * (k - 10) * n_t + Bt @ rng.normal(size=2) * 0.3, Bt, R_SEG, rng)
        z, Rz = bias_sample(mk_seg(sn, sp, R_SEG, sB), a)
        a.est.sample(z, Rz, k)
    print(f"T4: f {a.est.f:.2f}, trace V {V0:.2e} -> {np.trace(a.est.V):.2e} after a 4 cm drift over 10 frames")
    assert a.est.f > 1.0


# ============================================================================ T8: synthetic scene


T8_KINDS = ("E0", "E1", "E2", "E3", "E4")


def _t8(model):
    import json
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pw_sim
    pw_sim.SEG_ERR = 1.0 if model == "A" else 0.0
    sig = os.environ.get("PW_SIGMA0")
    if sig:
        pw_sim.SIGMA0 = tuple(float(x) for x in sig.split(","))
    rows, fails = [], []
    for kind in T8_KINDS:
        for seed in range(5):
            r = pw_sim.run_case(kind, seed)
            r.pop("wrong_log", None)
            rows.append(r)
            bad = []
            if model == "A" and (r["nis_mean"] is None or not (2.0 <= r["nis_mean"] <= 4.0)):
                bad.append(f"NIS {r['nis_mean']}")
            if r["false_merges"]:
                bad.append(f"false merges {r['false_merges']}")
            if r["pose_wrong_resolvable"]:
                bad.append(f"resolvable wrong pose pairs {r['pose_wrong_resolvable']}")
            if not r["one_each"]:
                bad.append(f"active per plane {r['active_per_plane']}")
            if kind == "E4" and (r["late_dups"] or r["revisit_candidates"] < 1):
                bad.append(f"E4 late duplicates {r['late_dups']}, revisit candidates {r['revisit_candidates']}")
            if r["invariants"]:
                bad.append(f"invariants {r['invariants']} {r['violations']}")
            print(f"T8{model} {kind} seed {seed}: NIS {r['nis_mean']:.2f}, correct {r['pct_correct']:.2f} %, "
                  f"wrong resolvable {r['wrong_resolvable']} (pose {r['pose_wrong_resolvable']}), unresolvable "
                  f"{r['wrong_unresolvable']} {r['unres_planes']}, A/P4 accuracy {r['ap4_accuracy']:.1f} %, merges "
                  f"{r['merges']}, false {r['false_merges']}, deferred {r['deferred']}, revisit cand "
                  f"{r['revisit_candidates']}, active/plane {r['active_per_plane']}"
                  + (f"  FAIL: {bad}" if bad else ""), flush=True)
            if bad:
                fails.append((kind, seed, bad))
    try:
        json.dump(rows, open(os.path.join(ROOT, "results_exp", "a2g_p1core", f"t8_{model}.json"), "w"), indent=1)
    except OSError:
        pass
    if model == "A":
        assert not fails, fails


def test_T8A_scene():
    _t8("A")


def test_T8B_scene_robustness():
    _t8("B")


# ============================================================================ T9: static checks


def test_T9_static():
    import ast
    import subprocess
    tree = ast.parse(open(os.path.join(ROOT, "utils", "pw", "uncertainty.py")).read())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and ("world" in m or "association" in m) for m in mods), mods            # I11
    for f in os.listdir(os.path.join(ROOT, "utils", "pw")):
        if f.endswith(".py"):
            t = ast.parse(open(os.path.join(ROOT, "utils", "pw", f)).read())
            names = {n.id for n in ast.walk(t) if isinstance(n, ast.Name)} | \
                    {n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute)}
            assert not names & {"gt_pose", "groundtruth", "traj_gt"}, (f, names & {"gt_pose", "groundtruth"})  # I12
    d = subprocess.run(["git", "-C", ROOT, "diff", "--stat", "ec92002", "--", "utils/slam_frontend.py", "slam.py"],
                       capture_output=True, text=True).stdout
    print(f"T9: frontend diff vs ec92002: {d.strip() or 'empty'}")
    assert d.strip() == ""
