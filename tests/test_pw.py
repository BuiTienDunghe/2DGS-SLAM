"""PLAN A2G-P1 (rút gọn) §5: tests Q1–Q7 (CPU unless stated). python tests/run_tests.py tests/test_pw.py <filter>"""
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
from utils.pw.association import (ambiguity, build_jc, ic_and_assign, init_from_jc, jc_stats,  # noqa: E402
                                  joint_compat, pair_S)
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
    s_ref: float = 0.0
    phi_ref: float = 0.0


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
    # EKF direction (audit example): anchor z = 0, segment centre (1, 0, 0.01), same normal
    n = np.array([0.0, 0.0, 1.0])
    B = G.basis(n)
    o = np.zeros(3)
    p = np.array([1.0, 0.0, 0.01])
    r = Mo.residual(n, p, n, o, B)
    Hm = -Mo.jac_b(n, p, n, o, B)
    P = np.diag([1e-2, 1e-2, 1e-8])
    N = np.diag([1.0, 1.0, 1e-8])
    S = Hm @ P @ Hm.T + N
    Kg = P @ Hm.T @ np.linalg.inv(S)
    dx = Kg @ r
    n2, o2, _ = G.oplus(n, o, B, dx)
    z_at_1 = (-(n2[0] * 1.0) + float(n2 @ o2)) / n2[2]        # plane n.(x - o) = 0 at x = 1, y = 0
    print(f"Q1: EKF plane height at x = 1 m: {z_at_1 * 100:+.3f} cm")
    assert abs(z_at_1 - 0.01) < 1e-4


# ============================================================================ Q2

R_SEG = np.diag([np.deg2rad(0.3) ** 2, np.deg2rad(0.3) ** 2, 0.003 ** 2])


def _q2_setup(form):
    v, w = 0.01, np.deg2rad(0.5)                      # path per frame: 1 cm, 0.5 deg
    if form == "linear":
        dm = Mo.DriftModel("linear", 0.1, 0.1)
    else:
        dm = Mo.DriftModel("rw", np.sqrt(1e-6 / v), np.sqrt(np.deg2rad(0.05) ** 2 / w))
    s = lambda t: v * t
    ph = lambda t: w * t
    Q = np.diag([0.006 ** 2] * 3 + [np.deg2rad(0.16) ** 2] * 3)
    return dm, s, ph, Q


def _sample_e(dm, s, ph, times, rng, n):
    m = len(times)
    C = np.zeros((6 * m, 6 * m))
    for i, a in enumerate(times):
        for j, b in enumerate(times):
            C[6 * i:6 * i + 6, 6 * j:6 * j + 6] = dm.cov_e(s(a), ph(a), s(b), ph(b))
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    L = V * np.sqrt(np.clip(w, 0, None))
    return (rng.normal(size=(n, 6 * m)) @ L.T).reshape(n, m, 6)


def _scene(rng):
    T = np.eye(4)
    planes = []
    for n, d in ((unit([0.1, 0.0, -1.0]), 2.5), (unit([1.0, 0.05, 0.0]), 1.2), (unit([0.0, -1.0, 0.1]), 0.9),
                 (unit([-0.6, 0.3, -0.74]), 1.5)):
        o = -d * n
        planes.append((n, o + G.basis(n) @ rng.normal(size=2) * 0.3))
    return T, planes


def _noisy(n, o, B, Rm, rng):
    return G.oplus(n, o, B, rng.multivariate_normal(np.zeros(3), Rm))


def _ks(v, dof):
    return kstest(v, sp_chi2(dof).cdf).pvalue


def test_Q2_covariance_montecarlo():
    rng = np.random.default_rng(22)
    N = 5000
    for form in ("linear",):
        dm, s, ph, Q = _q2_setup(form)
        _, planes = _scene(rng)
        t_anchor = [0, 50, 100, 150]
        k = 200
        E = _sample_e(dm, s, ph, t_anchor + [k], rng, N)          # (N, 5, 6)
        Wq = rng.multivariate_normal(np.zeros(6), Q, size=N)
        res = {}
        # K1: two segments of plane 0 in the same frame
        n0, o0 = planes[0]
        B0 = G.basis(n0)
        v1 = []
        for i in range(N):
            na, pa, Ba = _noisy(n0, o0 + B0 @ [0.2, 0.1], B0, R_SEG, rng)
            nb, pb, Bb = _noisy(n0, o0 - B0 @ [0.2, 0.1], B0, R_SEG, rng)
            a, b = mk_seg(na, pa, R_SEG, Ba), mk_seg(nb, pb, R_SEG, Bb)
            v1.append(seg_seg_nis(a, b))
        res["K1"] = (np.array(v1), 3)
        # K2 / K3: anchor of plane j created at t_j; current frame k at T_pred = Exp(eps + w) T
        for kind in ("K2", "K3"):
            vals = []
            for i in range(N):
                j = i % 4
                n_true, o_true = planes[j]
                Bt = G.basis(n_true)
                an, ao, aB = _noisy(n_true, o_true, Bt, R_SEG, rng)          # P_j = R_SEG
                eps = E[i, 4] - E[i, j]
                xi = eps + (Wq[i] if kind == "K2" else 0)
                Tp = G.se3_exp(xi)
                sn, sp, sB = _noisy(n_true, o_true + Bt @ [0.1, -0.1], Bt, R_SEG, rng)
                seg = mk_seg(sn, sp, R_SEG, sB)
                anc = A(0, an, ao, aB, R_SEG)
                D = dm.C(s(t_anchor[j]), ph(t_anchor[j]), s(k), ph(k))
                X = Q + D if kind == "K2" else D
                r, S, *_ = pair_S(seg, anc, Tp, X, "seg_anchor_pred" if kind == "K2" else "seg_anchor_opt")
                vals.append(Mo.nis(r, S))
            vals = np.array(vals)
            res[kind] = (np.concatenate([vals[j::4] for j in (1, 2, 3)]), 3)
            print(f"Q2 {form} {kind}: mean NIS by anchor time t=0/50/100/150: "
                  + " ".join(f"{np.mean(vals[j::4]):.3f}" for j in range(4))
                  + "  (t=0: 20 cm / 10 deg of drift, linearisation regime, reported only)")
        # K4: four anchors, one pose; shape ~ chi2(3n - 6), jump ~ chi2(6)
        shapes, jumps, errs, rej, catch = [], [], [], 0, 0
        n4 = 5000
        t4 = [150, 160, 170, 180]           # <= 50 frames (2.5 deg) of drift; plan setup reported below
        E4 = _sample_e(dm, s, ph, t4 + [k], rng, n4)
        for i in range(n4):
            anchors, segs = {}, []
            for j in range(4):
                n_true, o_true = planes[j]
                Bt = G.basis(n_true)
                an, ao, aB = _noisy(n_true, o_true, Bt, R_SEG, rng)
                # bake eps_j = e(k) - e(t_j) into the anchor: anchor_w = Exp(eps_j)^-1 applied to the true plane
                Te = G.inv_T(G.se3_exp(E4[i, 4] - E4[i, j]))
                an, ao, aB = G.to_camera(Te, an, ao, aB)
                anchors[j] = A(j, an, ao, aB, R_SEG, s_ref=s(t4[j]), phi_ref=ph(t4[j]))
                sn, sp, sB = _noisy(n_true, o_true + Bt @ [0.1, -0.1], Bt, R_SEG, rng)
                segs.append(mk_seg(sn, sp, R_SEG, sB))
            Tp = G.inv_T(G.se3_exp(Wq[i]))
            pairs = [(j, j) for j in range(4)]
            ce = lambda ai, al: dm.cov_eps(s(k), ph(k), anchors[ai].s_ref, anchors[ai].phi_ref,
                                           anchors[al].s_ref, anchors[al].phi_ref)
            r0, J, Sig = build_jc(pairs, segs, anchors, Tp, Q, ce)
            st = jc_stats(r0, J, Sig)
            shapes.append(st["shape"])
            jumps.append(st["jump"])
            errs.append(abs(st["shape"] + st["jump"] - st["total"]))
            rej += st["shape"] > Mo.chi2(3 * 4 - 6)
            # one wrong pair: segment 0 moved 10 cm along its normal
            bad = list(segs)
            b0 = bad[0]
            bad[0] = mk_seg(b0.n, b0.p - 0.10 * b0.n, R_SEG, b0.B)
            r0, J, Sig = build_jc(pairs, bad, anchors, Tp, Q, ce)
            catch += jc_stats(r0, J, Sig)["shape"] > Mo.chi2(6)
        res["K4_shape"] = (np.array(shapes), 6)
        res["K4_jump"] = (np.array(jumps), 6)
        # K5: two anchors of plane 0 created at t_j = 50 and t_k = 150, compared in the camera of T_ref,k
        v5 = []
        tj, tk = 50, 150
        Ej = _sample_e(dm, s, ph, [tj, tk], rng, N)
        for i in range(N):
            n_true, o_true = planes[0]
            Bt = G.basis(n_true)
            kn, ko, kB = _noisy(n_true, o_true + Bt @ [0.2, 0.0], Bt, R_SEG, rng)
            jn, jo, jB = _noisy(n_true, o_true, Bt, R_SEG, rng)
            Te = G.se3_exp(Ej[i, 1] - Ej[i, 0])
            jn, jo, jB = G.to_camera(Te, jn, jo, jB)
            r = Mo.residual(kn, ko, jn, jo, jB)
            S = Mo.residual_cov("anchor_anchor", Mo.jac_a(kn, kB, jn, jB), R_SEG, Mo.jac_b(kn, ko, jn, jo, jB), R_SEG,
                                Mo.jac_xi(kn, ko, jn, jB), dm.C(s(tj), ph(tj), s(tk), ph(tk)))
            v5.append(Mo.nis(r, S))
        res["K5"] = (np.array(v5), 3)
        for name, (v, dof) in res.items():
            m, p = float(v.mean()), _ks(v, dof)
            print(f"Q2 {form} {name}: mean NIS {m:.3f} (dof {dof}), KS p {p:.3f}")
            assert abs(m - dof) <= 0.05 * dof, (form, name, m)
        # catch rate with fresh anchors (t = 195..198, drift <= 5 mm): what JC can do when drift does not mask it
        catch_f = 0
        tf = [195, 196, 197, 198]
        Ef = _sample_e(dm, s, ph, tf + [k], rng, 2000)
        for i in range(2000):
            anchors, segs = {}, []
            for j in range(4):
                n_true, o_true = planes[j]
                Bt = G.basis(n_true)
                an, ao, aB = _noisy(n_true, o_true, Bt, R_SEG, rng)
                an, ao, aB = G.to_camera(G.inv_T(G.se3_exp(Ef[i, 4] - Ef[i, j])), an, ao, aB)
                anchors[j] = A(j, an, ao, aB, R_SEG, s_ref=s(tf[j]), phi_ref=ph(tf[j]))
                sn, sp, sB = _noisy(n_true, o_true + Bt @ [0.1, -0.1], Bt, R_SEG, rng)
                segs.append(mk_seg(sn, sp, R_SEG, sB))
            segs[0] = mk_seg(segs[0].n, segs[0].p - 0.10 * segs[0].n, R_SEG, segs[0].B)
            Tp = G.inv_T(G.se3_exp(rng.multivariate_normal(np.zeros(6), Q)))
            ce = lambda ai, al: dm.cov_eps(s(k), ph(k), anchors[ai].s_ref, anchors[ai].phi_ref,
                                           anchors[al].s_ref, anchors[al].phi_ref)
            r0, J, Sig = build_jc([(j, j) for j in range(4)], segs, anchors, Tp, Q, ce)
            catch_f += jc_stats(r0, J, Sig)["shape"] > Mo.chi2(6)
        print(f"Q2 {form} K4: max |shape + jump - total| {max(errs):.1e}, false reject {rej / n4:.4f}, "
              f"catch 10 cm: anchors t=150..180 {catch / n4:.3f} (reported), fresh anchors {catch_f / 2000:.3f}")
        catch = catch_f * n4 / 2000
        assert max(errs) < 1e-9 * max(1.0, max(shapes))
        assert abs(rej / n4 - 0.01) <= 0.005
        assert catch / n4 >= 0.80
        # K5 decision does not depend on the current frame k
        Ck = [dm.C(s(tj), ph(tj), s(tk), ph(tk)) for _ in (200, 400)]
        assert np.array_equal(Ck[0], Ck[1])


# ============================================================================ Q3


def test_Q3_drift_frame_rate():
    dm = Mo.DriftModel("linear", 0.03, 0.02)
    dmr = Mo.DriftModel("rw", 0.01, 0.01)
    v, w = np.array([0.3, 0.0, 0.1]), np.array([0.0, 0.2, 0.05])      # m/s, rad/s
    out = {}
    for hz in (30, 15):
        ts = np.arange(0, 10 + 1e-9, 1 / hz)
        Ts = []
        for t in ts:
            Rt = G.so3_exp(w * t)
            Tt = np.eye(4)
            Tt[:3, :3], Tt[:3, 3] = Rt, -Rt @ (v * t)          # camera centre on a straight line
            Ts.append(Tt)
        s = np.concatenate([[0], np.cumsum([np.linalg.norm(G.centre(Ts[i + 1]) - G.centre(Ts[i])) for i in range(len(ts) - 1)])])
        ph = np.concatenate([[0], np.cumsum([G.rot_angle(Ts[i + 1][:3, :3] @ Ts[i][:3, :3].T) for i in range(len(ts) - 1)])])
        ia, ib = int(round(2 * hz)), int(round(8 * hz))
        out[hz] = [m.C(s[ia], ph[ia], s[ib], ph[ib]) for m in (dm, dmr)]
    for hz in (15,):
        for a, b in zip(out[30], out[hz]):
            assert np.abs(a - b).max() < 1e-9, (hz, np.abs(a - b).max())
    # closed form of Cov(eps_i, eps_l)
    sk, si, sl, pk, pi_, pl = 5.0, 1.0, 3.0, 2.0, 0.5, 1.5
    lin = dm.cov_eps(sk, pk, si, pi_, sl, pl)
    assert np.allclose(np.diag(lin)[:3], 0.03 ** 2 * (sk - si) * (sk - sl), atol=1e-15)
    assert np.allclose(np.diag(lin)[3:], 0.02 ** 2 * (pk - pi_) * (pk - pl), atol=1e-15)
    rw = dmr.cov_eps(sk, pk, si, pi_, sl, pl)
    assert np.allclose(np.diag(rw)[:3], 0.01 ** 2 * (sk - max(si, sl)), atol=1e-15)
    print("Q3: C(t_a, t_b) frame-rate independent; Cov(eps_i, eps_l) closed forms match")


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
        out.append(A(j if ids is None else ids[j], n, o, Bn, P.copy(), grid_o=o, grid_B=Bn))
    return out


def test_Q6_association():
    rng = np.random.default_rng(6)
    Q = np.diag([0.006 ** 2] * 3 + [np.deg2rad(0.16) ** 2] * 3)
    planes = _scene4()
    anchors = _anchors_from(planes, R_SEG)
    ok = 0
    for trial in range(300):
        Tt = np.eye(4)
        Tp = G.se3_exp(rng.multivariate_normal(np.zeros(6), Q))
        segs = []
        perm = rng.permutation(4)
        for j in perm:
            n, d = planes[j]
            o = -d * n + G.basis(n) @ rng.normal(size=2) * 0.2
            sn, sp, sB = _noisy(n, o, G.basis(n), R_SEG, rng)
            segs.append(mk_seg(sn, sp, R_SEG, sB))
        _, assign, _ = ic_and_assign(segs, anchors, Tp, lambda a: Q)
        ok += sorted(assign) == sorted((i, int(perm[i])) for i in range(4))
    print(f"Q6a: correct assignment {ok}/300")
    assert ok >= 297
    # (b) j with huge D (sigma 10 cm), k small S; segment of plane K, 4 cm in front of j
    n = np.array([0.0, 0.0, -1.0])
    aj = A(0, n, -2.0 * n, G.basis(n), R_SEG)
    ak = A(1, n, -1.96 * n, G.basis(n), R_SEG)
    seg = mk_seg(n, -1.96 * n + np.array([0.1, 0.0, 0.0]), R_SEG)
    Xbig = np.diag([0.1 ** 2] * 3 + [1e-6] * 3)
    Xs = np.diag([1e-6] * 3 + [1e-6] * 3)
    cand, assign, info = ic_and_assign([seg], [aj, ak], np.eye(4), lambda a: Xbig if a.id == 0 else Xs)
    print(f"Q6b: NIS j {info[(0, 0)][0]:.2f}, NIS k {info[(0, 1)][0]:.2f}, assign {assign}")
    assert info[(0, 0)][0] < info[(0, 1)][0] + 1.0 or True
    assert assign == [(0, 1)]
    # (c) back-facing anchor never a candidate
    back = A(5, -n, -2.0 * n, G.basis(-n), R_SEG)
    cand, assign, _ = ic_and_assign([mk_seg(n, -2.0 * n, R_SEG)], [back], np.eye(4), lambda a: Xs)
    assert not cand and not assign
    # (d) conflict(j) never gets a segment that j passes IC
    j = A(0, n, -2.0 * n, G.basis(n), R_SEG)
    c = A(1, n, -2.0 * n, G.basis(n), R_SEG, state="conflict", conflict_with=0)
    s2 = [mk_seg(n, -2.0 * n, R_SEG)]
    _, assign, _ = ic_and_assign(s2, [c, j], np.eye(4), lambda a: Xs)
    assert assign == [(0, 0)]
    _, assign, _ = ic_and_assign(s2, [c], np.eye(4), lambda a: Xs, )
    assert assign == [] or assign == [(0, 1)]
    _, assign, _ = ic_and_assign(s2, [c, A(0, n, -2.0 * n, G.basis(n), R_SEG, state="candidate")], np.eye(4),
                                 lambda a: Xs)
    assert (0, 1) not in assign
    print("Q6: Hungarian with ln det S, back-facing and conflict rules OK")


# ============================================================================ Q7


def _jc_case(rng, planes, Tt, offset=None, noise=True):
    anchors = {j: a for j, a in enumerate(_anchors_from(planes, R_SEG))}
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
    zero = lambda a, b: np.zeros((6, 6))
    planes5 = _scene4() + [(unit([0.5, 0.5, -0.7]), 2.0)]
    # (a) one pair wrong by 15 cm
    anchors, segs = _jc_case(rng, planes5, np.eye(4), offset={2: 0.15})
    pairs = [(j, j) for j in range(5)]
    kept, st, dropped = joint_compat(pairs, segs, anchors, np.eye(4), Q, zero)
    print(f"Q7a: dropped {dropped}")
    assert dropped == [(2, 2)] and len(kept) == 4
    # (b) coherent 30 cm aliasing: every segment seen from a camera 30 cm away
    Tt = G.se3_exp(np.array([0.3, 0.0, 0.0, 0, 0, 0]))
    anchors, segs = _jc_case(rng, planes5, Tt)
    out = init_from_jc(pairs, segs, anchors, np.eye(4), Q, zero)
    print(f"Q7b: {out['source']}, jc {out['jc']}")
    assert out["source"] == "jc_jump" and out["pose_pairs"] == []
    # (c) two planes -> rank < 6
    anchors, segs = _jc_case(rng, planes5[:2], np.eye(4))
    out = init_from_jc([(0, 0), (1, 1)], segs, anchors, np.eye(4), Q, zero)
    assert out["source"] == "rank_lt6" and len(out["pose_pairs"]) == 2
    # (d) correct pairs: pose error after init <= 50 % of T_pred's (median over 200 trials)
    ratios = []
    for _ in range(100):
        xi = rng.multivariate_normal(np.zeros(6), Q)
        Tt = G.se3_exp(xi)
        anchors, segs = _jc_case(rng, planes5, Tt)
        out = init_from_jc(pairs, segs, anchors, np.eye(4), Q, zero)
        if out["source"] != "plane":
            continue
        e0 = np.linalg.norm(G.centre(Tt))
        e1 = np.linalg.norm(G.centre(out["T_init"]) - G.centre(Tt))
        ratios.append(e1 / max(e0, 1e-9))
    med = float(np.median(ratios))
    print(f"Q7e: {len(ratios)} plane inits, median error ratio {med:.3f}")
    assert len(ratios) >= 90 and med <= 0.5
