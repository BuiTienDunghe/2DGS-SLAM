"""PLAN MONOGS-PW2 §5: tests T8-T11 (CPU). T12 and the R1 reproducibility check are SLAM runs (tools/monogs/pw2/tests_doc_pw2.py).
run from monogs/: python tests/test_pw2.py [--out ../docs/monogs/pw2/tests_pw2.json]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "tools", "monogs", "pw2"))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "tools", "monogs"))
os.chdir(ROOT)

import tests.test_pw as T1  # noqa: E402  (helpers of the R1 tests)
from utils import pgo_planes as PG  # noqa: E402
from utils.obs_decouple import DecoupledPoseStep, ManualAdam, obs_basis  # noqa: E402
from utils.plane_anchor import NoiseModel, PWAnchorMap, plane_w2c  # noqa: E402
from utils.pw_frontend import basis_normals  # noqa: E402

CALP = "configs/rgbd/tum/pw2_calib.yaml"
CAL = yaml.safe_load(open(CALP))["Noise"] if os.path.exists(CALP) else dict(
    sigma_n=np.deg2rad(0.36), sigma_d_a=0.0076, sigma_d_b=0.0, sigma_eff_n=np.deg2rad(1.0), sigma_eff_d_a=0.02,
    sigma_eff_d_b=0.0, sigma_odo_t=0.01, sigma_odo_r=np.deg2rad(0.5))
EFF = NoiseModel(CAL["sigma_eff_n"], CAL["sigma_eff_d_a"], CAL["sigma_eff_d_b"])
PGN = NoiseModel(CAL["sigma_n"], CAL["sigma_d_a"], CAL["sigma_d_b"])


def cfg(**kw):
    return T1.cfg(match_select="e_first", **kw)


# ------------------------------------------------------------------ T8
def t8():
    from calib2 import fit_ab, mad, tbasis
    rng = np.random.default_rng(0)
    s = np.deg2rad(0.5)
    comps, angs = [], []
    for _ in range(20000):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        B = tbasis(n)
        na = n + B.T @ rng.normal(0, s, 2)
        nb = n + B.T @ rng.normal(0, s, 2)
        na /= np.linalg.norm(na)
        nb /= np.linalg.norm(nb)
        comps += list(tbasis(na) @ (nb - na))
        angs.append(np.arccos(np.clip(na @ nb, -1, 1)))
    est = mad(comps, True)
    r1 = mad(angs, True)
    a_ok = abs(est - s) / s < 0.05
    # (b) two "sequences" with different a, same b
    b_true = 0.002
    fits = []
    for a_true in (0.003, 0.009):
        z = rng.uniform(0.5, 3.5, 6000)
        dd = rng.normal(0, a_true + b_true * z ** 2) * np.sqrt(2)
        rows = []
        for lo in np.arange(0.5, 3.5, 0.5):
            m = (z >= lo) & (z < lo + 0.5)
            rows.append((lo + 0.25, mad(dd[m], True), int(m.sum())))
        fits.append(fit_ab([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]))
    b_med = float(np.median([f_[1] for f_ in fits]))
    return dict(passed=bool(a_ok and abs(b_med - b_true) / b_true < 0.15), a_sigma_est_deg=float(np.degrees(est)),
                a_sigma_true_deg=0.5, a_r1_estimator_deg=float(np.degrees(r1)), a_r1_ratio=float(r1 / s),
                b_fits=[list(map(float, f_)) for f_ in fits], b_median=b_med, b_true=b_true)


# ------------------------------------------------------------------ T9
def _map_with(active_d, cand_d, z=2.0):
    m = PWAnchorMap(cfg(), T1.K, T1.H, T1.W, EFF)
    I = np.eye(4)
    m.update_at_keyframe_pw(0, [T1.meas([0, 0, -1], active_d, z=z)], I)             # active (first keyframe)
    m.update_at_keyframe_pw(5, [T1.meas([0, 0, -1], cand_d, z=z)], I)               # candidate (outside inlier of active)
    return m


def t9():
    out = {}
    I = np.eye(4)
    sd = EFF.sd(2.0)
    da = min(4.5 * sd, 0.095)                       # active anchor: inside the E1c gate, e > 3
    if da / sd <= 3.0:
        return dict(passed=False, error=f"sigma_eff,d = {sd:.4f} m too large for a 10 cm gate test")
    m = _map_with(2.0, 2.0 + da)
    cand = max(m.anchors)
    p = T1.meas([0, 0, -1], 2.0 + da + 0.2 * sd, z=2.0)
    pr = m.match_nn([p], I)
    out["a"] = dict(passed=pr == [(0, cand)], pairs=pr, cand=cand, e_active=m.e_pair(p, m.anchors[0], I), e_cand=m.e_pair(p, m.anchors[cand], I))
    n_before = len(m.anchors)
    m.update_at_keyframe_pw(10, [p], I)
    out["b"] = dict(passed=m.anchors[cand].active and len(m.anchors) == n_before, active=m.anchors[cand].active,
                    n_anchors=[n_before, len(m.anchors)])
    m2 = _map_with(2.0, 2.0 + 2.5 * sd)
    p2 = T1.meas([0, 0, -1], 2.0 + 1.2 * sd, z=2.0)                # inlier for both
    pr2 = m2.match_nn([p2], I)
    out["c"] = dict(passed=pr2 == [(0, 0)], pairs=pr2)
    out["d"] = t9d()
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def t9d():
    """cost mode identical to R1 (commit 32820ba) on 10 real snt_far frames."""
    old = "/tmp/plane_anchor_32820ba.py"
    subprocess.run(f"git show 32820ba:monogs/utils/plane_anchor.py > {old}", shell=True, check=True, cwd=ROOT)
    spec = importlib.util.spec_from_file_location("pa_r1", old)
    pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa)
    from munch import munchify
    from utils.config_utils import load_config
    from utils.dataset import load_dataset
    from utils.plane_anchor import PlaneExtractor
    c = load_config("configs/rgbd/tum/fr3_snt_far_sp.yaml")
    mp_ = munchify(c["model_params"])
    ds = load_dataset(mp_, mp_.source_path, config=c)
    ex = PlaneExtractor(T1.E1C)
    noise = NoiseModel(*(yaml.safe_load(open("configs/rgbd/tum/pw_calib.yaml"))["Noise"][k] for k in ("sigma_n", "sigma_d_a", "sigma_d_b")))
    c_new = T1.cfg()
    c_old = dict(c_new)
    mn = PWAnchorMap(c_new, ds.K, T1.H, T1.W, noise)
    mo = pa.PWAnchorMap(c_old, ds.K, T1.H, T1.W, pa.NoiseModel(noise.sn, noise.a, noise.b))
    same = True
    for i in range(10):
        _, depth, pose = ds[i * 3]
        T = pose.cpu().numpy().astype(np.float64)
        planes = ex(depth, ds.K)
        if i == 0:
            mn.update_at_keyframe_pw(0, planes, T)
            mo.update_at_keyframe_pw(0, planes, T)
            continue
        same &= mn.match_nn(planes, T) == mo.match_nn(planes, T)
    return dict(passed=bool(same), identical=bool(same), mode=mn.match_select)


# ------------------------------------------------------------------ T10
def t10():
    out = {}
    rng = np.random.default_rng(1)
    # (a) no inlier pair: identity basis, no plane gradient -> exactly the Adam path of base on the same render gradients
    dec = DecoupledPoseStep(np.zeros((0, 3)), 0.3, 0.001, 0.003, device="cpu", dtype=torch.float64)
    pt = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    pr = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([{"params": [pr], "lr": 0.003}, {"params": [pt], "lr": 0.001}])
    worst = 0.0
    for _ in range(100):
        g_t, g_r = torch.from_numpy(rng.normal(size=3)), torch.from_numpy(rng.normal(size=3))
        d_t, d_r = dec.step(g_t, g_r, torch.from_numpy(rng.normal(size=3)), torch.from_numpy(rng.normal(size=3)))
        with torch.no_grad():
            pt.zero_()
            pr.zero_()
        pt.grad, pr.grad = g_t.clone(), g_r.clone()
        opt.step()
        worst = max(worst, float((d_t - pt.detach()).abs().max()), float((d_r - pr.detach()).abs().max()))
    out["a"] = dict(passed=worst < 1e-6, max_diff=worst, note="DecoupledPoseStep with no inlier pair vs torch Adam on the render gradients (100 random steps)")
    # (b) basis from exactly the inlier pairs
    I = np.eye(4)
    m = PWAnchorMap(cfg(), T1.K, T1.H, T1.W, EFF)
    walls = [([1, 0, 0], 1.0), ([0, 1, 0], 1.5), ([0, 0, 1], 2.0), ([1, 1, 0], 1.2)]
    m.update_at_keyframe_pw(0, [T1.meas(n, d) for n, d in walls], I)
    sd = EFF.sd(2.0)
    planes = [T1.meas(n, d + (0.0 if i != 3 else min(5 * sd, 0.09)), z=2.0) for i, (n, d) in enumerate(walls)]
    act = [(i, i) for i in range(4)]
    mb = basis_normals(m, planes, act, I)
    ref = np.stack([planes[i].n.numpy() for i in range(3)])
    b1, b2 = obs_basis(mb, 0.3), obs_basis(ref, 0.3)
    out["b"] = dict(passed=len(mb) == 3 and np.allclose(mb, ref) and np.allclose(b1["sig_t"], b2["sig_t"]), n_inlier=int(len(mb)))
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


# ------------------------------------------------------------------ T11
def _loop_problem(n_kf=40, wrong=False, rng=None):
    rng = rng or np.random.default_rng(7)
    walls = T1._room()
    Tt = T1._loop_poses(n_kf)
    kfs = list(range(n_kf))
    drift = rng.normal(size=6)
    drift = drift / np.linalg.norm(drift[:3]) * 0.005
    drift[3:] = drift[3:] / np.linalg.norm(drift[3:]) * np.deg2rad(0.2)
    Twc_t = [PG.inv(T) for T in Tt]
    Twc_e = [Twc_t[0]]
    for k in kfs[1:]:
        Twc_e.append(Twc_e[-1] @ (PG.inv(Twc_t[k - 1]) @ Twc_t[k]) @ PG.se3_exp(drift))
    Te = {k: PG.inv(Twc_e[k]) for k in kfs}
    obs, planes = [], {}
    for a, (n_w, d_w) in enumerate(walls):
        for k in kfs:
            if 2 < k < n_kf - 1:
                continue
            n_c, d_c = plane_w2c(n_w, d_w, Tt[k])
            obs.append((a, k, n_c + rng.normal(0, 1e-4, 3), d_c + rng.normal(0, 1e-4), 2.0, "kf_obs"))
        planes[a] = (n_w, d_w)                     # kf 0 is undrifted: landmark = true plane
    odo = [(i, j, Te[j] @ PG.inv(Te[i])) for i, j in zip(kfs[:-1], kfs[1:])]
    closure = [i for i, o in enumerate(obs) if o[1] == kfs[-1]]
    if wrong:
        n_c, d_c = plane_w2c(walls[0][0], walls[0][1] + 0.20, Tt[kfs[-1]])
        obs.append((0, kfs[-1], n_c, d_c, 2.0, "t2:0"))
        closure.append(len(obs) - 1)
    return Tt, kfs, Te, odo, planes, obs, closure


def t11():
    out = {}
    rng = np.random.default_rng(3)
    Tt, kfs, Te, odo, planes, obs, closure = _loop_problem()          # exactly the R1 T4 problem (seed 7)
    gv = PG.GraphVec(kfs, Te, odo, planes, obs, [], PGN, CAL["sigma_odo_t"], CAL["sigma_odo_r"])
    go = PG.Graph(kfs, Te, odo, planes, obs, [], PGN, CAL["sigma_odo_t"], CAL["sigma_odo_r"])
    worst = 0.0
    for _ in range(5):
        x = rng.normal(0, 0.01, gv.n6 + 3 * len(gv.aids))
        worst = max(worst, float(np.abs(gv.residuals(x) - go.residuals(x)).max()))
    out["a"] = dict(passed=worst < 1e-9, max_diff=worst)
    err0 = np.mean([PG.pose_change(Tt[q], Te[q])[0] for q in kfs])
    res = PG.solve_ours_irls(kfs, Te, odo, planes, obs, closure, PGN, CAL["sigma_odo_t"], CAL["sigma_odo_r"], kfs[-1])
    err1 = np.mean([PG.pose_change(Tt[q], res["poses"][q])[0] for q in kfs]) if res["status"] == "applied" else None
    out["b"] = dict(passed=bool(err1 is not None and err1 <= 0.5 * err0), err_before_cm=100 * err0,
                    err_after_cm=None if err1 is None else 100 * err1, status=res["status"])
    Tt, kfs, Te, odo, planes, obs, closure = _loop_problem(wrong=True, rng=np.random.default_rng(3))
    res2 = PG.solve_ours_irls(kfs, Te, odo, planes, obs, closure, PGN, CAL["sigma_odo_t"], CAL["sigma_odo_r"], kfs[-1])
    wi = len(obs) - 1
    w_last = res2.get("closure_w", {}).get(wi)
    out["c"] = dict(passed=bool(w_last is not None and w_last < 0.2 and wi in res2.get("dropped", [])), irls_w=w_last,
                    dropped=res2.get("dropped"), status=res2["status"])
    g = PG.GraphVec(kfs, Te, odo, planes, obs, [], PGN, CAL["sigma_odo_t"], CAL["sigma_odo_r"])
    x = np.random.default_rng(5).normal(0, 0.01, g.n6 + 3 * len(g.aids))
    r1 = g.residuals(x)[:6 * g.n_odo_rows].copy()
    g.irls_w = np.full(len(obs), 0.1)
    r2 = g.residuals(x)[:6 * g.n_odo_rows]
    out["d"] = dict(passed=bool(np.array_equal(r1, r2)), note="odometry rows unchanged when plane-factor IRLS weights change")
    # (e) speed: 70 keyframes, ~300 plane factors
    rng = np.random.default_rng(11)
    Tt, kfs, Te, odo, planes, obs, closure = _loop_problem(n_kf=70, rng=rng)
    walls = T1._room()
    obs = []
    for k in kfs:
        for a in rng.choice(len(walls), 4, replace=False):
            n_c, d_c = plane_w2c(walls[a][0], walls[a][1], Tt[k])
            obs.append((int(a), k, n_c + rng.normal(0, 1e-3, 3), d_c + rng.normal(0, 3e-3), 2.0, "kf_obs"))
    obs = obs[:300]
    t0 = time.perf_counter()
    res3 = PG.solve_ours_irls(kfs, Te, odo, planes, obs, [i for i, o in enumerate(obs) if o[1] == kfs[-1]], PGN,
                              CAL["sigma_odo_t"], CAL["sigma_odo_r"], kfs[-1])
    dt = time.perf_counter() - t0
    out["e"] = dict(passed=dt <= 10.0, seconds=dt, n_kf=len(kfs), n_plane=len(obs), status=res3["status"])
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for name, fn in (("T8", t8), ("T9", t9), ("T10", t10), ("T11", t11)):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            res[name] = dict(passed=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        print(name, "PASSED" if res[name]["passed"] else "FAILED", json.dumps(res[name], default=str)[:700], flush=True)
    res["noise_used"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in CAL.items()}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
