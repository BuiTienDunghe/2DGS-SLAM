"""PLAN MONOGS-PW §6: tests T1-T7. run from monogs/: python tests/test_pw.py [--out ../docs/monogs/pw/tests_pw.json]
Noise model from configs/rgbd/tum/pw_calib.yaml (the values the runs use). T1e, T6, T7b need datasets / GPU.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils import pgo_planes as PG  # noqa: E402
from utils.obs_decouple import DecoupledPoseStep  # noqa: E402
from utils.plane_anchor import (NoiseModel, PlaneMeas, PWAnchorMap, plane_c2w, plane_w2c,  # noqa: E402
                                pw_loss_ctx, pw_plane_loss, robust_weight)
from utils.pose_utils import SE3_exp  # noqa: E402

H, W = 480, 640
K = np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])
E1C = yaml.safe_load(open("configs/rgbd/tum/e1c.yaml"))["PlaneTracking"]
PWC = yaml.safe_load(open("configs/rgbd/tum/pw.yaml"))
CAL = yaml.safe_load(open("configs/rgbd/tum/pw_calib.yaml"))["Noise"]
NOISE = NoiseModel(CAL["sigma_n"], CAL["sigma_d_a"], CAL["sigma_d_b"])


def cfg(**kw):
    c = dict(E1C)
    c.update(PWC["PlaneTracking"])
    c.update(dict(match_mode="nn", robust=True, anchor_mode="attach"))
    c.update(kw)
    return c


def meas(n, d, pts=None, n_px=10000, z=None):
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    if pts is None:
        b1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(n, b1)
        g = np.random.default_rng(0).uniform(-0.5, 0.5, (200, 2))
        pts = -d * n + g[:, :1] * b1 + g[:, 1:] * b2
    p = PlaneMeas(n=torch.from_numpy(n), d=float(d), mask_cells=torch.ones(H // 20, W // 20, dtype=torch.bool),
                  n_px=int(n_px), mse=0.0, pts_c=np.asarray(pts, dtype=np.float64))
    if z is not None:
        p.z = z
    return p


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rot(axis, deg):
    axis = np.asarray(axis, dtype=np.float64)
    return PG.so3_exp(axis / np.linalg.norm(axis) * np.deg2rad(deg))


# ------------------------------------------------------------------ T1
def t1():
    out = {}
    m = PWAnchorMap(cfg(), K, H, W, NOISE)
    I = np.eye(4)
    m.update_at_keyframe_pw(0, [meas([0, 0, -1], 2.0)], I)                 # wall in front, active (first kf)
    a0 = 0
    # (a) two pieces of the same plane
    pa = meas([0, 0, -1], 2.0, pts=np.c_[np.random.default_rng(1).uniform(-1, -0.5, (50, 2)), np.full(50, 2.0)])
    pb = meas([0, 0, -1], 2.0, pts=np.c_[np.random.default_rng(2).uniform(0.5, 1, (50, 2)), np.full(50, 2.0)])
    pairs = m.match_nn([pa, pb], I)
    cls = m.classify([pa, pb], pairs, I)
    out["a"] = dict(passed=pairs == [(0, a0), (1, a0)] and all(c[4] for c in cls), pairs=pairs)
    # (b) parallel plane 6 cm away, inside the E1c gate
    pp = meas([0, 0, -1], 2.06)
    pr = m.match_nn([pp], I)
    c = m.classify([pp], pr, I)
    e, w = (c[0][2], c[0][3]) if c else (None, None)
    out["b"] = dict(passed=bool(pr) and e > 3 and w < 0.05 and not c[0][4], e=e, w=w, in_gate=bool(pr))
    # (c) far part of a large plane: no anchor sample in view
    far = meas([0, 0, -1], 2.0, pts=np.c_[np.random.default_rng(3).uniform(3, 4, (50, 2)), np.full(50, 2.0)])
    pr = m.match_nn([far], I)
    c = m.classify([far], pr, I)
    out["c"] = dict(passed=pr == [(0, a0)] and c[0][4], overlap_e1c=m._overlap(m.anchors[a0], far, I[:3, :3], I[:3, 3]))
    # (d) robust closed form: 5 pairs, one wrong by 6 cm
    normals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]
    ds = [2.0, 1.5, 2.5, 1.8, 2.2]
    T_gt = make_T(rot([0.3, 1, 0.2], 7), np.array([0.05, -0.03, 0.02]))
    m2 = PWAnchorMap(cfg(), K, H, W, NOISE)
    walls = []
    for n, d in zip(normals, ds):
        n = np.asarray(n, float) / np.linalg.norm(n)
        walls.append((n, d))
    m2.update_at_keyframe_pw(0, [meas(n, d) for n, d in walls], np.eye(4))  # anchors in world = camera 0
    planes = []
    for i, (n_w, d_w) in enumerate(walls):
        n_c, d_c = plane_w2c(n_w, d_w, T_gt)
        planes.append(meas(n_c, d_c + (0.06 if i == 3 else 0.0), z=2.0))
    pairs = [(i, i) for i in range(5)]
    T_prior = make_T(rot([1, 0, 0], 1.0) @ T_gt[:3, :3], T_gt[:3, 3] + 0.01)
    T, n_drop = m2.solve_pose_robust(planes, pairs, torch.from_numpy(T_prior))
    err = None if T is None else float(np.linalg.norm(PG.pose_change(T_gt, T.numpy())[0]))
    T_e1c = m2.solve_pose(planes, pairs, torch.from_numpy(T_prior))
    err_e1c = None if T_e1c is None else PG.pose_change(T_gt, T_e1c.numpy())[0]
    out["d"] = dict(passed=T is not None and n_drop == 1 and err < 0.005, n_dropped=n_drop, err_m=err, err_e1c_m=err_e1c)
    # (e) e1c mode identical to the old code on 10 real frames of snt_far
    out["e"] = t1e()
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def t1e():
    old = "/tmp/plane_anchor_b9099a3.py"
    subprocess.run(f"git show b9099a3:monogs/utils/plane_anchor.py > {old}", shell=True, check=True, cwd=ROOT)
    spec = importlib.util.spec_from_file_location("pa_old", old)
    pa_old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa_old)
    from utils import plane_anchor as pa_new
    from munch import munchify
    from utils.config_utils import load_config
    from utils.dataset import load_dataset
    c = load_config("configs/rgbd/tum/fr3_snt_far_sp.yaml")
    mp_ = munchify(c["model_params"])
    ds = load_dataset(mp_, mp_.source_path, config=c)
    ex = pa_new.PlaneExtractor(E1C)
    mo, mn = pa_old.PlaneAnchorMap(E1C, ds.K, H, W), pa_new.PlaneAnchorMap(E1C, ds.K, H, W)
    same = True
    worst = 0.0
    for i in range(10):
        _, depth, pose = ds[i]
        T = torch.from_numpy(pose.cpu().numpy().astype(np.float64)) if torch.is_tensor(pose) else torch.from_numpy(pose)
        planes = ex(depth, ds.K)
        if i == 0:
            mo.update_at_keyframe(0, planes, [], T)
            mn.update_at_keyframe(0, planes, [], T)
            continue
        po, pn = mo.match(planes, T), mn.match(planes, T)
        same &= po == pn
        lo, ln = mo.plane_loss(T, planes, po), mn.plane_loss(T, planes, pn)
        worst = max(worst, abs(float(lo) - float(ln)))
        if i % 5 == 0:
            mo.update_at_keyframe(i, planes, po, T)
            mn.update_at_keyframe(i, planes, pn, T)
    return dict(passed=bool(same) and worst == 0.0, match_identical=bool(same), max_loss_diff=worst)


# ------------------------------------------------------------------ T2
def t2():
    out = {}
    rng = np.random.default_rng(5)
    m = PWAnchorMap(cfg(), K, H, W, NOISE)
    T0 = make_T(rot([0, 1, 0], 10), np.array([0.1, 0, 0]))
    m.update_at_keyframe_pw(0, [meas([0, 0, -1], 2.0), meas([1, 0, 0], 1.0)], T0)
    Tn = PG.se3_exp(rng.normal(size=6) * 0.2) @ T0
    m.refresh({0: Tn})
    worst = 0.0
    for a in m.anchors.values():
        n_w, d_w = plane_c2w(a.obs_c[0], a.obs_c[1], Tn)
        worst = max(worst, float(np.abs(a.n_w.numpy() - n_w / np.linalg.norm(n_w)).max()), abs(a.d_w - d_w))
    out["a"] = dict(passed=worst < 1e-9, max_err=worst)
    # (b) an extra observation does not change the anchor value
    a = m.anchors[0]
    n0, d0 = a.n_w.numpy().copy(), a.d_w
    n_c, d_c = plane_w2c(n0, d0, Tn)
    m.update_at_keyframe_pw(5, [meas(n_c, d_c)], Tn)
    out["b"] = dict(passed=np.allclose(a.n_w.numpy(), n0, atol=0) and a.d_w == d0 and a.n_obs_kf == 2, n_obs_kf=a.n_obs_kf)
    # (c) candidates deleted after 2 keyframes when not active
    m3 = PWAnchorMap(cfg(), K, H, W, NOISE)
    I = np.eye(4)
    m3.update_at_keyframe_pw(0, [meas([0, 0, -1], 2.0)], I)
    m3.update_at_keyframe_pw(5, [meas([1, 0, 0], 1.0)], I)       # candidate id 1
    alive1 = 1 in m3.anchors
    m3.update_at_keyframe_pw(10, [meas([0, 0, -1], 2.0)], I)
    alive2 = 1 in m3.anchors
    m3.update_at_keyframe_pw(15, [meas([0, 0, -1], 2.0)], I)
    alive3 = 1 in m3.anchors
    out["c"] = dict(passed=alive1 and alive2 and not alive3, alive=[alive1, alive2, alive3])
    # (d) two coplanar unmatched pieces at the same keyframe -> one candidate
    m4 = PWAnchorMap(cfg(), K, H, W, NOISE)
    m4.update_at_keyframe_pw(0, [meas([0, 0, -1], 2.0)], I)
    n_before = len(m4.anchors)
    pa = meas([1, 0, 0], 1.0, pts=np.c_[np.full(30, -1.0), np.random.default_rng(1).uniform(-1, -0.5, (30, 2))], n_px=8000)
    pb = meas([1, 0, 0], 1.0, pts=np.c_[np.full(30, -1.0), np.random.default_rng(2).uniform(0.5, 1, (30, 2))], n_px=5000)
    _, created = m4.update_at_keyframe_pw(5, [pa, pb], I)
    out["d"] = dict(passed=len(m4.anchors) - n_before == 1 and created[0] == created[1], created=created)
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


# ------------------------------------------------------------------ T3
def _old_map():
    """Anchors 0 (wall x, seen at kf 0) and 1 (floor), kf_order 0..12 (window = last 8)."""
    m = PWAnchorMap(cfg(), K, H, W, NOISE)
    I = np.eye(4)
    m.update_at_keyframe_pw(0, [meas([1, 0, 0], 1.0), meas([0, 1, 0], 1.5)], I)
    for k in range(1, 13):
        m.kf_order.append(k)
    return m


def t3():
    out = {}
    I = np.eye(4)
    win = list(range(12, 4, -1))
    m = _old_map()
    p = meas([1, 0, 0], 1.20)                  # 20 cm drift: outside the E1c gate, inside the wide closure gate
    cls = m.classify([p], m.match_nn([p], I), I)
    t1_, t2_ = m.closure_candidates([p], cls, I, win, 12)
    out["a"] = dict(passed=t1_ == [] and t2_ == [(0, 0)], tier1=t1_, tier2=t2_)
    m2 = PWAnchorMap(cfg(), K, H, W, NOISE)
    m2.update_at_keyframe_pw(0, [meas([1, 0, 0], 1.0), meas([0, 1, 0], 1.5)], I)
    m2.update_at_keyframe_pw(1, [meas([1, 0, 0], 1.25)], I)        # second anchor of the same wall (drift)
    a2 = max(m2.anchors)
    m2.anchors[a2].active = True                                    # make it a confirmed (old) anchor
    for k in range(2, 13):
        m2.kf_order.append(k)
    p = meas([1, 0, 0], 1.12)
    cls = m2.classify([p], m2.match_nn([p], I), I)
    _, t2b = m2.closure_candidates([p], cls, I, win, 12)
    out["b"] = dict(passed=t2b == [], tier2=t2b)
    m3 = _old_map()
    a = m3.anchors[0]
    old_far = m3.is_old(a, 12, win)
    a.kf_obs.append(dict(kf=8, n_c=np.array([1.0, 0, 0]), d_c=1.0, n_px=1, z=1.0, pts_c=None))
    a.last_obs_kf = 8
    in_win = m3.is_old(a, 12, win)
    a.kf_obs = a.kf_obs[:1]
    a.last_obs_kf = 6
    gap_small = m3.is_old(a, 12, list(range(12, 8, -1)))
    out["c"] = dict(passed=old_far and not in_win and not gap_small, old=old_far, window_obs=in_win, gap6=gap_small)
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


# ------------------------------------------------------------------ T4
def _room():
    walls = [([1, 0, 0], 2.0), ([-1, 0, 0], 2.0), ([0, 1, 0], 2.0), ([0, -1, 0], 2.0), ([0, 0, 1], 1.0), ([0, 0, -1], 1.5)]
    return [(np.asarray(n, float), float(d)) for n, d in walls]


def _loop_poses(n=40):
    Ts = []
    for k in range(n):
        a = 2 * np.pi * k / n
        c = np.array([0.8 * np.cos(a), 0.8 * np.sin(a), 0.0])
        R = rot([0, 0, 1], np.degrees(a))                     # camera->world rotation
        Ts.append(PG.inv(make_T(R, c)))                         # w2c
    return Ts


def t4():
    out = {}
    rng = np.random.default_rng(7)
    walls = _room()
    Tt = _loop_poses()
    kfs = list(range(len(Tt)))
    drift = rng.normal(size=6)
    drift = drift / np.linalg.norm(drift[:3]) * 0.005
    drift[3:] = drift[3:] / np.linalg.norm(drift[3:]) * np.deg2rad(0.2)
    # odometry drift: every relative motion carries the same body-frame error (0.5 cm, 0.2 deg)
    Twc_t = [PG.inv(T) for T in Tt]
    Twc_e = [Twc_t[0]]
    for k in kfs[1:]:
        Twc_e.append(Twc_e[-1] @ (PG.inv(Twc_t[k - 1]) @ Twc_t[k]) @ PG.se3_exp(drift))
    Te = {k: PG.inv(Twc_e[k]) for k in kfs}                    # drifted estimate (w2c)
    obs = []
    planes = {}
    for a, (n_w, d_w) in enumerate(walls):
        for k in kfs:
            n_c, d_c = plane_w2c(n_w, d_w, Tt[k])
            if d_c <= 0:
                continue
            if k > 2 and k < len(kfs) - 1:            # observed at the start and at the closing keyframe only
                continue
            obs.append((a, k, n_c + rng.normal(0, 1e-4, 3), d_c + rng.normal(0, 1e-4), 2.0, "kf_obs"))
        first = min(o[1] for o in obs if o[0] == a)
        n0, d0 = plane_w2c(n_w, d_w, Tt[first])
        planes[a] = plane_c2w(n0, d0, Te[first])
    # (a) convention: residual zero at the true poses and planes
    worst = 0.0
    for a, k, m_, d_, z, _ in obs:
        r = PG.plane_residual(*plane_w2c(walls[a][0], walls[a][1], Tt[k]), walls[a][0], walls[a][1], Tt[k])
        worst = max(worst, float(np.abs(r).max()))
    out["a"] = dict(passed=worst < 1e-6, max_residual=worst)
    odo = [(i, j, Te[j] @ PG.inv(Te[i])) for i, j in zip(kfs[:-1], kfs[1:])]
    k = kfs[-1]
    closure = [i for i, o in enumerate(obs) if o[1] == k]
    err0 = np.mean([PG.pose_change(Tt[q], Te[q])[0] for q in kfs])
    res = PG.solve_ours(kfs, Te, odo, planes, obs, closure, NOISE, CAL["sigma_odo_t"], CAL["sigma_odo_r"], k)
    err1 = np.mean([PG.pose_change(Tt[q], res["poses"][q])[0] for q in kfs]) if res["status"] == "applied" else None
    out["b"] = dict(passed=bool(err1 is not None and err1 <= 0.5 * err0), err_before_cm=100 * err0,
                    err_after_cm=None if err1 is None else 100 * err1, status=res["status"])
    # (c) wrong tier-2 pair: a parallel plane 20 cm away attached to wall 0 at the closing keyframe
    n_c, d_c = plane_w2c(walls[0][0], walls[0][1] + 0.20, Tt[k])
    obs2 = obs + [(0, k, n_c, d_c, 2.0, "t2:0")]
    res2 = PG.solve_ours(kfs, Te, odo, planes, obs2, closure + [len(obs2) - 1], NOISE, CAL["sigma_odo_t"],
                         CAL["sigma_odo_r"], k)
    out["c"] = dict(passed=(len(obs2) - 1) in res2.get("dropped", []), dropped=res2.get("dropped"), status=res2["status"])
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


# ------------------------------------------------------------------ T5
def t5():
    rng = np.random.default_rng(11)
    T_true = make_T(rot([0.2, 1, 0.1], 8), np.array([0.06, -0.04, 0.03]))     # x_j = R x_k + t
    planes_k = [(np.array([0, 0, -1.0]), 2.0), (np.array([1.0, 0, 0]), 1.2)]
    pk, pj, src, dst, nrm = [], [], [], [], []
    for n, d in planes_k:
        b1 = np.cross(n, [0, 1.0, 0])
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(n, b1)
        g = rng.uniform(-0.6, 0.6, (400, 2))
        P = -d * n + g[:, :1] * b1 + g[:, 1:] * b2
        nj = T_true[:3, :3] @ n
        dj = d - nj @ T_true[:3, 3]
        pk.append((n, d))
        pj.append((nj, dj))
        src.append(P + rng.normal(0, 0.002, P.shape))
        Pj = P @ T_true[:3, :3].T + T_true[:3, 3]
        dst.append(Pj + rng.normal(0, 0.002, P.shape))
        nrm.append(np.repeat(nj[None], len(P), 0))
    T0 = PG.relpose_from_planes(pk, pj)
    T1, fit, rmse = PG.icp_refine(T0, np.concatenate(src), np.concatenate(dst), np.concatenate(nrm))
    dt, dr = PG.pose_change(PG.inv(T_true), PG.inv(T1))
    line = np.cross(pj[0][0], pj[1][0])
    line /= np.linalg.norm(line)
    return dict(passed=dt < 0.01 and dr < 0.5, trans_err_cm=100 * dt, rot_err_deg=dr,
                true_t_along_intersection_cm=100 * float(abs(T_true[:3, 3] @ line)), icp_fitness=fit)


# ------------------------------------------------------------------ T6 (GPU)
def _map_and_cams():
    from munch import munchify
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    from utils.camera_utils import Camera
    from utils.config_utils import load_config
    from utils.dataset import load_dataset
    from utils.slam_utils import get_loss_mapping
    torch.manual_seed(0)
    np.random.seed(0)
    c = load_config("configs/rgbd/tum/fr3_snt_far_sp.yaml")
    c["Training"]["monocular"] = False
    mp_ = munchify(c["model_params"])
    mp_.sh_degree = 0
    ds = load_dataset(mp_, mp_.source_path, config=c)
    proj = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=ds.fx, fy=ds.fy, cx=ds.cx, cy=ds.cy, W=ds.width,
                                H=ds.height).transpose(0, 1).cuda()
    g = GaussianModel(0, config=c)
    g.init_lr(6.0)
    g.training_setup(munchify(c["opt_params"]))
    pipe = munchify(c["pipeline_params"])
    bg = torch.zeros(3, device="cuda")
    cam0 = Camera.init_from_dataset(ds, 0, proj)
    cam0.update_RT(cam0.R_gt.float(), cam0.T_gt.float())
    cam0.compute_grad_mask(c)
    g.extend_from_pcd_seq(cam0, kf_id=0, init=True, scale=2.0, depthmap=cam0.depth.copy())
    for _ in range(500):
        pkg = render(cam0, g, pipe, bg)
        get_loss_mapping(c, pkg["render"], pkg["depth"], cam0, pkg["opacity"], initialization=True).backward()
        with torch.no_grad():
            g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
    return c, ds, proj, g, pipe, bg, cam0, render


def t6(shared):
    from gaussian_splatting.utils.general_utils import build_rotation
    from utils.slam_backend import apply_pgo_to_gaussians
    c, ds, proj, g, pipe, bg, cam0, render = shared
    with torch.no_grad():
        r_old = render(cam0, g, pipe, bg)
        d_old, o_old = r_old["depth"].clone(), r_old["opacity"].clone()
        xyz0 = g._xyz.data.clone().double()
        R0 = build_rotation(g._rotation.data.clone()).double()
        T_old = torch.eye(4, device="cuda", dtype=torch.float64)
        T_old[:3, :3], T_old[:3, 3] = cam0.R.double(), cam0.T.double()
        Tn = torch.from_numpy(PG.se3_exp(np.array([0.05, -0.02, 0.03, 0.02, -0.03, 0.01]))).cuda() @ T_old
        apply_pgo_to_gaussians(g, {0: cam0}, {0: (Tn[:3, :3].float(), Tn[:3, 3].float())})
        Wm = torch.linalg.inv(Tn) @ T_old
        e_xyz = float((g._xyz.data.double() - (xyz0 @ Wm[:3, :3].T + Wm[:3, 3])).abs().max())
        e_rot = float((build_rotation(g._rotation.data).double() - Wm[:3, :3] @ R0).abs().max())
        r_new = render(cam0, g, pipe, bg)
        d_new = r_new["depth"]
        valid = (d_old > 0) & (d_new > 0) & (o_old > 0.95) & (r_new["opacity"] > 0.95)   # MonoGS valid-depth rule
        diff = (d_old - d_new)[valid].abs()
        e_d = float(diff.max())
        anyv = (d_old > 0) & (d_new > 0)
        e_any = (d_old - d_new)[anyv].abs()
    return dict(passed=e_xyz < 1e-5 and e_rot < 1e-5 and e_d < 1e-3, a_xyz_err=e_xyz, a_rot_err=e_rot,
                b_depth_err_m=e_d, b_depth_p99_m=float(diff.quantile(0.99)), b_valid_px=int(valid.sum()),
                b_any_depth_max_m=float(e_any.max()), b_any_depth_p99_m=float(e_any.quantile(0.99)),
                note="(a) xyz stored in float32: tolerance 1e-5; (b) valid = depth > 0 and opacity > 0.95 in both renders")


# ------------------------------------------------------------------ T7
def t7(shared):
    out = {}
    from utils.render_stiffness import render_stiffness
    from utils.camera_utils import Camera
    c, ds, proj, g, pipe, bg, cam0, render = shared
    R0, T0 = cam0.R.clone(), cam0.T.clone()
    render_stiffness(cam0, g, pipe, bg, c)
    out["a"] = dict(passed=bool(torch.equal(R0, cam0.R) and torch.equal(T0, cam0.T)))
    deltas = []
    for i in range(1, 11):
        v = Camera.init_from_dataset(ds, i, proj)
        v.compute_grad_mask(c)
        v.update_RT(v.R_gt.float(), v.T_gt.float())
        cr, L0, _ = render_stiffness(v, g, pipe, bg, c)
        deltas.append((cr * np.array([0.01] * 3 + [np.deg2rad(0.5)] * 3) ** 2 / 2).tolist())
    dmin = float(np.min(deltas))
    out["b"] = dict(passed=dmin >= 0, min_delta=dmin)
    # (c) two quadratic springs through DecoupledPoseStep
    a_, b_, xa, xb = 4.0, 1.0, 0.03, -0.02
    dec = DecoupledPoseStep(np.eye(3), 0.3, 0.001, 0.003, device="cpu", dtype=torch.float64)
    x = torch.zeros(3, dtype=torch.float64)
    for _ in range(300):
        g_r = torch.tensor([a_ * (x[0] - xa), 0, 0], dtype=torch.float64)
        g_p = torch.tensor([b_ * (x[0] - xb), 0, 0], dtype=torch.float64)
        d_t, _ = dec.step(g_r, torch.zeros(3, dtype=torch.float64), g_p, torch.zeros(3, dtype=torch.float64))
        x = x + d_t                       # ManualAdam returns the update itself (already -lr m/sqrt(v))
    target = (a_ * xa + b_ * xb) / (a_ + b_)
    out["c"] = dict(passed=abs(float(x[0]) - target) <= 0.05 * abs(target), x=float(x[0]), target=target)
    # (d) kappa estimator on synthetic data
    rng = np.random.default_rng(3)
    kappa = 37.0
    cc = rng.uniform(10, 1000, 20000)
    eps = rng.normal(0, np.sqrt(2 / (kappa * cc)))
    est = 2 * 0.455 / np.median(eps ** 2 * cc)
    out["d"] = dict(passed=abs(est - kappa) / kappa < 0.1, kappa_true=kappa, kappa_est=float(est))
    # (e) 1b loss against the formula on 3 synthetic pairs
    m = PWAnchorMap(cfg(), K, H, W, NOISE)
    I = np.eye(4)
    walls = [([1, 0, 0], 1.0), ([0, 1, 0], 1.5), ([0, 0, 1], 2.0)]
    m.update_at_keyframe_pw(0, [meas(n, d) for n, d in walls], I)
    T = make_T(rot([1, 1, 0], 2), np.array([0.02, -0.01, 0.03]))
    planes = [meas(*plane_w2c(np.asarray(n, float), d, I), z=1.0 + i) for i, (n, d) in enumerate(walls)]
    ctx = pw_loss_ctx(planes, [(i, i) for i in range(3)], m.anchors, NOISE, "cpu", torch.float64, 0.02, 0.01)
    kap = 5.0
    l, _ = pw_plane_loss(torch.from_numpy(T), ctx, True, True, kap)
    ref = 0.0
    for i, (n, d) in enumerate(walls):
        n = np.asarray(n, float)
        Rn = T[:3, :3] @ n
        rn = planes[i].n.numpy() - Rn
        rd = planes[i].d - (d - Rn @ T[:3, 3])
        e2 = rn @ rn / NOISE.sn ** 2 + rd ** 2 / NOISE.sd(1.0 + i) ** 2
        ref += float(robust_weight(np.sqrt(e2))) * 0.5 * e2
    ref /= kap
    out["e"] = dict(passed=abs(float(l) - ref) < 1e-9 * max(1.0, abs(ref)), loss=float(l), formula=ref)
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()
    res = {}
    for name, fn in (("T1", t1), ("T2", t2), ("T3", t3), ("T4", t4), ("T5", t5)):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            res[name] = dict(passed=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        print(name, "PASSED" if res[name]["passed"] else "FAILED", json.dumps(res[name], default=str)[:600], flush=True)
    if not a.skip_gpu:
        try:
            shared = _map_and_cams()
        except Exception as e:  # noqa: BLE001
            shared = None
            res["T6"] = res["T7"] = dict(passed=False, error=f"setup: {e}", trace=traceback.format_exc()[-2000:])
        if shared is not None:
            for name, fn in (("T7", t7), ("T6", t6)):          # T7a first: T6 moves the Gaussians
                try:
                    res[name] = fn(shared)
                except Exception as e:  # noqa: BLE001
                    res[name] = dict(passed=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
                print(name, "PASSED" if res[name]["passed"] else "FAILED", json.dumps(res[name], default=str)[:600], flush=True)
    res["noise_used"] = CAL
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
