"""PLAN A2G-P1 gen, Step 0 (tools only; GT is read here and never in utils/pw).

Shared helpers: config/dataset loading as the runs do it, run logs, SE(3) conventions of the plan (§2.3):
  T_cw world->camera; tracking left update T <- Exp(xi) T, xi = [rho, theta];
  world-frame pose error  T_hat_cw = T_cw Exp(eps),  xi = Ad(T_hat_cw) eps,  Exp(eps)X ~ X + rho + theta x X;
  true error (tools only) eps_k = Log(T_cw,k^-1 T_hat_cw,k), no trajectory alignment.
usage (self test): python tools/a2g_gen/common.py
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gaussian_splatting"))

from utils.pw import geometry as G  # noqa: E402
from utils.pw.uncertainty import jac_xi, residual  # noqa: E402
from utils.pw.world import np_pose  # noqa: E402

RX = "results_exp"
OUT0 = os.path.join(RX, "a2g_p1gen", "stage0")

SEQ_CFG = {
    "snt_near": "configs/tum/a2g_p1/fr3_snt_near.yaml",
    "snt_far": "configs/tum/a2g_p1/fr3_snt_far.yaml",
    "st_far": "configs/tum/a2g_p1/fr3_st_far.yaml",
    "room": "configs/tum/a2g_p1/fr1_room.yaml",
}

# PLAN §3.1 (runs reused, never re-run)
RUNS = {
    "snt_near": {"dec": f"{RX}/e1f/fr3_snt_near/dec/seed0"},
    "snt_far": {"base": f"{RX}/e1/fr3_snt_far/base/seed0", "dec": f"{RX}/e1c/fr3_snt_far/dec/seed0",
                "core": f"{RX}/a2g_p1core/fr3_snt_far/p1core/seed0"},
    "st_far": {"base": f"{RX}/e1f/fr3_st_far/base/seed0", "dec": f"{RX}/e1f/fr3_st_far/dec/seed0",
               "core": f"{RX}/a2g_p1core/fr3_st_far/p1core/seed0"},
    "room": {"base": f"{RX}/e1f/fr1_room/base/seed0", "dec": f"{RX}/e1f/fr1_room/dec/seed0",
             "core": f"{RX}/a2g_p1core/fr1_room/p1core/seed0"},
}


# ------------------------------------------------------------------ config / dataset


def load_cfg(seq):
    from utils.config_utils import load_config
    return load_config(SEQ_CFG[seq])


def load_ds(cfg):
    from munch import munchify
    from utils.dataset import load_dataset
    ds = load_dataset(munchify(cfg["model_params"]), "", config=cfg)
    ds.device = "cpu"
    return ds


def frame_ids(cfg, ds):
    return list(range(int(cfg["Dataset"]["frame_begin"]), len(ds), int(cfg["Dataset"]["frame_step"])))


def read_frame(ds, f, undistort_depth=False):
    """(gray HxW float64 in [0,1], depth HxW float64 in m, T_cw GT 4x4) exactly as the loader gives them
    (RGB undistorted when calibration says distorted; depth as stored unless undistort_depth)."""
    import cv2
    img, _, depth, pose = ds[f]
    img = img.double().numpy()
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    d = depth.double().numpy()
    if undistort_depth and getattr(ds, "disorted", False):
        d = cv2.remap(d.astype(np.float32), ds.map1x, ds.map1y, cv2.INTER_NEAREST).astype(np.float64)
    return gray, d, pose.double().numpy()


# ------------------------------------------------------------------ run logs


def load_tum(path):
    from scipy.spatial.transform import Rotation
    out = []
    for line in open(path):
        v = line.split()
        if len(v) < 8 or line.startswith("#"):
            continue
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([float(x) for x in v[4:8]]).as_matrix()
        T[:3, 3] = [float(x) for x in v[1:4]]
        out.append(T)
    return out


def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def parse_pairs(s):
    if not s:
        return []
    return [tuple(int(x) for x in p.split(":")) for p in s.split(";") if ":" in p]


def load_run(run):
    """Per tracked frame (row of frame_log): frame id, is_kf, pose pairs (s, anchor id), T_hat_cw (rotation
    projected on SO(3)), T_cw GT, true eps (6)."""
    rows = list(csv.DictReader(open(os.path.join(run, "frame_log.csv"))))
    est = load_tum(os.path.join(run, "traj_est_all.txt"))
    gt = load_tum(os.path.join(run, "traj_gt_all.txt"))
    assert len(est) == len(rows) == len(gt), (run, len(est), len(rows), len(gt))
    Th = [np_pose(G.inv_T(np_pose(T))) for T in est]
    Tg = [G.inv_T(np_pose(T)) for T in gt]
    eps = np.array([G.se3_log(G.inv_T(Tg[i]) @ Th[i]) for i in range(len(rows))])
    R = dict(dir=run, fids=[int(r["frame_id"]) for r in rows], is_kf=np.array([int(r["is_kf"]) for r in rows], bool),
             pairs=[parse_pairs(r.get("pairs", "")) for r in rows], Th=Th, Tg=Tg, eps=eps)
    R["has_planes"] = os.path.exists(os.path.join(run, "anchors_kf.jsonl"))
    return R


# ------------------------------------------------------------------ SE(3) of the plan


def Ad(T):
    """Adjoint for xi = [rho, theta]: Exp(Ad(T) x) T = T Exp(x)."""
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = G.skew(t) @ R
    A[3:, 3:] = R
    return A


def J_eps(n_a, p_a, n_c, B_c, T_hat):
    """Jacobian of r(s | anchor) w.r.t. the world-frame pose error eps (PLAN M2): J_xi Ad(T_hat)."""
    return jac_xi(n_a, p_a, n_c, B_c) @ Ad(T_hat)


def rel_cov(V_a, V_b, C_ab):
    """Cov(eps_a - eps_b) = C[a,a] + C[b,b] - C[a,b] - C[b,a]."""
    S = V_a + V_b - C_ab - C_ab.T
    return 0.5 * (S + S.T)


# ------------------------------------------------------------------ self test


def _selftest():
    rng = np.random.default_rng(0)
    ok = True
    # Ad identity
    T = G.se3_exp(rng.normal(size=6) * [0.5, 0.5, 0.5, 0.4, 0.4, 0.4])
    x = rng.normal(size=6) * 1e-3
    e1 = G.se3_exp(Ad(T) @ x) @ T
    e2 = T @ G.se3_exp(x)
    err = np.abs(e1 - e2).max()
    print(f"Ad: max |Exp(Ad x)T - T Exp(x)| = {err:.2e}")
    ok &= err < 1e-12
    # J_eps vs finite differences: residual of segment a vs anchor (world) taken into the camera of T Exp(eps)
    worst = 0.0
    for _ in range(20):
        T = G.se3_exp(rng.normal(size=6) * [0.5, 0.5, 0.5, 0.4, 0.4, 0.4])
        n_w = rng.normal(size=3)
        n_w /= np.linalg.norm(n_w)
        o_w = rng.normal(size=3)
        B_w = G.basis(n_w)
        n_c, o_c, B_c = G.to_camera(T, n_w, o_w, B_w)
        n_a = n_c + rng.normal(size=3) * 0.02
        n_a /= np.linalg.norm(n_a)
        p_a = o_c + B_c @ rng.normal(size=2) * 0.5 + n_c * 0.01

        def r_of(eps):
            nc, oc, Bc = G.to_camera(T @ G.se3_exp(eps), n_w, o_w, B_w)
            return residual(n_a, p_a, nc, oc, Bc)

        J = J_eps(n_a, p_a, n_c, B_c, T)
        h = 1e-6
        Jn = np.stack([(r_of(h * np.eye(6)[i]) - r_of(-h * np.eye(6)[i])) / (2 * h) for i in range(6)], 1)
        worst = max(worst, np.abs(J - Jn).max() / max(np.abs(Jn).max(), 1e-12))
    print(f"J_eps = J_xi Ad vs finite differences: worst relative error {worst:.2e}")
    ok &= worst < 1e-6
    # pose sanity on a real run: frame 0 est == GT, eps_0 = 0, TUM c2w inverted correctly
    for seq, runs in RUNS.items():
        for m, d in runs.items():
            if not os.path.exists(os.path.join(d, "frame_log.csv")):
                print(f"missing {d}")
                ok = False
                continue
            R = load_run(d)
            e0 = np.abs(R["eps"][0]).max()
            med = np.median(np.linalg.norm(R["eps"][:, :3], axis=1))
            print(f"{seq}/{m}: {len(R['fids'])} frames, {int(R['is_kf'].sum())} kf, |eps_0| {e0:.1e}, "
                  f"median |rho| {100 * med:.1f} cm, planes {R['has_planes']}")
            ok &= e0 < 1e-5
    cfg = load_cfg("snt_near")
    ds = load_ds(cfg)
    R = load_run(RUNS["snt_near"]["dec"])
    _, _, Tgt = read_frame(ds, R["fids"][5])
    err = np.abs(Tgt - R["Tg"][5]).max()
    print(f"dataset GT pose (loader, T_cw) vs traj_gt_all frame {R['fids'][5]}: max diff {err:.2e}")
    ok &= err < 1e-4
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
