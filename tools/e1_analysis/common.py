"""Shared loaders and metrics for the E1 read-only analysis (no file in results_exp/e1/<seq>/ is modified)."""
import csv
import json
import os

import numpy as np
from evo.core import trajectory
from evo.core.trajectory import PosePath3D
from scipy.spatial.transform import Rotation

ROOT = os.environ.get("E1_ROOT", os.path.expanduser("~/2DGS-SLAM/results_exp/e1"))
SEQ = "fr3_snt_far"
OUT = os.path.join(ROOT, "analysis")
SEGMENTS = [("S1", 0, 84), ("S2", 85, 220), ("S3", 221, 295), ("S4", 296, 480), ("S5", 481, 790)]
METHODS = ("base", "ours")
GT_C, BASE_C, OURS_C = "#0b0b0b", "#8f8e88", "#2a78d6"
INK2, GRID = "#52514e", "#e4e3df"
SEG_BG = ["#f4f3ef", "#ffffff"]          # alternating segment shading


def run_dir(method, seed="seed0"):
    return os.path.join(ROOT, SEQ, method, seed)


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float("nan") if x in ("", None) else float(x)


def load_run(method):
    """Frame ids, est and gt c2w poses (N,4,4) in tracked order, frame_log rows keyed by frame id."""
    d = run_dir(method)
    ape = read_csv(os.path.join(d, "ape_per_frame.csv"))
    ids = np.array([int(r["frame_id"]) for r in ape])

    def tum(p):
        a = np.loadtxt(p)
        T = np.tile(np.eye(4), (len(a), 1, 1))
        T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix()
        T[:, :3, 3] = a[:, 1:4]
        return a[:, 0], T

    ts_e, est = tum(os.path.join(d, "traj_est_all.txt"))
    ts_g, gt = tum(os.path.join(d, "traj_gt_all.txt"))
    ts_a = np.array([float(r["timestamp"]) for r in ape])
    assert len(ids) == len(est) == len(gt) and np.allclose(ts_e, ts_a, atol=1e-5) and np.allclose(ts_g, ts_a, atol=1e-5)
    flog = {int(r["frame_id"]): r for r in read_csv(os.path.join(d, "frame_log.csv"))}
    metrics = json.load(open(os.path.join(d, "e1_metrics.json")))
    return dict(ids=ids, est=est, gt=gt, ape=np.array([float(r["ape_m"]) for r in ape]), flog=flog, metrics=metrics)


def seg_of(fid):
    for name, a, b in SEGMENTS:
        if a <= fid <= b:
            return name
    return None


def rel_err(P_i, P_j, Q_i, Q_j):
    """evo RPE error for one pair: E = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j); returns (trans m, rot deg)."""
    E = np.linalg.inv(np.linalg.inv(Q_i) @ Q_j) @ (np.linalg.inv(P_i) @ P_j)
    ang = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(E[:3, 3])), float(ang)


def rpe_pairs(run, delta, sliding=False):
    """Pairs (i, i+delta) in tracked index; returns list of (fid_i, fid_j, trans_m, rot_deg).
    Default = evo's pairing (delta_unit frames, all_pairs=False): i = 0, delta, 2*delta, ...
    sliding=True: every i (more pairs inside short segments; not evo's definition)."""
    out = []
    for i in range(0, len(run["ids"]) - delta, 1 if sliding else delta):
        j = i + delta
        t, r = rel_err(run["est"][i], run["est"][j], run["gt"][i], run["gt"][j])
        out.append((int(run["ids"][i]), int(run["ids"][j]), t, r))
    return out


def rmse(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def local_ate(run, a, b):
    m = (run["ids"] >= a) & (run["ids"] <= b)
    if m.sum() < 3:
        return float("nan")
    ref = PosePath3D(poses_se3=list(run["gt"][m]))
    est = PosePath3D(poses_se3=list(run["est"][m]))
    est = trajectory.align_trajectory(est, ref, correct_scale=False)
    return rmse(np.linalg.norm(np.array(est.positions_xyz) - np.array(ref.positions_xyz), axis=1))


def motion_ratio(run):
    """Per consecutive tracked pair k: (fid_k, |dt_est|, |dt_gt|)."""
    pe, pg = run["est"][:, :3, 3], run["gt"][:, :3, 3]
    de = np.linalg.norm(np.diff(pe, axis=0), axis=1)
    dg = np.linalg.norm(np.diff(pg, axis=0), axis=1)
    return run["ids"][1:], de, dg


def shade_segments(ax):
    for k, (name, a, b) in enumerate(SEGMENTS):
        ax.axvspan(a, b, color=SEG_BG[k % 2], zorder=0, lw=0)
        ax.text((a + b) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=INK2)


def mpl_style(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def vn(x, fmt="{:.2f}"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return fmt.format(x).replace(".", ",")
