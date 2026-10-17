"""PLAN E1c P4: per-run metrics for the whole sequence and segments S1–S5 (read-only on run folders).

usage: python tools/e1c_analysis/seg_metrics.py RUN_DIR [--out OUT_DIR] [--basis BASIS_NPZ]
  RUN_DIR has traj_est_all.txt, traj_gt_all.txt, ape_per_frame.csv, frame_log.csv, e1_metrics.json (E1 layout).
  OUT_DIR defaults to RUN_DIR, except under results_exp/e1/ (read-only): then results_exp/e1c/reused/<method>_<seed>/.
  BASIS_NPZ = V_scene cache from scene_basis.py (same for every method).

Metrics (PLAN E1c §8, ANALYSIS): ATE-all, ATE-kf (from e1_metrics.json), ATE local (SE(3) per segment), RPE Δ1,
RPE Δ15 (evo pairing and sliding), path ratio, r_k, unaligned error start/end/min/max, e_free / e_obs.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "tools", "e1_analysis"))
from common import (BASE_C, INK2, SEGMENTS, fnum, local_ate, mpl_style, read_csv, rmse,  # noqa: E402
                    rel_err, shade_segments)
from evo.core import trajectory  # noqa: E402
from evo.core.trajectory import PosePath3D  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--out", default=None)
ap.add_argument("--basis", default=os.path.expanduser("~/2DGS-SLAM/results_exp/e1c/scene_basis_fr3_snt_far.npz"))
a = ap.parse_args()
run_dir = os.path.normpath(a.run_dir)
parts = run_dir.split(os.sep)
method, seed = parts[-2], parts[-1]
out = a.out
if out is None:
    out = run_dir
    if f"{os.sep}results_exp{os.sep}e1{os.sep}" in run_dir + os.sep:
        out = os.path.join(os.path.expanduser("~/2DGS-SLAM/results_exp/e1c/reused"), f"{method}_{seed}")
os.makedirs(out, exist_ok=True)


def tum(p):
    x = np.loadtxt(p)
    T = np.tile(np.eye(4), (len(x), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(x[:, 4:8]).as_matrix()
    T[:, :3, 3] = x[:, 1:4]
    return x[:, 0], T


ape = read_csv(os.path.join(run_dir, "ape_per_frame.csv"))
ids = np.array([int(r["frame_id"]) for r in ape])
_, est = tum(os.path.join(run_dir, "traj_est_all.txt"))
_, gt = tum(os.path.join(run_dir, "traj_gt_all.txt"))
flog = {int(r["frame_id"]): r for r in read_csv(os.path.join(run_dir, "frame_log.csv"))}
m1 = json.load(open(os.path.join(run_dir, "e1_metrics.json")))
run = dict(ids=ids, est=est, gt=gt)


def pairs(delta, sliding):
    o = []
    for i in range(0, len(ids) - delta, 1 if sliding else delta):
        t, r = rel_err(est[i], est[i + delta], gt[i], gt[i + delta])
        o.append((ids[i], ids[i + delta], t, r))
    return o


P = {("d1", False): pairs(1, False), ("d15", False): pairs(15, False), ("d15", True): pairs(15, True)}
raw = np.linalg.norm(est[:, :3, 3] - gt[:, :3, 3], axis=1) * 100
de = np.linalg.norm(np.diff(est[:, :3, 3], axis=0), axis=1)
dg = np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1)

# e_k: translation of dT_est * dT_gt^-1 with dT = T_k T_{k-1}^-1 (w2c), i.e. in camera frame k
w_est = np.linalg.inv(est)
w_gt = np.linalg.inv(gt)
e_k = np.full((len(ids), 3), np.nan)
for k in range(1, len(ids)):
    dE = w_est[k] @ np.linalg.inv(w_est[k - 1])
    dG = w_gt[k] @ np.linalg.inv(w_gt[k - 1])
    e_k[k] = (dE @ np.linalg.inv(dG))[:3, 3]

basis = None
if os.path.exists(a.basis):
    B = np.load(a.basis)
    bid = {int(f): i for i, f in enumerate(B["frame_ids"])}
    basis = dict(V=B["Vt"], sig=B["sig_t"], thr=float(B["threshold"]), bid=bid)


def e_split(k, bs):
    """(|V_free^T e_k|, |V_obs^T e_k|) with the basis bs of frame k (V_scene or V_track)."""
    if bs is None or int(ids[k]) not in bs["bid"] or np.isnan(e_k[k, 0]):
        return np.nan, np.nan
    j = bs["bid"][int(ids[k])]
    V, s = bs["V"][j], bs["sig"][j]
    obs = s >= bs["thr"]
    proj = V.T @ e_k[k]
    return float(np.linalg.norm(proj[~obs])), float(np.linalg.norm(proj[obs]))


# V_track: the basis the run actually used (dec_log.csv of dec / dec_edge runs)
track = None
dl = os.path.join(run_dir, "dec_log.csv")
if os.path.exists(dl):
    rows_ = read_csv(dl)
    track = dict(V=np.array([[float(r[f"Vt{i}"]) for i in range(9)] for r in rows_]).reshape(-1, 3, 3),
                 sig=np.array([[float(r[f"sig_t{i}"]) for i in range(3)] for r in rows_]),
                 thr=basis["thr"] if basis is not None else 0.3,
                 bid={int(r["frame_id"]): i for i, r in enumerate(rows_)})

ES = np.array([e_split(k, basis) for k in range(len(ids))])
ET = np.array([e_split(k, track) for k in range(len(ids))])


def block(a_, b_):
    s = (ids >= a_) & (ids <= b_)
    o = dict(frames=[int(a_), int(min(b_, ids.max()))], n_tracked=int(s.sum()))
    for (dk, sl), pp in P.items():
        sel = [x for x in pp if a_ <= x[0] <= b_ and a_ <= x[1] <= b_]
        tag = dk + ("_slide" if sl else "")
        o[f"rpe_trans_cm_{tag}"] = 100 * rmse([x[2] for x in sel])
        o[f"rpe_rot_deg_{tag}"] = rmse([x[3] for x in sel])
        o[f"n_pairs_{tag}"] = len(sel)
    o["ate_local_cm"] = 100 * local_ate(run, a_, b_)
    sp = (ids[1:] >= a_) & (ids[1:] <= b_) & (ids[:-1] >= a_)
    keep = sp & (dg >= 0.002)
    o["path_ratio"] = float(de[sp].sum() / dg[sp].sum()) if sp.any() else None
    o["r_median"] = float(np.median(de[keep] / dg[keep])) if keep.any() else None
    o["r_mean"] = float(np.mean(de[keep] / dg[keep])) if keep.any() else None
    v = raw[s]
    o.update(raw_start=float(v[0]), raw_end=float(v[-1]), raw_min=float(v.min()), raw_max=float(v.max()))
    for tag, E in (("", ES), ("_track", ET)):
        ef, eo = E[s, 0], E[s, 1]
        ok = ~np.isnan(ef)
        o[f"e_free_rms_cm{tag}"] = 100 * rmse(ef[ok]) if ok.any() else None
        o[f"e_obs_rms_cm{tag}"] = 100 * rmse(eo[ok]) if ok.any() else None
    if track is not None:
        js = [track["bid"][int(f)] for f in ids[s] if int(f) in track["bid"]]
        o["n_obs_t_track_mean"] = float(np.mean([(track["sig"][j] >= track["thr"]).sum() for j in js])) if js else None
    sm = [fnum(flog[f]["sigma_min"]) for f in ids[s] if f in flog and flog[f].get("sigma_min", "") != ""]
    o["sigma_min_median_run"] = float(np.median(sm)) if sm else None
    if basis is not None:
        js = [basis["bid"][int(f)] for f in ids[s] if int(f) in basis["bid"]]
        o["n_obs_t_scene_mean"] = float(np.mean([(basis["sig"][j] >= basis["thr"]).sum() for j in js])) if js else None
    src = [flog[f].get("init_source", "") for f in ids[s] if f in flog and f > 0]
    o["pct_init_plane"] = 100.0 * np.mean([x == "plane" for x in src]) if src else None
    return o


res = dict(run_dir=run_dir, method=method, seed=seed, n_frames=len(ids),
           ate_all_cm=100 * m1["ate_rmse_m"], ate_kf_cm=100 * m1["ate_kf_m"],
           whole=block(0, 10 ** 6), segments={n: block(x, y) for n, x, y in SEGMENTS},
           basis=a.basis if basis is not None else None)
res["e_free_obs_S3uS5"] = None
s = ((ids >= 221) & (ids <= 295)) | ((ids >= 481) & (ids <= 790))
out_ = {}
for tag, E, bs in (("scene", ES, basis), ("track", ET, track)):
    if bs is None:
        continue
    ok = s & ~np.isnan(E[:, 0])
    out_[tag] = dict(e_free_rms_cm=100 * rmse(E[ok, 0]), e_obs_rms_cm=100 * rmse(E[ok, 1]), n=int(ok.sum()))
res["e_free_obs_S3uS5"] = out_ or None
json.dump(res, open(os.path.join(out, "seg_metrics.json"), "w"), indent=1)

# per-run figure: unaligned error + scene sigma (t) per frame
mpl_style(plt)
fig, axs = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
for ax in axs:
    shade_segments(ax)
axs[0].plot(ids, raw, color=BASE_C if method == "base" else "#2a78d6", lw=1.6)
axs[0].set_ylabel("sai số không căn (cm)")
if basis is not None:
    js = [basis["bid"].get(int(f)) for f in ids]
    S_ = np.array([basis["sig"][j] if j is not None else [np.nan] * 3 for j in js])
    for c in range(3):
        axs[1].plot(ids, S_[:, c], lw=1.0, color=["#0b0b0b", "#52514e", "#a3a29c"][c], label=f"σ_t,{c + 1}")
    axs[1].axhline(basis["thr"], ls="--", lw=1, color=INK2)
    axs[1].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
axs[1].set_ylabel("σ_t (V_scene)")
axs[1].set_xlabel("frame (chỉ số dataset)")
fig.suptitle(f"{method} {seed}: sai số không căn và σ của cảnh", fontsize=10, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(out, "seg_metrics_raw_sigma.png"))
plt.close(fig)
print(json.dumps({k: res[k] for k in ("method", "seed", "ate_all_cm", "e_free_obs_S3uS5")}, indent=1))
print(out)
