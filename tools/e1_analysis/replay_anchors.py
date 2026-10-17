"""E1 analysis Q4: replay the anchor logic outside SLAM on the ours run (CPU only, read-only).

Per tracked frame k (dataset depth, locked PlaneTracking parameters, ours final poses):
  CAPE-lite -> match(planes, T_{k-1}) -> solve_pose (T_init) -> at keyframes (flag from frame_log)
  update_at_keyframe(k, planes, match(planes, T_k), T_k).
With loop closure / PGO off the backend never changes camera poses (slam_backend.py optimises exposure only),
so the final poses in traj_est_all.txt are the poses the run used; the replay should match frame_log.csv.

Extra (not in the task list, item 7): the same keyframe observations re-integrated with GT poses give
"GT anchors"; their difference to the replayed anchors is the error the anchors inherited from the poses.
usage: python tools/e1_analysis/replay_anchors.py   (E1_REPLAY_DEVICE=cuda|cpu, default cuda: same device as the run)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, ".")
sys.path.insert(0, "gaussian_splatting")
from munch import munchify  # noqa: E402

from common import GT_C, INK2, OUT, SEGMENTS, fnum, load_run, mpl_style, rel_err, rpe_pairs  # noqa: E402
from utils.config_utils import load_config  # noqa: E402
from utils.dataset import load_dataset  # noqa: E402
from utils.plane_anchor import PlaneAnchorMap, PlaneExtractor, plane_c2w, plane_w2c  # noqa: E402

os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
mpl_style(plt)
cfg = load_config("configs/tum/e1/fr3_snt_far.yaml")
pt = cfg["PlaneTracking"]
ds = load_dataset(munchify(cfg["model_params"]), "", config=cfg)
ds.device = os.environ.get("E1_REPLAY_DEVICE", "cuda")   # the run extracted planes on the GPU
run = load_run("ours")
ids, est, gt, flog = run["ids"], run["est"], run["gt"], run["flog"]
# the run kept poses as float32 w2c tensors; round-trip through float32 to mimic it
w2c = {int(f): np.linalg.inv(P).astype(np.float32).astype(np.float64) for f, P in zip(ids, est)}
c2w = {int(f): P for f, P in zip(ids, est)}
gtc = {int(f): P for f, P in zip(ids, gt)}

ext = PlaneExtractor(pt)
amap = PlaneAnchorMap(pt, ds.K, ds.height, ds.width)
back = {}                    # anchor id -> times skipped as back-facing
kf_obs = {}                  # anchor id -> list of (kf, n_c, d_c, n_px) observations used to update it
records = []
plane_resid = {}             # anchor id -> list of (dist resid m, angle deg) with the frame's final pose


def count_back(T):
    for aid, a in amap.anchors.items():
        if plane_w2c(a.n_w, a.d_w, T)[1] <= 0:
            back[aid] = back.get(aid, 0) + 1


def kf_update(fid, planes, T):
    pairs_kf = amap.match(planes, torch.from_numpy(T))
    before = set(amap.anchors)
    amap.update_at_keyframe(fid, planes, pairs_kf, torch.from_numpy(T))
    m = dict(pairs_kf)
    new = sorted(set(amap.anchors) - before)
    unmatched = [pi for pi in range(len(planes)) if pi not in m]
    for pi, aid in list(m.items()) + list(zip(unmatched, new)):
        kf_obs.setdefault(aid, []).append((fid, planes[pi].n.numpy().copy(), planes[pi].d, planes[pi].n_px))


prev = None
prev_active = set()
for k, fid in enumerate(ids):
    fid = int(fid)
    _, _, depth, _ = ds[fid]
    planes = ext(depth, ds.K)
    row = flog[fid]
    is_kf = row["is_kf"] == "1"
    if k == 0:
        amap.update_at_keyframe(fid, planes, [], torch.from_numpy(w2c[fid]))
        for aid in amap.anchors:
            kf_obs.setdefault(aid, []).append((fid, None, None, None))
        for pi, aid in enumerate(sorted(amap.anchors)):
            kf_obs[aid] = [(fid, planes[pi].n.numpy().copy(), planes[pi].d, planes[pi].n_px)]
        records.append(dict(fid=fid, n_planes=len(planes), n_matched=0, n_matched_active=0, sigma_min=float("nan"),
                            init_source="prev", T_init=None, active_ids=[]))
        prev = fid
        continue
    T_prev = w2c[prev]
    count_back(T_prev)
    pairs = amap.match(planes, torch.from_numpy(T_prev))
    T_plane = amap.solve_pose(planes, pairs, torch.from_numpy(T_prev))
    src = "prev"
    if amap.last_solve_status == "rejected":
        src = "rejected"
    if T_plane is not None:
        src = "plane"
    act_ids = sorted(aid for _, aid in amap.active_pairs(pairs))
    sigma = amap.last_sigma_min
    # plane residuals against anchors predicted with this frame's final pose (anchors as used for tracking)
    Tk = w2c[fid]
    for pi, aid in amap.active_pairs(pairs):
        n_c, d_c = plane_w2c(amap.anchors[aid].n_w, amap.anchors[aid].d_w, Tk)
        m_ = planes[pi].n.numpy()
        plane_resid.setdefault(aid, []).append((planes[pi].d - d_c, float(np.degrees(np.arccos(np.clip(m_ @ n_c, -1, 1))))))
    if is_kf:
        kf_update(fid, planes, Tk)
    # frame_log logs n_matched_active after the keyframe update (anchor activation may change)
    records.append(dict(fid=fid, n_planes=len(planes), n_matched=len(pairs),
                        n_matched_active=len(amap.active_pairs(pairs)), sigma_min=sigma, init_source=src,
                        T_init=None if T_plane is None else T_plane.numpy(), active_ids=act_ids, is_kf=is_kf))
    prev = fid

# ---------------------------------------------------------------- replay vs run
def same(rec, r):
    s_run, s_rep = fnum(r["sigma_min"]), rec["sigma_min"]
    s_ok = (np.isnan(s_run) and np.isnan(s_rep)) or abs(s_run - s_rep) <= 1e-3 + 1e-3 * abs(s_run)
    return (int(r["n_planes"] or 0) == rec["n_planes"] and int(r["n_matched_active"] or 0) == rec["n_matched_active"]
            and r["init_source"] == rec["init_source"] and s_ok)


match_flags = [same(rec, flog[rec["fid"]]) for rec in records]
match_rate = float(np.mean(match_flags))
field_rates = {
    "n_planes": float(np.mean([int(flog[r["fid"]]["n_planes"] or 0) == r["n_planes"] for r in records])),
    "n_matched_active": float(np.mean([int(flog[r["fid"]]["n_matched_active"] or 0) == r["n_matched_active"] for r in records])),
    "init_source": float(np.mean([flog[r["fid"]]["init_source"] == r["init_source"] for r in records])),
}
first_mismatch = next((r["fid"] for r, ok in zip(records, match_flags) if not ok), None)
res = dict(match_rate=match_rate, field_rates=field_rates, first_mismatch=first_mismatch, n_frames=len(records))
print(f"replay match rate {match_rate:.4f} fields {field_rates} first mismatch {first_mismatch}")
if match_rate < 0.95:
    res["stopped"] = "match rate < 95 %: Q4 stopped as required"
    json.dump(res, open(os.path.join(OUT, "q4.json"), "w"), indent=1)
    sys.exit(0)

# ---------------------------------------------------------------- 1) anchors
anchors = []
for aid, a in sorted(amap.anchors.items()):
    anchors.append(dict(id=aid, n_w=[round(float(x), 4) for x in a.n_w.numpy()], d_w=round(a.d_w, 4), n_obs_kf=a.n_obs_kf,
                        active=bool(a.active), created_kf=a.created_kf, back_facing=back.get(aid, 0)))
res["anchors"] = anchors
# ---------------------------------------------------------------- 2) 3) duplicates
dup_same, dup_opp = [], []
al = list(amap.anchors.values())
for i in range(len(al)):
    for j in range(i + 1, len(al)):
        c = float(al[i].n_w.numpy() @ al[j].n_w.numpy())
        if c > 0.98 and abs(al[i].d_w - al[j].d_w) < 0.05:
            dup_same.append((al[i].id, al[j].id, round(c, 4), round(abs(al[i].d_w - al[j].d_w) * 100, 2)))
        if c < -0.98 and abs(al[i].d_w + al[j].d_w) < 0.05:
            dup_opp.append((al[i].id, al[j].id, round(c, 4), round(abs(al[i].d_w + al[j].d_w) * 100, 2)))
res["dup_same"], res["dup_opposite"] = dup_same, dup_opp
# ---------------------------------------------------------------- 4) active-set churn vs RPE
rpe1 = {j: t for i, j, t, r in rpe_pairs(run, 1)}
churn, err = [], []
per_frame = []
for r0, r1 in zip(records[:-1], records[1:]):
    ch = len(set(r0["active_ids"]) ^ set(r1["active_ids"]))
    per_frame.append((r1["fid"], ch))
    if r1["fid"] in rpe1:
        churn.append(ch)
        err.append(rpe1[r1["fid"]])
rho, pv = spearmanr(churn, err)
churn = np.array(churn)
err = np.array(err)
res["churn"] = dict(spearman=float(rho), p=float(pv), frac_frames_changed=float(np.mean(churn > 0)),
                    rpe_cm_median_changed=float(np.median(err[churn > 0]) * 100) if (churn > 0).any() else None,
                    rpe_cm_median_same=float(np.median(err[churn == 0]) * 100) if (churn == 0).any() else None,
                    by_segment={})
for name, a, b in SEGMENTS:
    sel = [(c, e) for (f, c), e in zip(per_frame, err) if a <= f <= b]
    if sel:
        c_ = np.array([x for x, _ in sel])
        res["churn"]["by_segment"][name] = dict(frac_changed=float(np.mean(c_ > 0)), mean_churn=float(c_.mean()))
# ---------------------------------------------------------------- 5) init vs final motion accuracy
fid_idx = {int(f): i for i, f in enumerate(ids)}
inits = {}
for rec in records:
    if rec["init_source"] != "plane" or rec["T_init"] is None:
        continue
    i = fid_idx[rec["fid"]]
    P_prev, P_init, P_fin = est[i - 1], np.linalg.inv(rec["T_init"]), est[i]
    ei = rel_err(P_prev, P_init, gt[i - 1], gt[i])
    ef = rel_err(P_prev, P_fin, gt[i - 1], gt[i])
    # absolute error of the pose itself (frames share the GT world: run starts at the GT pose of frame 0)
    abs_i = float(np.linalg.norm(P_init[:3, 3] - gt[i][:3, 3]))
    abs_f = float(np.linalg.norm(P_fin[:3, 3] - gt[i][:3, 3]))
    abs_p = float(np.linalg.norm(P_prev[:3, 3] - gt[i - 1][:3, 3]))
    seg = next(s for s, a, b in SEGMENTS if a <= rec["fid"] <= b)
    inits.setdefault(seg, []).append((ei, ef, abs_p, abs_i, abs_f))
res["init_accuracy"] = {}
for seg, v in inits.items():
    q = lambda x, p: float(np.percentile(x, p))  # noqa: E731
    et_i = [x[0][0] * 100 for x in v]
    er_i = [x[0][1] for x in v]
    et_f = [x[1][0] * 100 for x in v]
    er_f = [x[1][1] for x in v]
    res["init_accuracy"][seg] = dict(n=len(v),
                                     e_init_trans_cm_med=q(et_i, 50), e_init_trans_cm_p90=q(et_i, 90),
                                     e_init_rot_deg_med=q(er_i, 50), e_init_rot_deg_p90=q(er_i, 90),
                                     e_final_trans_cm_med=q(et_f, 50), e_final_trans_cm_p90=q(et_f, 90),
                                     e_final_rot_deg_med=q(er_f, 50), e_final_rot_deg_p90=q(er_f, 90),
                                     abs_prev_cm_med=q([x[2] * 100 for x in v], 50),
                                     abs_init_cm_med=q([x[3] * 100 for x in v], 50),
                                     abs_final_cm_med=q([x[4] * 100 for x in v], 50))
# ---------------------------------------------------------------- 6) plane measurement noise per active anchor
res["plane_noise"] = []
for aid, v in sorted(plane_resid.items()):
    if len(v) < 5:
        continue
    d = np.array([x[0] for x in v]) * 100
    g = np.array([x[1] for x in v])
    res["plane_noise"].append(dict(id=aid, n=len(v), dist_std_cm=float(d.std()), dist_mean_cm=float(d.mean()),
                                   ang_std_deg=float(g.std()), ang_mean_deg=float(g.mean())))
# ---------------------------------------------------------------- 7) anchor error inherited from the poses
inh = []
for aid, obs in sorted(kf_obs.items()):
    if aid not in amap.anchors:
        continue
    ns, ds_, ws = [], [], []
    for kf, n_c, d_c, n_px in obs:
        if n_c is None:
            continue
        n_w, d_w = plane_c2w(n_c, d_c, np.linalg.inv(gtc[kf]))
        ns.append(n_w)
        ds_.append(d_w)
        ws.append(n_px)
    if not ns:
        continue
    w = np.array(ws, float)
    n_gt = (np.array(ns) * w[:, None]).sum(0)
    n_gt /= np.linalg.norm(n_gt)
    d_gt = float((np.array(ds_) * w).sum() / w.sum())
    a = amap.anchors[aid]
    inh.append(dict(id=aid, active=bool(a.active), created_kf=a.created_kf, n_obs=len(ns),
                    ang_err_deg=float(np.degrees(np.arccos(np.clip(a.n_w.numpy() @ n_gt, -1, 1)))),
                    dist_err_cm=float((a.d_w - d_gt) * 100)))
res["anchor_vs_gt"] = inh
act = [x for x in inh if x["active"]]
res["anchor_vs_gt_summary"] = dict(
    n_active=len(act),
    ang_med=float(np.median([x["ang_err_deg"] for x in act])) if act else None,
    ang_max=float(np.max([x["ang_err_deg"] for x in act])) if act else None,
    dist_abs_med_cm=float(np.median([abs(x["dist_err_cm"]) for x in act])) if act else None,
    dist_abs_max_cm=float(np.max([abs(x["dist_err_cm"]) for x in act])) if act else None)
json.dump(res, open(os.path.join(OUT, "q4.json"), "w"), indent=1)

# ---------------------------------------------------------------- F9 anchor map, top view on GT principal axes
gt_xyz = gt[:, :3, 3]
mu = gt_xyz.mean(0)
_, _, Vt = np.linalg.svd(gt_xyz - mu)
P = Vt[:2].T
dup_ids = {i for p in dup_same + dup_opp for i in p[:2]}
pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948", "#008300"]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
k = 0
for aid, a in sorted(amap.anchors.items()):
    q = (a.samples_w.numpy() - mu) @ P
    if aid in dup_ids:
        c = pal[k % len(pal)]
        k += 1
        ax.scatter(q[:, 0], q[:, 1], s=4, color=c, zorder=3)
        ax.annotate(f"{aid}", q.mean(0), fontsize=9, color=c, weight="bold", zorder=4)
    else:
        ax.scatter(q[:, 0], q[:, 1], s=2, color="#c3c2b7", zorder=2)
        ax.annotate(f"{aid}", q.mean(0), fontsize=7, color=INK2, zorder=4)
ax.plot(*((gt_xyz - mu) @ P).T, color=GT_C, lw=1.2, label="quỹ đạo GT")
ax.set_aspect("equal")
ax.set_xlabel("trục chính 1 của GT (m)")
ax.set_ylabel("trục chính 2 (m)")
ax.legend(frameon=False, loc="lower right")
ax.set_title("F9 — điểm mẫu của anchor nhìn từ trên (màu: anchor thuộc cặp trùng; xám: còn lại)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figs", "F9_anchor_map.png"))
plt.close(fig)
print(json.dumps({k: v for k, v in res.items() if k not in ("anchors", "anchor_vs_gt", "plane_noise")}, indent=1))
