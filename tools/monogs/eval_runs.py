"""PLAN MONOGS-E1C §4.2, §6.1: metrics of MonoGS runs (results_monogs/...).
usage (repo root, env monogs): python tools/monogs/eval_runs.py RUN_DIR [RUN_DIR ...] --out metrics.json
       python tools/monogs/eval_runs.py --pair RUN_A RUN_B --out pair.json        (repeatability, §4.2)
ATE: evo, SE(3) Umeyama without scale (MonoGS eval_ate), translation RMSE; keyframes (MonoGS protocol, main) and
every frame (same alignment method, on traj_all). RPE delta 1 frame, translation. Unaligned error: run starts at GT.
"""
import argparse
import json
import os

import numpy as np

COS, DD = 0.98, 0.05      # PLAN §6.1 near-duplicate / ID-switch link: |cos| > 0.98, |delta d| < 5 cm


def load_tum(path):
    from scipy.spatial.transform import Rotation
    ts, P = [], []
    for line in open(path):
        v = line.split()
        if len(v) < 8 or line.startswith("#"):
            continue
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([float(x) for x in v[4:8]]).as_matrix()
        T[:3, 3] = [float(x) for x in v[1:4]]
        ts.append(float(v[0]))
        P.append(T)
    return np.array(ts), P


def ate(est, gt):
    from evo.core import metrics, trajectory
    from evo.core.trajectory import PosePath3D
    ref = PosePath3D(poses_se3=[g.copy() for g in gt])
    e = PosePath3D(poses_se3=[x.copy() for x in est])
    e = trajectory.align_trajectory(e, ref, correct_scale=False)
    m = metrics.APE(metrics.PoseRelation.translation_part)
    m.process_data((ref, e))
    return float(m.get_statistic(metrics.StatisticsType.rmse)), m.error


def rpe1(est, gt):
    from evo.core import metrics
    from evo.core.trajectory import PosePath3D
    ref = PosePath3D(poses_se3=[g.copy() for g in gt])
    e = PosePath3D(poses_se3=[x.copy() for x in est])
    m = metrics.RPE(metrics.PoseRelation.translation_part, delta=1, delta_unit=metrics.Unit.frames, all_pairs=False)
    m.process_data((ref, e))
    return float(m.get_statistic(metrics.StatisticsType.rmse))


def jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def near(a, b):
    na, nb = np.asarray(a["n_w"]), np.asarray(b["n_w"])
    c = float(na @ nb)
    if abs(c) <= COS:
        return False
    db = b["d_w"] if c > 0 else -b["d_w"]          # same orientation before comparing offsets
    return abs(a["d_w"] - db) < DD


def plane_metrics(run, max_frame=None):
    fr = [r for r in jsonl(os.path.join(run, "e1c_frames.jsonl")) if max_frame is None or r["frame"] < max_frame]
    an = [r for r in jsonl(os.path.join(run, "anchors.jsonl")) if max_frame is None or r["frame"] < max_frame]
    if not fr:
        return None
    trk = [r for r in fr if r["init_source"] != "gt"]
    out = dict(n_frames_logged=len(fr),
               pct_init_plane=100.0 * np.mean([r["init_source"] == "plane" for r in trk]) if trk else None,
               pct_sigma_ge_03=100.0 * np.mean([(r["sigma_min"] or 0) >= 0.3 for r in trk]) if trk else None,
               ms_plane_mean=float(np.mean([r["ms_extract"] + r["ms_match"] + r["ms_solve"] for r in trk])) if trk else None,
               mean_pairs=float(np.mean([r["n_pairs"] for r in trk])) if trk else None,
               mean_active_pairs=float(np.mean([r["n_active_pairs"] for r in trk])) if trk else None)
    if an:
        last = an[-1]["anchors"]
        out["anchors_total"] = len(last)
        out["anchors_active"] = sum(a["active"] for a in last)
        dup = set()
        for i in range(len(last)):
            for j in range(i + 1, len(last)):
                if near(last[i], last[j]):
                    dup.add((last[i]["id"], last[j]["id"]))
        out["near_dup_pairs"] = len(dup)
        out["anchors_per_kf"] = [[r["frame"], len(r["anchors"]), sum(a["active"] for a in r["anchors"])] for r in an]
        # union of anchor states over time for the duplicate classification of ID switches
        table = {}
        for r in an:
            for a in r["anchors"]:
                table[a["id"]] = a
    else:
        table = {}
    sw_dup = sw_sus = 0
    for prev, cur in zip(fr[:-1], fr[1:]):
        for p in prev["planes"]:
            for c in cur["planes"]:
                if p["anchor"] is None or c["anchor"] is None or p["anchor"] == c["anchor"]:
                    continue
                if not near(p, c):
                    continue
                a, b = table.get(p["anchor"]), table.get(c["anchor"])
                if a is not None and b is not None and near(a, b):
                    sw_dup += 1
                else:
                    sw_sus += 1
    out["id_switch_dup"] = sw_dup
    out["id_switch_suspect"] = sw_sus
    return out


def run_metrics(run, max_frame=None):
    """max_frame: keep frames < N only (PLAN MONOGS-PW 7.3: cut a full-sequence run to the range of a shorter one)."""
    m = dict(run=run)
    ts, est = load_tum(os.path.join(run, "traj_all.txt"))
    _, gt = load_tum(os.path.join(run, "traj_gt_all.txt"))
    kf = [int(x) for x in open(os.path.join(run, "kf_indices.txt")).read().split()]
    if max_frame is not None:
        est, gt, ts = est[:max_frame], gt[:max_frame], ts[:max_frame]
        kf = [k for k in kf if k < max_frame]
    tim = json.load(open(os.path.join(run, "timing.json")))
    off = 0
    ids = list(range(len(est)))                       # frames are 0..N-1 in the run (after --range)
    pos = {i: k for k, i in enumerate(ids)}
    kf_pos = [pos[i] for i in kf if i in pos]
    m["n_frames"], m["n_kf"] = len(est), len(kf_pos)
    m["ate_kf_cm"] = 100 * ate([est[i] for i in kf_pos], [gt[i] for i in kf_pos])[0]
    a_all, err_all = ate(est, gt)
    m["ate_all_cm"] = 100 * a_all
    m["rpe_d1_cm"] = 100 * rpe1(est, gt)
    un = np.array([np.linalg.norm(e[:3, 3] - g[:3, 3]) for e, g in zip(est, gt)])
    m["unaligned_end_cm"] = 100 * float(un[-1])
    m["unaligned_max_cm"] = 100 * float(un.max())
    m["unaligned_cm"] = (100 * un).tolist()
    m["ms_track"] = tim.get("ms_track_mean")
    m["ms_plane"] = tim.get("ms_plane_mean")
    m["total_s"] = tim.get("total_s")
    m["s_per_frame"] = tim["total_s"] / max(len(est), 1)
    m["max_orth_dev"] = tim.get("max_orth_dev")
    # MonoGS's own number (eval_ate, keyframes) for cross-check
    for p in ("plot/stats_final.json", "stats_final.json"):
        q = os.path.join(run, p)
        if os.path.exists(q):
            m["monogs_ate_kf_cm"] = 100 * json.load(open(q))["rmse"]
    m["plane"] = plane_metrics(run, max_frame)
    m["max_frame"] = max_frame
    m["kf_indices"] = kf
    m["repro"] = tim.get("repro")
    return m


def pair_metrics(a, b):
    kfa = [int(x) for x in open(os.path.join(a, "kf_indices.txt")).read().split()]
    kfb = [int(x) for x in open(os.path.join(b, "kf_indices.txt")).read().split()]
    _, ea = load_tum(os.path.join(a, "traj_all.txt"))
    _, eb = load_tum(os.path.join(b, "traj_all.txt"))
    common = sorted(set(kfa) & set(kfb))
    dc = [np.linalg.norm(ea[i][:3, 3] - eb[i][:3, 3]) for i in common if i < len(ea) and i < len(eb)]
    n = min(len(ea), len(eb))
    df = [np.linalg.norm(ea[i][:3, 3] - eb[i][:3, 3]) for i in range(n)]
    ma, mb = run_metrics(a), run_metrics(b)
    first = next((i for i in range(n) if df[i] > 1e-4), None)
    return dict(a=a, b=b, kf_identical=kfa == kfb, n_kf=[len(kfa), len(kfb)], n_kf_common=len(common),
                max_kf_centre_diff_cm=100 * float(max(dc)) if dc else None,
                max_frame_centre_diff_cm=100 * float(max(df)) if df else None, first_frame_diff_gt_1mm=first,
                ate_kf_cm=[ma["ate_kf_cm"], mb["ate_kf_cm"]], ate_all_cm=[ma["ate_all_cm"], mb["ate_all_cm"]],
                rel_noise_kf=abs(ma["ate_kf_cm"] - mb["ate_kf_cm"]) / ma["ate_kf_cm"],
                rel_noise_all=abs(ma["ate_all_cm"] - mb["ate_all_cm"]) / ma["ate_all_cm"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--pair", nargs=2, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frame", type=int, default=None)
    a = ap.parse_args()
    if a.pair:
        res = pair_metrics(*a.pair)
    else:
        res = {r: run_metrics(r, a.max_frame) for r in a.runs if os.path.exists(os.path.join(r, "traj_all.txt"))}
    json.dump(res, open(a.out, "w"), indent=1)
    show = res if a.pair else {k: {kk: vv for kk, vv in v.items() if kk not in ("unaligned_cm", "kf_indices")} for k, v in res.items()}
    print(json.dumps(show, indent=1)[:4000])


if __name__ == "__main__":
    main()
