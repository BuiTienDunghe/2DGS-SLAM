"""PLAN A2G-P1 (rút gọn) §9: label-free plane-world metrics, identical for E1c and P1 runs.

* near-duplicate pairs at the end: active/frozen anchors with cos > 0.98 and |d_a - d_b| < 5 cm
  (also the antiparallel case, as tools/e1_analysis/replay_anchors.py); P1 anchors reduced to primaries.
* ID switches: segments of consecutive frames are linked when, in the world frame (estimated pose), |cos| > 0.98 and
  |d_a - d_b| < 5 cm (nearest such segment). The plan also asks for >= 30 % cell overlap; masks are not logged by
  E1c, so the overlap condition is not applied (same for both methods). A switch = both linked segments assigned,
  to different anchors. Split: between near-duplicate anchors (final geometry) vs not (suspected mismatch).
* suspected false merge (P1): merge event with > 3 deg or > 3 cm at o of the dropped anchor.
* NIS (P1): EKF innovation (K3) and IC NIS of assigned pairs.
usage: python tools/a2g_p1/metrics.py --run RUN_DIR --out OUT.json
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation


def load_tum(path):
    P = []
    for ln in open(path):
        v = ln.split()
        if len(v) != 8:
            continue
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([float(x) for x in v[4:8]]).as_matrix()
        T[:3, 3] = [float(x) for x in v[1:4]]
        P.append(T)
    return P


def jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def near_dup(a, b):
    c = float(np.dot(a["n_w"], b["n_w"]))
    return (c > 0.98 and abs(a["d_w"] - b["d_w"]) < 0.05) or (c < -0.98 and abs(a["d_w"] + b["d_w"]) < 0.05)


def compute(run):
    fl = list(csv.DictReader(open(os.path.join(run, "frame_log.csv"))))
    planes = {r["frame_id"]: r["planes"] for r in jsonl(os.path.join(run, "planes.jsonl"))}
    est = load_tum(os.path.join(run, "traj_est_all.txt"))
    gt = load_tum(os.path.join(run, "traj_gt_all.txt"))
    tab = jsonl(os.path.join(run, "anchors_kf.jsonl"))
    final = tab[-1]["anchors"] if tab else []
    is_p1 = bool(final) and "primary_id" in final[0]
    prim = {a["id"]: a.get("primary_id", a["id"]) for a in final}
    byid = {a["id"]: a for a in final}
    live = [a for a in final if (a.get("state") in ("active", "frozen")) or (not is_p1 and a["active"])]
    if is_p1:
        live = [a for a in live if a["primary_id"] == a["id"]]
    dup_pairs = [(a["id"], b["id"]) for i, a in enumerate(live) for b in live[i + 1:] if near_dup(a, b)]

    def P(aid):
        return prim.get(aid, aid)

    def dup_ids(x, y):
        a, b = byid.get(P(x)), byid.get(P(y))
        return a is not None and b is not None and near_dup(a, b)

    # per frame: segments in world, assignment
    frames = []
    for i, r in enumerate(fl):
        fid = int(r["frame_id"])
        if fid not in planes or i >= len(est):
            continue
        Twc = est[i]
        segs = []
        for p in planes[fid]:
            n_w = Twc[:3, :3] @ np.array(p["n"])
            d_w = p["d"] - float(n_w @ Twc[:3, 3])
            segs.append((n_w, d_w))
        asg = {}
        if r["pairs"]:
            for t in r["pairs"].split(";"):
                pi, aid = t.split(":")
                asg[int(pi)] = int(aid)
        frames.append((fid, segs, asg))
    sw_dup = sw_bad = links = 0
    for (f0, s0, a0), (f1, s1, a1) in zip(frames[:-1], frames[1:]):
        for j, (n1, d1) in enumerate(s1):
            best, bd = None, None
            for i, (n0, d0) in enumerate(s0):
                c = float(n0 @ n1)
                dd = abs(d0 - d1) if c > 0 else abs(d0 + d1)
                if abs(c) > 0.98 and dd < 0.05 and (bd is None or dd < bd):
                    best, bd = i, dd
            if best is None:
                continue
            links += 1
            if best in a0 and j in a1 and P(a0[best]) != P(a1[j]):
                if dup_ids(a0[best], a1[j]):
                    sw_dup += 1
                else:
                    sw_bad += 1
    # tracking
    m = json.load(open(os.path.join(run, "e1_metrics.json")))
    e_end = float(np.linalg.norm(est[-1][:3, 3] - gt[-1][:3, 3])) if est else None
    e_un = [float(np.linalg.norm(a[:3, 3] - b[:3, 3])) for a, b in zip(est, gt)]
    out = dict(run=run, ate_all_cm=100 * m.get("ate_rmse_m", float("nan")), ate_kf_cm=100 * m.get("ate_kf_m", float("nan")),
               rpe_d1_cm=100 * m.get("rpe_trans_rmse_m_d1", float("nan")), end_err_cm=100 * e_end,
               max_err_cm=100 * max(e_un), pct_init_plane=m.get("pct_init_plane"),
               ms_track=m.get("ms_track_mean"), ms_pw=m.get("ms_pw_mean"), n_frames=m.get("n_frames"),
               has_nan=m.get("has_nan_pose"), n_anchor_final=len([a for a in final if (not is_p1) or a["primary_id"] == a["id"]]),
               n_live_final=len(live), near_dup_end=len(dup_pairs), near_dup_pairs=dup_pairs, id_links=links,
               switch_dup=sw_dup, switch_suspect=sw_bad, err_unaligned_cm=[100 * e for e in e_un])
    if is_p1:
        pw = m.get("pw", {})
        ev = jsonl(os.path.join(run, "pw_events.jsonl"))
        merges = [e for e in ev if e["type"] == "merge"]
        sus = [e for e in merges if e["angle_deg"] > 3 or abs(e["dist_at_o_drop"]) > 0.03]
        kf = jsonl(os.path.join(run, "pw_kf.jsonl"))
        ekf = [x["nis"] for r in kf for x in r.get("ekf", [])]
        fr = jsonl(os.path.join(run, "pw_frames.jsonl"))
        ic = []
        rank6 = shape_fail = jump_fail = dropped = n_jc = 0
        for r in fr:
            asg = {tuple(p) for p in r["assign"]}
            ic += [c["nis"] for c in r["cand"] if (c["s"], c["j"]) in asg and c.get("at", "T_pred") == "T_pred"]
            jc = r.get("jc")
            if jc:
                n_jc += 1
                rank6 += jc["rank"] == 6
                dropped += jc.get("dropped", 0)
                shape_fail += jc.get("dropped", 0) > 0
                jump_fail += jc.get("decision") == "jump"
        out.update(merges=len(merges), merges_suspect=len(sus), merges_suspect_list=[
            dict(k=e["k"], keep=e["keep"], drop=e["drop"], angle_deg=e["angle_deg"], dist_cm=100 * e["dist_at_o_drop"])
            for e in sus],
            conflicts=pw.get("conflicts"), revisits=pw.get("revisits"), cand_deleted=pw.get("cand_deleted"),
            invariant_violations=pw.get("n_invariant"),
            nis_ekf_mean=float(np.mean(ekf)) if ekf else None, nis_ekf_median=float(np.median(ekf)) if ekf else None,
            n_ekf=len(ekf), nis_ic_mean=float(np.mean(ic)) if ic else None,
            nis_ic_median=float(np.median(ic)) if ic else None, n_ic=len(ic),
            jc_frames=n_jc, pct_rank6=100 * rank6 / max(n_jc, 1), pct_shape_drop=100 * shape_fail / max(n_jc, 1),
            pct_jump=100 * jump_fail / max(n_jc, 1), pairs_dropped=dropped,
            init_sources={s: sum(1 for r in fr if r["init"]["source"] == s) for s in
                          ("plane", "jc_jump", "rank_lt6", "none_pairs")})
    if is_p1:                                             # P1 core extras (pairs carry the unambiguous flag)
        fr = jsonl(os.path.join(run, "pw_frames.jsonl"))
        if fr and fr[0].get("pairs") and "unambiguous" in (fr[0]["pairs"][0] if fr[0]["pairs"] else {"unambiguous": 0}):
            icf, icn, una, dT = [], [], [], []
            for r in fr:
                for q in r.get("pairs", []):
                    if q.get("nis") is None:
                        continue
                    (icf if q.get("fresh") else icn).append(q["nis"])
                    una.append(bool(q.get("unambiguous")))
                if r["init"].get("dT_gate") is not None:
                    dT.append(r["init"]["dT_gate"])
            kf = jsonl(os.path.join(run, "pw_kf.jsonl"))
            rf = [x["nis"] for r in kf for x in r.get("reassign", []) if x.get("fresh")]
            fin = {}
            if os.path.exists(os.path.join(run, "anchors_final.json")):
                fin = json.load(open(os.path.join(run, "anchors_final.json")))
            prim = [a for a in fin.get("anchors", []) if a["primary_id"] == a["id"]]
            bth = [float(np.degrees(np.linalg.norm(a["bhat"][:2]))) for a in prim]
            bd = [1000 * abs(a["bhat"][2]) for a in prim]
            pw = m.get("pw", {})
            out.update(core=True, nis_ic_fresh_mean=float(np.mean(icf)) if icf else None,
                       nis_ic_fresh_median=float(np.median(icf)) if icf else None, n_ic_fresh=len(icf),
                       nis_ic_stale_mean=float(np.mean(icn)) if icn else None, n_ic_stale=len(icn),
                       nis_reassign_fresh_mean=float(np.mean(rf)) if rf else None, n_reassign_fresh=len(rf),
                       pct_unambiguous=100.0 * np.mean(una) if una else None,
                       bhat_theta_deg_median=float(np.median(bth)) if bth else None,
                       bhat_d_mm_median=float(np.median(bd)) if bd else None,
                       deferred=pw.get("deferred"), revisit_candidates=pw.get("revisit_candidates"),
                       bias_accept=pw.get("bias_accept"), bias_reject=pw.get("bias_reject"),
                       bias_ambiguous=pw.get("bias_ambiguous"),
                       dT_gate_median=float(np.median(dT)) if dT else None)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = compute(a.run)
    json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("err_unaligned_cm",)}, indent=1))
