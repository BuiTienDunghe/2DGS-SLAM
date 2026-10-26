"""PLAN MONOGS-PW §8: metrics of the PW runs (and of the E1c / base runs they are compared with).
usage (repo root, env monogs): python tools/monogs/pw/eval_pw.py --spec SPEC.json --out docs/monogs/pw/metrics_pw.json
SPEC: {"<seq>": {"max_frame": N or null, "runs": {"<method>": "<run dir>", ...}}, ...}
8.6 (no GT labels): tools/monogs/eval_runs.py (unchanged, --max-frame) + kernel, keyframe reasons, anchors, 1b, PGO.
8.1-8.2 (GT labels, tool only): true planes by greedy clustering of the measured planes taken to the world with GT.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "tools", "monogs"))
sys.path.insert(0, os.path.join(ROOT, "monogs"))
from eval_runs import jsonl, load_tum, run_metrics  # noqa: E402

LAB_DEG, LAB_M = 3.0, 0.03
ZB = [(0, 1), (1, 2), (2, 3), (3, 99)]


def plane_c2w(n_c, d_c, T):
    R, t = T[:3, :3], T[:3, 3]
    n_c = np.asarray(n_c, dtype=np.float64)
    return R.T @ n_c, float(d_c) + float(n_c @ t)


def plane_w2c(n_w, d_w, T):
    R, t = T[:3, :3], T[:3, 3]
    n_c = R @ np.asarray(n_w, dtype=np.float64)
    return n_c, float(d_w) - float(n_c @ t)


def cut(rows, mf):
    return [r for r in rows if mf is None or r["frame"] < mf]


# ------------------------------------------------------------------ 8.6 extras
def pw_extras(run, mf):
    out = {}
    fr = cut(jsonl(os.path.join(run, "pw_frames.jsonl")), mf)
    e1 = cut(jsonl(os.path.join(run, "e1c_frames.jsonl")), mf)
    if e1:
        trk = [r for r in e1 if r["init_source"] != "gt"]
        out["pct_enough"] = 100.0 * np.mean([(r["n_active_pairs"] >= 3) and ((r["sigma_min"] or 0) >= 0.3) for r in trk]) if trk else None
    if fr:
        out["kf_reasons"] = dict(collections.Counter(r["kf_reason"] for r in fr if r["is_kf"]))
        cand = [p for r in fr if r["init_source"] != "gt" for p in r["planes"] if p["anchor"] is not None and p["e"] is not None]
        out["pct_rejected"] = 100.0 * np.mean([not p["inlier"] for p in cand]) if cand else None
        out["mean_w"] = float(np.mean([p["w"] for p in cand])) if cand else None
        out["init_dropped"] = int(sum(r.get("n_init_dropped", 0) for r in fr))
        out["frames_init_dropped"] = int(sum(1 for r in fr if r.get("n_init_dropped", 0) > 0))
        eb = {}
        for lo, hi in ZB:
            es = [p["e"] for p in cand if lo <= p["z"] < hi]
            if es:
                eb[f"{lo}-{hi}"] = dict(n=len(es), median_e=float(np.median(es)), pct_rejected=100.0 * np.mean(np.array(es) > 3))
        out["e_by_depth"] = eb
    ev = cut([dict(e, frame=e.get("kf", 0)) for e in jsonl(os.path.join(run, "events.jsonl"))], mf)
    if ev:
        c = collections.Counter(e["type"] for e in ev)
        out["anchors_created"], out["anchors_deleted"] = c.get("create", 0), c.get("delete", 0)
        out["anchors_activated"], out["merges"] = c.get("activate", 0), c.get("merge", 0)
        out["merge_reasons"] = dict(collections.Counter(e.get("reason") for e in ev if e["type"] == "merge"))
        out["closure_t1"] = sum(1 for e in ev if e["type"] == "closure_detect" and e["tier"] == 1)
        out["closure_t2"] = sum(1 for e in ev if e["type"] == "closure_detect" and e["tier"] == 2)
    an = cut(jsonl(os.path.join(run, "anchors.jsonl")), mf)
    if an and "aliases" in an[-1]:
        out["aliases_end"] = len(an[-1]["aliases"])
    pg = cut(jsonl(os.path.join(run, "pgo.jsonl")), mf)
    out["pgo"] = dict(n=len(pg), applied=sum(r["status"] == "applied" for r in pg), aborted=sum(r["status"] == "aborted" for r in pg),
                      skipped=sum(r["status"] == "skipped" for r in pg), dropped=int(sum(r.get("dropped") or 0 for r in pg)),
                      closure_accepted=int(sum(r.get("closure_accepted") or 0 for r in pg)),
                      ms_mean=float(np.mean([r["ms"] for r in pg])) if pg else None,
                      max_shift_cm=100 * max([r.get("max_kf_shift_m") or 0 for r in pg] or [0]))
    st = cut(jsonl(os.path.join(run, "stiffness.jsonl")), mf)
    if st:
        kf = [r for r in st if r.get("is_kf")] or st
        cr = np.array([r["c_render"] for r in kf])
        out["stiff"] = dict(n=len(kf), ms_mean=float(np.mean([r["ms"] for r in st])),
                            c_render_med=np.median(cr, 0).tolist(), c_render_q25=np.percentile(cr, 25, 0).tolist(),
                            c_render_q75=np.percentile(cr, 75, 0).tolist())
        sh = [[r["share"][k] for r in kf if r.get("share") and r["share"][k] is not None] for k in range(6)]
        if any(sh):
            out["stiff"].update(share_med=[float(np.median(s)) if s else None for s in sh],
                                share_q25=[float(np.percentile(s, 25)) if s else None for s in sh],
                                share_q75=[float(np.percentile(s, 75)) if s else None for s in sh],
                                share_all_med=float(np.median([v for s in sh for v in s])))
    corr = os.path.join(run, "traj_all_corr.txt")
    if os.path.exists(corr):
        from eval_runs import ate
        _, est = load_tum(corr)
        _, gt = load_tum(os.path.join(run, "traj_gt_all.txt"))
        n = len(est) if mf is None else min(mf, len(est))
        out["ate_all_corr_cm"] = 100 * ate(est[:n], gt[:n])[0]
    return out


# ------------------------------------------------------------------ 8.1-8.2 GT labels
def label_obs(obs):
    heads, out = [], []
    for o in obs:
        hit = None
        for ci, h in enumerate(heads):
            c = float(o["n_g"] @ h["n_g"])
            if c > 0 and np.degrees(np.arccos(min(1.0, c))) <= LAB_DEG and abs(o["d_g"] - h["d_g"]) <= LAB_M:
                hit = ci
                break
        if hit is None:
            heads.append(o)
            hit = len(heads) - 1
        out.append(hit)
    return out


def gt_obs(run, kind, mf):
    """Measured planes per frame in the GT world, with matched anchor (E1c: any match; PW: inlier pairs) and, for
    PW, every candidate pair (anchor + inlier flag) from pw_frames."""
    _, est = load_tum(os.path.join(run, "traj_all.txt"))
    _, gt = load_tum(os.path.join(run, "traj_gt_all.txt"))
    W = [np.linalg.inv(T) for T in est]          # w2c estimated
    G = [np.linalg.inv(T) for T in gt]
    obs = []
    if kind == "pw":
        for r in cut(jsonl(os.path.join(run, "pw_frames.jsonl")), mf):
            f = r["frame"]
            for i, p in enumerate(r["planes"]):
                n_g, d_g = plane_c2w(p["n_c"], p["d_c"], G[f])
                obs.append(dict(frame=f, i=i, n_g=n_g / np.linalg.norm(n_g), d_g=d_g, cand=p["anchor"], inlier=bool(p["inlier"]),
                                anchor=p["anchor"] if p["inlier"] else None, init=r["init_source"]))
    else:
        for r in cut(jsonl(os.path.join(run, "e1c_frames.jsonl")), mf):
            f = r["frame"]
            for i, p in enumerate(r["planes"]):
                n_c, d_c = plane_w2c(p["n_w"], p["d_w"], W[f])
                n_g, d_g = plane_c2w(n_c, d_c, G[f])
                obs.append(dict(frame=f, i=i, n_g=n_g / np.linalg.norm(n_g), d_g=d_g, cand=p["anchor"],
                                inlier=p["anchor"] is not None, anchor=p["anchor"], init=r["init_source"]))
    lab = label_obs(obs)
    for o, l in zip(obs, lab):
        o["label"] = l
    return obs, W, G


def gt_metrics(run, kind, mf):
    obs, W, G = gt_obs(run, kind, mf)
    votes = collections.defaultdict(collections.Counter)
    for o in obs:
        if o["anchor"] is not None:
            votes[o["anchor"]][o["label"]] += 1
    alab = {a: c.most_common(1)[0][0] for a, c in votes.items()}
    an = cut(jsonl(os.path.join(run, "anchors.jsonl")), mf)
    alias = {}
    for r in an:
        alias.update({int(k): int(v) for k, v in (r.get("aliases") or {}).items()})
    ever_active = {a["id"] for r in an for a in r["anchors"] if a["active"]}
    per_label = collections.Counter(alab[a] for a in ever_active if a in alab and a not in alias)
    pairs = [o for o in obs if o["anchor"] is not None]
    mism = [o["label"] != alab.get(o["anchor"]) for o in pairs]
    out = dict(n_true_planes=len(set(o["label"] for o in obs)), labels_with_anchor=len(per_label),
               anchors_per_true_plane=(sum(per_label.values()) / len(per_label)) if per_label else None,
               duplicates=int(sum(v - 1 for v in per_label.values())),
               pct_mismatch=100.0 * np.mean(mism) if mism else None)
    if kind == "pw":
        cand = [o for o in obs if o["cand"] is not None]
        lab_c = [o["label"] != alab.get(o["cand"], -1) for o in cand]
        rej = [o for o, w in zip(cand, lab_c) if not o["inlier"]]
        acc = [o for o, w in zip(cand, lab_c) if o["inlier"]]
        out.update(kernel_pct_rejected=100.0 * len(rej) / len(cand) if cand else None,
                   kernel_wrong_in_rejected=100.0 * np.mean([o["label"] != alab.get(o["cand"], -1) for o in rej]) if rej else None,
                   kernel_wrong_in_accepted=100.0 * np.mean([o["label"] != alab.get(o["cand"], -1) for o in acc]) if acc else None,
                   kernel_wrong_in_all=100.0 * np.mean(lab_c) if lab_c else None)
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    for o in pairs:
        by[o["label"]][o["frame"]].add(o["anchor"])
    simult = flip = 0
    flip_frames = set()
    for l, fs in by.items():
        frames = sorted(fs)
        for f in frames:
            if len(fs[f]) >= 2:
                simult += 1
        for f0, f1 in zip(frames[:-1], frames[1:]):
            if len(fs[f0]) == 1 and len(fs[f1]) == 1 and fs[f0] != fs[f1]:
                flip += 1
                flip_frames.add(f1)
    out.update(id_simultaneous=simult, id_flip=flip)
    # anchor value change outside PGO (consecutive "kf" rows)
    rows = [r for r in an if r.get("event", "kf") == "kf"]
    dang, dd = collections.defaultdict(float), collections.defaultdict(float)
    prev = {}
    for r in rows:
        cur = {a["id"]: a for a in r["anchors"] if a["active"]}
        for i, a in cur.items():
            if i in prev:
                c = float(np.clip(np.dot(a["n_w"], prev[i]["n_w"]), -1, 1))
                dang[i] += np.degrees(np.arccos(c))
                dd[i] += abs(a["d_w"] - prev[i]["d_w"])
        prev = cur
    # remove the jumps of PGO steps: rows "pre_pgo"/"post_pgo" are not in `rows`, and a kf row after a PGO compares
    # with the kf row before it -> subtract the pre->post change
    pre = {r["frame"]: r for r in an if r.get("event") == "pre_pgo"}
    post = {r["frame"]: r for r in an if r.get("event") == "post_pgo"}
    for f, r0 in pre.items():
        r1 = post.get(f)
        if not r1:
            continue
        a0 = {a["id"]: a for a in r0["anchors"]}
        for a in r1["anchors"]:
            if a["id"] in a0 and a["id"] in dang:
                c = float(np.clip(np.dot(a["n_w"], a0[a["id"]]["n_w"]), -1, 1))
                dang[a["id"]] = max(0.0, dang[a["id"]] - np.degrees(np.arccos(c)))
                dd[a["id"]] = max(0.0, dd[a["id"]] - abs(a["d_w"] - a0[a["id"]]["d_w"]))
    if dang:
        out.update(change_ang_med_deg=float(np.median(list(dang.values()))), change_ang_max_deg=float(max(dang.values())),
                   change_d_med_cm=100 * float(np.median(list(dd.values()))), change_d_max_cm=100 * float(max(dd.values())))
    out["_flip_frames"] = sorted(flip_frames)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    res = {}
    for seq, sv in spec.items():
        mf = sv.get("max_frame")
        res[seq] = {}
        for meth, run in sv["runs"].items():
            if not os.path.exists(os.path.join(run, "traj_all.txt")):
                res[seq][meth] = dict(missing=run)
                continue
            m = run_metrics(run, mf)
            m.pop("unaligned_cm", None)
            m.pop("kf_indices", None)
            m["pw"] = pw_extras(run, mf)
            kind = "pw" if os.path.exists(os.path.join(run, "pw_frames.jsonl")) else ("e1c" if os.path.exists(os.path.join(run, "e1c_frames.jsonl")) else None)
            if kind:
                try:
                    g = gt_metrics(run, kind, mf)
                    g.pop("_flip_frames", None)
                    m["gt"] = g
                except Exception as e:  # noqa: BLE001
                    m["gt"] = dict(error=str(e))
            res[seq][meth] = m
            print(seq, meth, f"ATE kf {m['ate_kf_cm']:.2f} all {m['ate_all_cm']:.2f} rpe {m['rpe_d1_cm']:.3f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
