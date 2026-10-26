"""PLAN MONOGS-PW2 §7: metrics of the R2 runs (+ the reused base / e1c / R1 full runs).
usage (repo root, env monogs): python tools/monogs/pw2/eval_pw2.py [--only SEQ:METHOD ...] --out docs/monogs/pw2/metrics_pw2.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "tools", "monogs"))
sys.path.insert(0, os.path.join(ROOT, "tools", "monogs", "pw"))
from eval_pw import gt_metrics, gt_obs, pw_extras  # noqa: E402
from eval_runs import ate, jsonl, load_tum, run_metrics  # noqa: E402

B, P1, P2 = "results_monogs/B", "results_monogs/PW", "results_monogs/PW2"
SPEC = {
    "snt_far": dict(max_frame=None, title="fr3/structure_notexture_far (đủ chuỗi)",
                    runs=dict(base=f"{B}/snt_far/base", e1c=f"{B}/snt_far/e1c", full_R1=f"{P1}/snt_far/full",
                              pw2=f"{P2}/snt_far/pw2", pw2_nopgo=f"{P2}/snt_far/pw2_nopgo", base_kf=f"{P2}/snt_far/base_kf")),
    "st_far": dict(max_frame=500, title="fr3/structure_texture_far (frame 0–500)",
                   runs=dict(base=f"{P1}/st_far/base", e1c=f"{P1}/st_far/e1c", full_R1=f"{P1}/st_far/full", pw2=f"{P2}/st_far/pw2")),
    "snt_near": dict(max_frame=700, title="fr3/structure_notexture_near (frame 0–700)",
                     runs=dict(base=f"{B}/snt_near/base", e1c=f"{B}/snt_near/e1c", full_R1=f"{P1}/snt_near/full",
                               pw2=f"{P2}/snt_near/pw2", pw2_nopgo=f"{P2}/snt_near/pw2_nopgo")),
}


def correct_rejected(run, mf):
    """% of candidate pairs with the correct GT label that the kernel rejected (PW logs only)."""
    obs, _, _ = gt_obs(run, "pw", mf)
    votes = collections.defaultdict(collections.Counter)
    for o in obs:
        if o["anchor"] is not None:
            votes[o["anchor"]][o["label"]] += 1
    alab = {a: c.most_common(1)[0][0] for a, c in votes.items()}
    good = [o for o in obs if o["cand"] is not None and alab.get(o["cand"]) == o["label"]]
    return 100.0 * np.mean([not o["inlier"] for o in good]) if good else None, len(good), obs, alab


def pgo_table(run, mf, obs, alab):
    rows = []
    _, gt = load_tum(os.path.join(run, "traj_gt_all.txt"))
    lab_at = collections.defaultdict(dict)
    for o in obs:
        lab_at[o["frame"]][o["i"]] = o
    fr = {r["frame"]: r for r in jsonl(os.path.join(run, "pw_frames.jsonl"))}
    for r in jsonl(os.path.join(run, "pgo_poses.jsonl")):
        k = r["frame"]
        if mf is not None and k >= mf:
            continue
        cl = []
        for c in r["closures"]:
            aL = alab.get(c["anchor"])
            if c["tier"] == 2 and c["plane"] is not None:
                o = lab_at[c["kf"]].get(c["plane"])
                ok = None if o is None or aL is None else (o["label"] == aL)
            else:
                pl = fr.get(c["kf"], {}).get("planes", [])
                labs = [lab_at[c["kf"]][i]["label"] for i, p in enumerate(pl) if p.get("anchor") == c["anchor"] and p.get("inlier")
                        and i in lab_at[c["kf"]]]
                ok = None if not labs or aL is None else (aL in labs)
            cl.append(dict(c, correct=ok))
        kfs = [int(q) for q in r["before"]]
        G = [gt[q] for q in kfs]
        a_b = 100 * ate([np.linalg.inv(np.array(r["before"][str(q)])) for q in kfs], G)[0] if len(kfs) >= 3 else None
        a_a = (100 * ate([np.linalg.inv(np.array(r["after"][str(q)])) for q in kfs], G)[0]
               if r.get("after") and len(kfs) >= 3 else None)
        shift = max(np.linalg.norm(np.linalg.inv(np.array(r["after"][str(q)]))[:3, 3] - np.linalg.inv(np.array(r["before"][str(q)]))[:3, 3])
                    for q in kfs) if r.get("after") else None
        rows.append(dict(frame=k, status=r["status"], n_t1=sum(c["tier"] == 1 for c in cl), n_t2=sum(c["tier"] == 2 for c in cl),
                         closures=cl, ate_before_cm=a_b, ate_after_cm=a_a,
                         max_shift_cm=None if shift is None else 100 * shift, solve_s=r.get("solve_s")))
    acc = [c for r in rows if r["status"] == "applied" for c in r["closures"] if not c["dropped"] and c["correct"] is not None]
    prec = 100.0 * np.mean([c["correct"] for c in acc]) if acc else None
    return rows, prec, len(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    res = {}
    for seq, sv in SPEC.items():
        mf = sv["max_frame"]
        res[seq] = {}
        for meth, run in sv["runs"].items():
            if a.only and f"{seq}:{meth}" not in a.only and seq not in a.only:
                continue
            if not os.path.exists(os.path.join(run, "traj_all.txt")):
                res[seq][meth] = dict(missing=run)
                continue
            m = run_metrics(run, mf)
            m["unaligned"] = m.pop("unaligned_cm")
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
            if kind == "pw":
                cr, n_good, obs, alab = correct_rejected(run, mf)
                m["gt"]["pct_correct_rejected"] = cr
                m["gt"]["n_correct_pairs"] = n_good
                if os.path.exists(os.path.join(run, "pgo_poses.jsonl")):
                    rows, prec, n_acc = pgo_table(run, mf, obs, alab)
                    m["pgo_table"] = rows
                    m["closure_precision"] = prec
                    m["closure_n_accepted_labeled"] = n_acc
            pf = [r["frame"] for r in jsonl(os.path.join(run, "pgo.jsonl")) if r["status"] == "applied" and (mf is None or r["frame"] < mf)]
            m["pgo_frames"] = pf
            res[seq][meth] = m
            print(seq, meth, f"ATE kf {m['ate_kf_cm']:.2f} all {m['ate_all_cm']:.2f} corr {m.get('ate_all_corr_cm')} "
                  f"rpe {m['rpe_d1_cm']:.3f} rpe_corr {m.get('rpe_d1_corr_cm')}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
