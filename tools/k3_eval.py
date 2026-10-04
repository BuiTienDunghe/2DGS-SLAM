"""plan v4 K3: F1o-hash against F1o (random cap) with the sampling spread of both draws.

usage: python tools/k3_eval.py --diag results_exp/reports/v4/diag --v3 results_exp/reports/v3 --out results_exp/reports/v4/k3
Per event, R_X = e_dl reduction vs rigid on the Pi* intersection over rigid + every config of the V4 run
(aggregate_diag_v2.per_event). Families: hash = {F1o-hash, F1o-hash-s1..s4}, random = {F1o, F1o-r1..r4}
(Replica events only have F1o and F1o-hash).
Calibrated K3 threshold (user decision 2026-10-03 (4)): thr_event = max(1 point, 2 * sqrt(s_h^2/n_h + s_r^2/n_r))
with the sample standard deviations of the two families; pass iff |mean_hash - mean_random| <= thr_event on
every TUM event, plus the plan's other K3 rules (wrong pairs <= 10 %, A1-A4 as in v3, Replica e_dl and Acc
within 5 % of v3's F1o). Writes OUT/k3.json and prints the table.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from aggregate_diag_v2 import load_rows, per_event, red  # noqa: E402

HASH = ["F1o-hash", "F1o-hash-s1", "F1o-hash-s2", "F1o-hash-s3", "F1o-hash-s4"]
RAND = ["F1o", "F1o-r1", "F1o-r2", "F1o-r3", "F1o-r4"]


def acc_checks(row):
    ac = row.get("accept_checks") or {}
    return {k: ac.get(k) for k in ("A1", "A2", "A3", "A4")}, row.get("accepted"), row.get("reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--v3", default=None, help="results_exp/reports/v3 (agg/summary_v3.json + diag/rows.jsonl)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_rows(os.path.join(a.diag, "rows.jsonl"))
    v3R, v3rows = {}, {}
    if a.v3:
        p = os.path.join(a.v3, "agg", "summary_v3.json")
        if os.path.exists(p):
            v3R = json.load(open(p))["R"].get("F1o", {})
        p = os.path.join(a.v3, "diag", "rows.jsonl")
        if os.path.exists(p):
            v3rows = load_rows(p)
    out = {"events": {}, "pass": True, "thresholds": {}}
    print(f"{'event':28s} {'scene':8s} {'R hash (5)':>34s} {'R rand (5)':>34s} {'mean_h':>7s} {'mean_r':>7s} {'diff':>6s} {'thr':>5s} {'v3 F1o':>7s}")
    for mp in sorted(glob.glob(os.path.join(a.diag, "maps", "*.pt"))):
        dump = os.path.basename(mp)[:-3]
        store = torch.load(mp, weights_only=False)
        cfgs = [c for c in HASH + RAND if c in store["configs"]]
        if not cfgs:
            continue
        ev = per_event(store, cfgs, cfgs)
        R = {c: red(ev, c) for c in cfgs}
        scene = rows[(dump, "rigid")]["scene"]
        rh = [100 * R[c] for c in HASH if c in R]
        rr = [100 * R[c] for c in RAND if c in R]
        mh, mr = float(np.mean(rh)), float(np.mean(rr))
        sh = float(np.std(rh, ddof=1)) if len(rh) > 1 else 0.0
        sr = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
        thr = max(1.0, 2.0 * float(np.sqrt(sh ** 2 / max(1, len(rh)) + sr ** 2 / max(1, len(rr)))))
        diff = mh - mr
        tum = scene.startswith("tum")
        e = {"scene": scene, "n_pix": ev["n_pix"], "R": {c: 100 * R[c] for c in cfgs}, "mean_hash": mh, "mean_random": mr,
             "sd_hash": sh, "sd_random": sr, "range_hash": [min(rh), max(rh)], "range_random": [min(rr), max(rr)],
             "diff_points": diff, "thr_points": thr, "v3_F1o_R": None if dump not in v3R else 100 * v3R[dump],
             "diff_vs_v3": None if dump not in v3R else 100 * R["F1o-hash"] - 100 * v3R[dump]}
        # plan rules on the F1o-hash row
        r = rows[(dump, "F1o-hash")]
        chk, acc, reason = acc_checks(r)
        e["wrong_pairs_pct"] = None if r.get("frac_res_gt30mm") is None else 100 * r["frac_res_gt30mm"]
        e["accept_checks"] = chk
        e["accepted"] = (acc, reason)
        if (dump, "F1o") in v3rows:
            chk3, acc3, reason3 = acc_checks(v3rows[(dump, "F1o")])
            e["v3_accept_checks"] = chk3
            e["A_same_as_v3"] = (chk == chk3) and (acc == acc3)
            e["v3_acc"] = v3rows[(dump, "F1o")].get("acc")
        e["acc"] = r.get("acc")
        e["ate"] = (r.get("ate"), rows[(dump, "rigid")].get("ate"))
        e["psnr"] = (r.get("psnr"), rows[(dump, "rigid")].get("psnr"))
        e["t_event_s"] = (r.get("t_stage_s") or {}).get("total")
        e["anchor"] = r.get("anchor")
        e["det_warnings"] = r.get("det_warnings")
        ok = True
        if tum:
            ok &= abs(diff) <= thr
            e["K3_main"] = abs(diff) <= thr
            if e["diff_vs_v3"] is not None:
                e["K3_vs_v3_1pt"] = abs(e["diff_vs_v3"]) <= 1.0
                e["K3_vs_v3_stop_3pt"] = e["diff_vs_v3"] < -3.0
        else:
            rel_e = None if e["v3_F1o_R"] is None else (100 * R["F1o-hash"] - e["v3_F1o_R"])
            rel_a = None if (e.get("v3_acc") is None or e["acc"] is None or e["v3_acc"] == 0) else (e["acc"] - e["v3_acc"]) / abs(e["v3_acc"])
            e["replica_dR_points_vs_v3"] = rel_e
            e["replica_dAcc_rel_vs_v3"] = rel_a
            # "e_dl lệch <= 5 %": read as the relative change of the e_dl median of F1o-hash vs v3's F1o
            e["K3_replica"] = (rel_a is None or abs(rel_a) <= 0.05)
            ok &= e["K3_replica"]
        ok &= e["wrong_pairs_pct"] is not None and e["wrong_pairs_pct"] <= 10.0
        e["K3_pass"] = bool(ok)
        out["pass"] &= bool(ok)
        out["events"][dump] = e
        print(f"{dump:28s} {scene[:8]:8s} {str([round(x, 1) for x in rh]):>34s} {str([round(x, 1) for x in rr]):>34s} "
              f"{mh:7.1f} {mr:7.1f} {diff:+6.1f} {thr:5.1f} {('' if e['v3_F1o_R'] is None else round(e['v3_F1o_R'], 1))!s:>7s}  "
              f"wrong {e['wrong_pairs_pct']:.1f}% acc {acc} {'OK' if ok else 'MISS'}")
    with open(os.path.join(a.out, "k3.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("K3_PASS" if out["pass"] else "K3_MISS")


if __name__ == "__main__":
    main()
