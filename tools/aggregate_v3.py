"""Plan v3 steps B-C: aggregate the rerun with the new solver (3 TUM + 2 Replica events) and apply the
selection rule of handoffPlan_v3_solver.md (step C).

usage: python tools/aggregate_v3.py --diag results_exp/reports/v3/diag --out results_exp/reports/v3/agg [--no-oracle-set]
Pi* intersection per event as in v2: a pixel is kept when rigid and every config of the run still render
both layers there (--no-oracle-set: intersection without O1, sensitivity).
Rule (step C), candidates F6, F1o, F1t, F1w, F1t+F6, F1o+F6 against A-1c (= A-1 with the new solver):
  C1 e_dl better than A-1c in each of the 3 TUM events (same pixels)
  C2 wrong-pair proxy (residual > 30 mm after the solve) <= 10 % in every event (TUM and Replica)
  C3 keyframe ATE not worse than rigid by more than max(5 %, 1 mm) in every event (plan v1 S3 semantics)
  C4 median dPSNR vs rigid >= -0.3 dB (per scene)
  C5 Replica: e_dl and Acc not worse than A-1c by more than 5 % (each Replica event)
Pick: highest median R on TUM; within 3 points the less relaxed (F6 < F1o < F1t < F1w < combinations);
none passes -> A-1c.
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

from aggregate_diag_v2 import load_rows, per_event, red, rel_red  # noqa: E402

REF = "A-1-ref"  # A-1c in the report
SET_V3 = ["A-1-ref", "O1", "F1t", "F1o", "F1w", "F6", "F1t+F6", "F1o+F6"]
CANDIDATES = ["F6", "F1o", "F1t", "F1w", "F1t+F6", "F1o+F6"]
RELAX = {"F6": 0, "F1o": 1, "F1t": 2, "F1w": 3, "F1t+F6": 4, "F1o+F6": 4}


def scene_of(rows, dump):
    return rows[(dump, "rigid")]["scene"]


def criteria(evs, rows, name):
    tum = [(d, e) for d, e in evs if scene_of(rows, d).startswith("tum")]
    rep = [(d, e) for d, e in evs if scene_of(rows, d).startswith("replica")]
    out = {}
    per = [(red(e, name), red(e, REF)) for _, e in tum]
    out["C1"] = ([None if a is None or b is None else round(100 * (a - b), 2) for a, b in per],
                 bool(per) and all(a is not None and b is not None and a > b for a, b in per))
    px = [rows[(d, name)].get("frac_res_gt30mm") for d, _ in evs]
    out["C2"] = ([None if x is None else round(100 * x, 1) for x in px], all(x is not None and x <= 0.10 for x in px))
    worst = []
    for d, _ in evs:
        a, b = rows[(d, name)].get("ate"), rows[(d, "rigid")].get("ate")
        worst.append(None if a is None or b is None else (a - b) - max(0.05 * b, 0.001))
    out["C3"] = ([None if w is None else round(1e3 * w, 3) for w in worst], all(w is not None and w <= 0 for w in worst))
    c4 = {}
    for lab, group in (("tum", tum), ("replica", rep)):
        dp = [rows[(d, name)]["psnr"] - rows[(d, "rigid")]["psnr"] for d, _ in group
              if rows[(d, name)].get("psnr") is not None and rows[(d, "rigid")].get("psnr") is not None]
        if dp:
            c4[lab] = float(np.median(dp))
    out["C4"] = ({k: round(v, 3) for k, v in c4.items()}, bool(c4) and all(v >= -0.3 for v in c4.values()))
    c5v, c5ok = [], True
    for d, e in rep:
        ex, er = e[name]["median"], e[REF]["median"]
        ax, ar = rows[(d, name)].get("acc"), rows[(d, REF)].get("acc")
        r_edl = None if ex is None or er is None or er == 0 else ex / er - 1
        r_acc = None if ax is None or ar is None or ar == 0 else ax / ar - 1
        c5v.append({"edl_vs_A1c_%": None if r_edl is None else round(100 * r_edl, 2),
                    "acc_vs_A1c_%": None if r_acc is None else round(100 * r_acc, 2)})
        c5ok &= r_edl is not None and r_edl <= 0.05 and (r_acc is None or r_acc <= 0.05)
    out["C5"] = (c5v, bool(rep) and c5ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-oracle-set", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_rows(os.path.join(a.diag, "rows.jsonl"))
    evs = []
    for mp in sorted(glob.glob(os.path.join(a.diag, "maps", "*.pt"))):
        store = torch.load(mp, weights_only=False)
        dump = os.path.basename(mp)[:-3]
        configs = [c for c in SET_V3 if c in store["configs"]]
        set_cfgs = [c for c in configs if not (a.no_oracle_set and c.startswith("O1"))]
        e = per_event(store, configs, set_cfgs)
        e["_dump"], e["_scene"], e["_ev"] = dump, scene_of(rows, dump), rows[(dump, "rigid")]["event_id"]
        evs.append((dump, e))
    configs = [c for c in SET_V3 if all(c in e for _, e in evs)]
    tum = [(d, e) for d, e in evs if e["_scene"].startswith("tum")]
    R = {c: {d: red(e, c) for d, e in evs} for c in configs}
    R_tum = {c: float(np.median([R[c][d] for d, _ in tum if R[c][d] is not None])) for c in configs}
    G = {c: [None if not red(e, "O1") else red(e, c) / red(e, "O1") for _, e in tum] for c in configs} if "O1" in configs else {}
    table = {c: criteria(evs, rows, c) for c in CANDIDATES if c in configs}
    passed = [c for c in CANDIDATES if c in table and all(ok for _, ok in table[c].values())]
    chosen = None
    if passed:
        best = max(R_tum[c] for c in passed)
        near = [c for c in passed if R_tum[c] >= best - 0.03]
        chosen = sorted(near, key=lambda c: (RELAX[c], -R_tum[c]))[0]
    pick = chosen or "A-1c"
    oracle = None
    if "O1" in configs:
        g_a1 = [g for g in G[REF] if g is not None]
        oracle = {"R_O1_median_tum": R_tum["O1"], "G_A1c_median": float(np.median(g_a1)) if g_a1 else None,
                  "missing_pairs_main_cause": R_tum["O1"] >= 0.5 and bool(g_a1) and float(np.median(g_a1)) <= 0.5}
    summary = {"configs": configs, "intersection": "no-oracle" if a.no_oracle_set else "all v3 configs",
               "R": R, "R_median_tum": R_tum, "G": G, "criteria": table, "passed": passed, "chosen": pick,
               "oracle": oracle, "per_event": {d: e for d, e in evs},
               "rows": {f"{d}|{c}": rows[(d, c)] for d, _ in evs for c in ["rigid"] + configs if (d, c) in rows}}
    with open(os.path.join(a.out, "summary_v3.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("events:", [(e["_scene"], e["_ev"], e["n_pix"], e["n_pi"]) for _, e in evs])
    for c in ["rigid"] + configs:
        cells = []
        for d, e in evs:
            x = e[c]["median"]
            rr = red(e, c)
            cells.append(f"{1e3 * x:7.2f}mm{('' if rr is None else f'{100 * rr:+6.1f}%')}")
        extra = "" if c == "rigid" else f" | medR_TUM {100 * R_tum[c]:5.1f}%  alpha-loss% " + ",".join(
            f"{100 * e['alpha_loss'][c]:.0f}" for _, e in evs)
        print(f"{c:9s} " + " ".join(cells) + extra)
    for c, t in table.items():
        print(f"{c:9s}", {k: (v, ok) for k, (v, ok) in t.items()})
    print("oracle:", oracle)
    print("passed:", passed, "-> chosen:", pick)


if __name__ == "__main__":
    main()
