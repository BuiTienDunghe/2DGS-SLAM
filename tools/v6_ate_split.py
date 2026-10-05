"""plan v6 A2: ATE of each run over all frames and over the two segments split at a frame (default 1144).

usage: python tools/v6_ate_split.py --out DIR [--split 1144] RUN_DIR [RUN_DIR ...]
One SE(3) alignment (Umeyama without scale, as evo in utils/eval_utils.py) on the whole trajectory; the segment RMSEs
are taken under that same alignment. Poses come from final_state.pt (state before refinement). Writes DIR/ate_split.json.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch


def umeyama_rigid(est, ref):
    """R, t with ref ~ R est + t (least squares, no scale). est, ref: (N, 3)."""
    mu_e, mu_r = est.mean(0), ref.mean(0)
    S = (ref - mu_r).T @ (est - mu_e) / est.shape[0]
    U, _, Vt = np.linalg.svd(S)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    return R, mu_r - R @ mu_e


def ate_report(est, gt, split):
    """est, gt: {uid: 4x4 c2w}. RMSE in metres: all frames, uid <= split, uid > split (same alignment)."""
    ids = sorted(u for u in est if gt.get(u) is not None)
    pe = np.stack([np.asarray(est[u], dtype=np.float64)[:3, 3] for u in ids])
    pg = np.stack([np.asarray(gt[u], dtype=np.float64)[:3, 3] for u in ids])
    R, t = umeyama_rigid(pe, pg)
    err = np.linalg.norm(pe @ R.T + t - pg, axis=1)
    u = np.asarray(ids)
    rm = lambda m: float(np.sqrt((err[m] ** 2).mean())) if m.any() else None
    return {"all": rm(np.ones_like(u, bool)), "upto": rm(u <= split), "after": rm(u > split),
            "n_all": int(u.size), "n_upto": int((u <= split).sum()), "n_after": int((u > split).sum()),
            "sq_share_after": float((err[u > split] ** 2).sum() / (err ** 2).sum()) if (u > split).any() else 0.0}


def load_final(rd):
    f = torch.load(os.path.join(rd, "final_state.pt"), map_location="cpu", weights_only=False)
    est = {int(u): v.double().numpy() for u, v in f["poses"].items()}
    gt = {int(u): (None if v is None else v.double().numpy()) for u, v in f["poses_gt"].items()}
    return f, est, gt


def logged_ate(rd):
    for name in ("metrics_prerefine.csv", "metrics.csv"):
        p = os.path.join(rd, name)
        if os.path.exists(p):
            row = list(csv.DictReader(open(p)))[0]
            return float(row["ate_rmse_all_tracked_m"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", type=int, default=1144)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {}
    for rd in a.runs:
        rd = rd.rstrip("/")
        f, est, gt = load_final(rd)
        r = ate_report(est, gt, a.split)
        r["logged_all"] = logged_ate(rd)
        r["loop_pairs"] = [list(map(int, p)) for p in f.get("loop_uid_pairs", [])]
        out[os.path.basename(rd)] = r
        print("%-34s all %.2f cm (logged %s)  <=%d: %.2f cm (n %d)  >%d: %s cm (n %d, %.0f%% of squared error)  loops %s" % (
            os.path.basename(rd), 100 * r["all"], "n/a" if r["logged_all"] is None else f"{100 * r['logged_all']:.2f}",
            a.split, 100 * r["upto"], r["n_upto"], a.split, "n/a" if r["after"] is None else f"{100 * r['after']:.2f}",
            r["n_after"], 100 * r["sq_share_after"], r["loop_pairs"]))
    json.dump(out, open(os.path.join(a.out, "ate_split.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
