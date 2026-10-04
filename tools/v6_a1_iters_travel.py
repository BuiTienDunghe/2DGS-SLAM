"""plan v6 A1 (Q4), from data that existed before the plan: tracking iterations per call and the travel floor.

usage: python tools/v6_a1_iters_travel.py --out DIR --logs LOG [LOG ...] --trajs TRAJ_TUM [TRAJ_TUM ...]
(1) Histogram of `track frame=… iters=k/n` lines of the run logs (tracking and relocalisation calls).
(2) Travel floor from final trajectories (TUM format, c2w): the tracker updates T <- Exp(tau) T with Adam, i.e. at
    most about one learning rate per axis per iteration, so going from the initial pose to the final one needs at
    least k_min = max_axis(|rho| / lr_t, |phi| / lr_r) iterations, D = T_final T_init^-1 = Exp((rho, phi)).
    "prev": T_init = previous frame; "const_vel": T_init = T_a T_b^-1 T_a (two previous frames).
(3) Lag simulation under an iteration cap K: what cannot be travelled in K iterations is carried to the next frame
    (frame-to-model tracking catches up); additive in the tangent space, no tracking loss, keyframes at lagged poses
    do not damage the map. Rough by construction.
Writes DIR/a1.json.
"""
import argparse
import json
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as Rot

LR = np.array([0.0015] * 3 + [0.003] * 3)


def load_w2c(p):
    a = np.loadtxt(p)
    T = np.tile(np.eye(4), (len(a), 1, 1))
    T[:, :3, :3] = Rot.from_quat(a[:, 4:8]).as_matrix()
    T[:, :3, 3] = a[:, 1:4]
    return np.linalg.inv(T)


def vec(D):
    return np.concatenate([D[:, :3, 3], Rot.from_matrix(D[:, :3, :3]).as_rotvec()], 1)


def q(x, qs=(50, 80, 95)):
    return [float(v) for v in np.percentile(x, qs)]


def lag_sim(d, K, cv):
    cap = K * LR
    L1, L2, out = np.zeros(6), np.zeros(6), []
    for x in d:
        need = x + ((2 * L1 - L2) if cv else L1)
        lag = need - np.clip(need, -cap, cap)
        out.append(lag)
        L2, L1 = L1, lag
    out = np.array(out)
    lt, lr = 1e3 * np.linalg.norm(out[:, :3], axis=1), np.degrees(np.linalg.norm(out[:, 3:], axis=1))
    return {"frac_lagging": float(((lt > 0) | (lr > 0)).mean()), "rms_mm": float(np.sqrt((lt ** 2).mean())), "max_mm": float(lt.max()),
            "rms_deg": float(np.sqrt((lr ** 2).mean())), "max_deg": float(lr.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--trajs", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {"iters": {}, "travel": {}}
    pat = re.compile(r"track frame=(\d+) iters=(\d+)/(\d+)")
    for p in a.logs:
        it = [(int(m.group(2)), int(m.group(3))) for m in pat.finditer(open(p, errors="replace").read())]
        k = np.array([x for x, _ in it])
        n = np.array([x for _, x in it])
        full = k >= n
        name = os.path.basename(p).replace(".log", "")
        out["iters"][name] = {"n_calls": int(k.size), "frac_full": float(full.mean()), "mean": float(k.mean()),
                              "early_median": float(np.median(k[~full])), "early_min": int(k[~full].min()), "iters": k.tolist()}
        print("%-44s calls %d  full %.1f%%  mean %.1f  early: median %.0f min %d" % (
            name, k.size, 100 * full.mean(), k.mean(), np.median(k[~full]), k[~full].min()))
    for p in a.trajs:
        W = load_w2c(p)
        dp = vec(W[1:] @ np.linalg.inv(W[:-1]))
        P = W[1:-1] @ np.linalg.inv(W[:-2]) @ W[1:-1]
        dc = vec(W[2:] @ np.linalg.inv(P))
        name = os.path.basename(os.path.dirname(p))
        res = {}
        for lab, d in (("prev", dp), ("const_vel", dc)):
            k = np.max(np.abs(d) / LR, axis=1)
            res[lab] = {"travel_mm_p50_p80_p95": q(1e3 * np.linalg.norm(d[:, :3], axis=1)),
                        "travel_deg_p50_p80_p95": q(np.degrees(np.linalg.norm(d[:, 3:], axis=1))),
                        "k_min_p50_p80_p95": q(k), "frac_over_15": float((k > 15).mean()), "frac_over_30": float((k > 30).mean()),
                        "k_min": k.tolist()}
            print("%-28s %-9s travel med %.1f mm / %.1f deg  k_min p50/p80/p95 %.1f/%.1f/%.1f  >15: %.0f%%  >30: %.0f%%" % (
                name, lab, res[lab]["travel_mm_p50_p80_p95"][0], res[lab]["travel_deg_p50_p80_p95"][0], *res[lab]["k_min_p50_p80_p95"],
                100 * res[lab]["frac_over_15"], 100 * res[lab]["frac_over_30"]))
        res["lag"] = {"R5 prev,30": lag_sim(dp, 30, False), "R6 const_vel,30": lag_sim(dc, 30, True),
                      "R7 const_vel,15": lag_sim(dc, 15, True), "prev,15": lag_sim(dp, 15, False)}
        for lab, v in res["lag"].items():
            print("    lag %-16s frames %.1f%%  rms %.2f mm max %.1f mm  rms %.3f deg max %.2f deg" % (
                lab, 100 * v["frac_lagging"], v["rms_mm"], v["max_mm"], v["rms_deg"], v["max_deg"]))
        out["travel"][name] = res
    json.dump(out, open(os.path.join(a.out, "a1.json"), "w"))


if __name__ == "__main__":
    main()
