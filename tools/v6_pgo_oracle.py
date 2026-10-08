"""plan v6 A3 (Q3): how far could a better pose graph bring the ATE? Offline, CPU, one run at a time.

usage: python tools/v6_pgo_oracle.py --out DIR [--split 1144] RUN_DIR [RUN_DIR ...]
The final pose graph of the run is rebuilt (tools/pgo_replay.py): the fixed prior, one odometry factor per consecutive
pair of tracked frames with the measurement the backend used (poses as they were when the later frame was inserted;
frames after the last PGO event: the final poses) and the loop factors that stayed in the graph. Every variant edits
that graph, optimises ONCE from the run's final poses and reports the ATE over all frames and over the two segments
(one SE(3) alignment, as tools/v6_ate_split.py). The map/tracking feedback of the online system is not simulated.

  P0        nothing changed; must reproduce the run's own ATE (validity check of the replay)
  P-none    all loop factors removed (the as-tracked odometry chain)                         [outside the plan]
  P-keep    loop attempts that were removed again (error < 50) or rejected are put back; needs loop_transform in
            loop_attempts.jsonl (plan v6 X5), otherwise "not applicable"
  P-robust  loop sigma = s x odometry sigma, s in {1, 3, 10}, with and without a Huber kernel on the loop factors
            (s in {0.1, 0.3} is reported as an extra outside the plan)
  O-loop    every loop measurement replaced by the ground-truth relative pose (same candidates)
  O-cov     O-loop plus a ground-truth edge between every pair of keyframes more than 12 keyframes apart whose GT
            camera centres are < 0.5 m apart and whose viewing directions differ by < 30 degrees
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import gtsam  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deform.dump import list_dumps, load_dump  # noqa: E402
from pgo_replay import Replay, load_attempts, mat, rel, sym  # noqa: E402
from v6_ate_split import ate_report, logged_ate, umeyama_rigid  # noqa: E402

HUBER_K = 1.345


def noise(sig, scale=1.0, huber=False):
    base = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sig) * scale)
    if not huber:
        return base
    return gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K), base)


def build(R, ids, odom, edges):
    """edges: [((i, j), 4x4 measurement T_i<-j, noise model)]."""
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.PriorFactorPose3(sym(ids[0]), gtsam.Pose3(R.prior_pose), R.fixed))
    for a, b in zip(ids[:-1], ids[1:]):
        g.add(gtsam.BetweenFactorPose3(sym(a), sym(b), gtsam.Pose3(odom[(a, b)]), R.cov))
    for (i, j), m, nm in edges:
        g.add(gtsam.BetweenFactorPose3(sym(i), sym(j), gtsam.Pose3(np.asarray(m, dtype=np.float64)), nm))
    return g


def optimise(g, v, max_iter):
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(max_iter)
    opt = gtsam.LevenbergMarquardtOptimizer(g, v, p)
    x = opt.optimizeSafely()
    return x, int(opt.iterations())


def run_one(rd, split, max_iter):
    rd = rd.rstrip("/")
    name = os.path.basename(rd)
    dumps = list_dumps(rd)
    if not dumps:
        print(f"{name}: no loop dumps -> A3 not applicable")
        return {"note": "no loop dumps"}
    D = [load_dump(p) for p in dumps]
    R = Replay(D, load_attempts(rd), D[0]["config"]["Training"])
    loops, checks = R.recover_all()
    final = torch.load(os.path.join(rd, "final_state.pt"), map_location="cpu", weights_only=False)
    P_fin = {int(u): v.double().numpy() for u, v in final["poses"].items()}
    P_gt = {int(u): (None if v is None else v.double().numpy()) for u, v in final["poses_gt"].items()}
    ids_evt = [int(u) for u in D[-1]["all_cam_ids"]]
    tail = [u for u in sorted(P_fin) if u > ids_evt[-1]]
    ids = ids_evt + tail
    missing = sorted(set(u for u in P_fin if u <= ids_evt[-1]) ^ set(ids_evt))
    odom = R.odom_tracked(ids_evt)
    for a, b in zip(ids[len(ids_evt) - 1:-1], ids[len(ids_evt):]):
        odom[(a, b)] = rel(P_fin[a], P_fin[b])
    v0 = R.values(ids, P_fin)
    sig = R.sig
    kept = [(k, R.loops[k]) for k in R.loop_order]

    def gt_rel(i, j):
        return np.linalg.inv(P_gt[i]) @ P_gt[j]

    def evaluate(edges, label):
        g = build(R, ids, odom, edges)
        e0 = float(g.error(v0))
        x, it = optimise(g, v0, max_iter)
        est = {u: x.atPose3(sym(u)).matrix() for u in ids}
        r = ate_report(est, P_gt, split)
        if label in ("P0", "O-loop", "O-cov"):  # per-frame error under the variant's own alignment (for the report)
            ug = [u for u in ids if P_gt.get(u) is not None]
            pe = np.stack([est[u][:3, 3] for u in ug])
            pg = np.stack([P_gt[u][:3, 3] for u in ug])
            Ra, ta = umeyama_rigid(pe, pg)
            r["per_frame"] = {"uid": ug, "err_m": np.linalg.norm(pe @ Ra.T + ta - pg, axis=1).tolist()}
        r.update({"err_before": e0, "err_after": float(g.error(x)), "lm_iters": it, "n_edges": len(edges),
                  "max_move_m": float(max(np.linalg.norm(est[u][:3, 3] - P_fin[u][:3, 3]) for u in ids))})
        print("  %-22s ATE all %.2f cm  <=%d %.2f  >%d %s  | edges %3d  graph err %.1f -> %.1f (%d it)  max move %.3f m" % (
            label, 100 * r["all"], split, 100 * r["upto"], split, "n/a" if r["after"] is None else f"{100 * r['after']:.2f}",
            len(edges), e0, r["err_after"], it, r["max_move_m"]))
        return r

    online = ate_report(P_fin, P_gt, split)
    print(f"== {name}: frames {len(ids)} (after last event: {len(tail)}), kept loops {[f'{c}<->{l}' for (l, c), _ in kept]}, "
          f"online ATE {100 * online['all']:.2f} cm (logged {logged_ate(rd)})" + (f", id mismatch {missing[:6]}" if missing else ""))
    out = {"online": online, "logged_ate_all": logged_ate(rd), "n_frames": len(ids), "n_tail": len(tail),
           "kept_loops": [[int(c), int(l)] for (l, c), _ in kept], "loop_checks": {f"{c}<->{l}": v for (l, c), v in checks.items()},
           "variants": {}}
    V = out["variants"]
    V["P0"] = evaluate([(k, m, noise(sig)) for k, m in kept], "P0")
    V["P-none"] = evaluate([], "P-none (extra)")
    # P-keep: attempts that are not in the graph and carry their measurement (X5)
    rows = [json.loads(l) for l in open(os.path.join(rd, "loop_attempts.jsonl")) if l.strip()]
    kept_keys = {k for k, _ in kept}
    dropped = [r for r in rows if "loop_transform" in r and (int(r["loop_uid"]), int(r["cur_uid"])) not in kept_keys]
    out["attempts"] = [{"cur": int(r["cur_uid"]), "loop": int(r["loop_uid"]), "in_graph": bool(r.get("loop_factor_in_graph")),
                        "reason": r.get("reason"), "err_with_loop": r.get("graph_err_with_loop")} for r in rows]
    if dropped:
        small = [r for r in dropped if r.get("reason") == "removed_small_error"]
        for lab, sel in (("P-keep (small)", small), ("P-keep (all)", dropped)):
            if sel:
                extra = [((int(r["loop_uid"]), int(r["cur_uid"])), np.asarray(r["loop_transform"]), noise(sig)) for r in sel]
                V[lab] = evaluate([(k, m, noise(sig)) for k, m in kept] + extra, lab)
                V[lab]["added"] = [[int(r["cur_uid"]), int(r["loop_uid"]), r.get("reason")] for r in sel]
    else:
        n_not_in_graph = sum(1 for r in rows if not r.get("loop_factor_in_graph", True))
        out["P-keep"] = ("not applicable: no dropped attempt" if n_not_in_graph == 0
                         else f"not applicable: {n_not_in_graph} dropped attempt(s) without a stored measurement")
        print("  P-keep:", out["P-keep"])
    for s in (0.1, 0.3, 1, 3, 10):
        for hub in (False, True):
            if s == 1 and not hub:
                continue
            lab = f"P-robust s={s}{' +Huber' if hub else ''}" + (" (extra)" if s < 1 else "")
            V[lab] = evaluate([(k, m, noise(sig, s, hub)) for k, m in kept], lab)
    V["O-loop"] = evaluate([(k, gt_rel(*k), noise(sig)) for k, _ in kept], "O-loop")
    kfs = sorted(int(u) for u in final["keyframe_uids"])
    C = np.stack([P_gt[u][:3, 3] for u in kfs])
    Z = np.stack([P_gt[u][:3, 2] for u in kfs])
    cov = []
    for a in range(len(kfs)):
        for b in range(a + 13, len(kfs)):
            if np.linalg.norm(C[a] - C[b]) < 0.5 and float(Z[a] @ Z[b]) > np.cos(np.radians(30.0)):
                cov.append(((kfs[a], kfs[b]), gt_rel(kfs[a], kfs[b]), noise(sig)))
    V["O-cov"] = evaluate([(k, gt_rel(*k), noise(sig)) for k, _ in kept] + cov, "O-cov")
    V["O-cov"]["n_gt_edges"] = len(cov)
    V["O-cov"]["gt_edge_pairs"] = [[int(i), int(j)] for (i, j), _, _ in cov]
    out["keyframe_uids"] = kfs
    # outside the plan: the same two oracles with the GT edges trusted 10x more than odometry, to tell a limit set by
    # the edge weights from a limit set by where the edges are
    V["O-loop s=0.1 (extra)"] = evaluate([(k, gt_rel(*k), noise(sig, 0.1)) for k, _ in kept], "O-loop s=0.1 (extra)")
    V["O-cov s=0.1 (extra)"] = evaluate([(k, gt_rel(*k), noise(sig, 0.1)) for k, _ in kept]
                                        + [(k, m, noise(sig, 0.1)) for k, m, _ in cov], "O-cov s=0.1 (extra)")
    V["O-cov s=0.1 (extra)"]["n_gt_edges"] = len(cov)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", type=int, default=1144)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {os.path.basename(rd.rstrip("/")): run_one(rd, a.split, a.max_iter) for rd in a.runs}
    json.dump(out, open(os.path.join(a.out, "pgo_oracle.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
