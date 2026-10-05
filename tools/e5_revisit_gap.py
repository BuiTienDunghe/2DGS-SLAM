"""plan v6 quick, E5 post-analysis: do the missed revisit bursts carry a real two-layer gap, and did the keyframe-0
fix close the loop back to the start?

usage: python tools/e5_revisit_gap.py --out DIR [--floor-mm X] [--baseline RUN_DIR] E5_RUN_DIR
1 revisit dumps (revisit_dumps/rv_*.pt): layers by birth time with the split s = (candidate keyframe + requesting
  frame) / 2, rendered from the TRACKED pose of the requesting frame; pairs by the pipeline's own pair builder
  (deform.correspondences.build_pairs, selected_v6, reliability S computed as in the pipeline on the keyframes of
  the dump); number of pairs, median / p90 gap |n . (x_new - x_old)|, share of the grid pixels that show both layers.
  "raw" = the same without the 50 mm distance gate only (a real offset above 50 mm would otherwise be filtered out).
2 bursts from loop_checks.jsonl (gate-passing frames at most 2 tracked frames apart): frames, candidate keyframe,
  largest observed_ratio, outcome, gap of its dumps; number of bursts for the thresholds 0,3 / 0,4 / 0,5; the full
  distribution of observed_ratio.
3 keyframe 0: candidates, relocalisation results, accepted loops.
4 ATE (keyframes / all frames) and the RMSE of the segment after frame 1144 (tools/v6_ate_split.py), E5 vs baseline.
Writes DIR/e5.json.
"""
import argparse
import collections
import copy
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402


def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def bursts_of(frames, order, gap=2):
    """frames: sorted frame ids passing; order: {frame: index among the checked frames}."""
    out, cur = [], []
    for f in frames:
        if cur and order[f] - order[cur[-1]] > gap + 1:
            out.append(cur)
            cur = []
        cur.append(f)
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor-mm", type=float, default=None, help="event-time noise floor from E3 (median, mm)")
    ap.add_argument("--baseline", default=None, help="baseline run to compare the ATE with")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-gap", action="store_true", help="skip the GPU part (dump gaps)")
    ap.add_argument("run")
    a = ap.parse_args()
    rd = a.run.rstrip("/")
    os.makedirs(a.out, exist_ok=True)
    res = {"run": os.path.basename(rd)}

    # ------------------------------------------------ 1 gaps at the revisit dumps (GPU)
    gaps = {}
    if not a.no_gap:
        import deform  # noqa: F401
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from deform import config as dcfg
        from deform import metrics as M
        from deform.correspondences import build_pairs
        from deform.dump import EventState, load_dump
        from deform.pipeline import delta_t_frames, inp_from_dump
        from deform.reliability import compute_reliability
        from deform.render_utils import DEV, det_scope, make_cam, torch_det_scope
        from utils.loop_dump import load_deform_config

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        user = load_deform_config(a.config or os.path.join(root, "configs", "deform", "selected_v6.yaml"))
        for p in sorted(glob.glob(os.path.join(rd, "revisit_dumps", "rv_*.pt"))):
            d = load_dump(p)
            req = d["request"]
            C, cand = int(req["uid"]), req.get("cand_kf")
            row = {"file": os.path.basename(p), "tag": req["tag"], "frame": C, "cand_kf": cand, "observed_ratio": req.get("observed_ratio"),
                   "latest_kf_uid": req.get("latest_kf_uid"), "n_gauss": int(d["gauss_pre"]["xyz"].shape[0])}
            if cand is None:
                row["note"] = "no candidate keyframe"
                gaps[req["tag"]] = row
                continue
            st = EventState(d)
            cfg = dcfg.resolve(user, st.config)
            inp = inp_from_dump(d, st.frame)
            kf_uids = sorted(inp["kf_uids"])
            g = inp["g"]
            split = 0.5 * (float(cand) + float(C))
            m_old, m_new = M.layer_masks_birth(inp["t0"], split)
            row.update({"split_t0": split, "n_old": int(m_old.sum()), "n_new": int(m_new.sum())})
            sink = []
            with torch.no_grad(), det_scope(True), torch_det_scope(True, sink):
                dt = delta_t_frames(kf_uids, inp["W"])
                cams = {u: make_cam(u, inp["poses_pre"][u], inp["intr"]) for u in kf_uids}
                rel = compute_reliability(g, cams, inp["frame_fn"], inp["t0"], dt, cfg, inp["tr"], kf_uids)
                del cams
                cam = make_cam(C, req["pose_c2w"], inp["intr"])
                x = g["xyz"].float()
                eye = torch.eye(3, device=DEV)[None].expand(x.shape[0], 3, 3)
                for name, c2 in (("gated", cfg), ("raw", dcfg._merge(cfg, {"corr": {"eps_d": 1.0e9}}))):
                    pr = build_pairs({C: cam}, g, m_old, m_new, rel["S"], [(C, None)], inp["t0"], x, x, eye, c2, inp["seed"])
                    o = {"pairs": int(pr["P"]), "both_layers_frac": pr["n_both_opaque"] / max(1, pr["n_raw"]), "n_grid": int(pr["n_raw"]),
                         "gate_counts": pr["gate_counts"]}
                    if pr["P"] > 0:
                        gap = (pr["n"] * (pr["x_new"].double() - pr["x_old"].double())).sum(-1).abs()
                        o.update({"gap_med_mm": 1e3 * float(gap.median()), "gap_p90_mm": 1e3 * float(torch.quantile(gap, 0.9)),
                                  "frac_gt_50mm": float((gap > 0.05).double().mean())})
                    dd = pr.get("dD_vals")
                    if dd is not None and len(dd):
                        o.update({"dD_med_mm": 1e3 * float(np.median(dd)), "dD_p90_mm": 1e3 * float(np.quantile(dd, 0.9))})
                    row[name] = o
            if a.floor_mm and row.get("gated", {}).get("gap_med_mm") is not None:
                row["gated_over_floor"] = row["gated"]["gap_med_mm"] / a.floor_mm
                row["raw_over_floor"] = row["raw"]["gap_med_mm"] / a.floor_mm
            gaps[req["tag"]] = row
            gt, rw = row.get("gated", {}), row.get("raw", {})
            print(f"dump {req['tag']} frame {C} cand kf {cand} ratio {req.get('observed_ratio'):.3f}  both layers {100 * gt.get('both_layers_frac', 0):.0f} % of grid  "
                  f"gated: {gt.get('pairs')} pairs, gap {gt.get('gap_med_mm')} / p90 {gt.get('gap_p90_mm')} mm   raw: {rw.get('pairs')} pairs, gap {rw.get('gap_med_mm')} / p90 {rw.get('gap_p90_mm')} mm", flush=True)
            del g, inp, rel
            torch.cuda.empty_cache()
    res["dumps"] = gaps
    dl = jsonl(os.path.join(rd, "revisit_dumps.jsonl"))
    res["dump_time"] = {"n": len(dl), "total_s": dl[-1]["t_total_s"] if dl else 0.0, "max_s": max([x["t_dump_s"] for x in dl], default=None)}

    # ------------------------------------------------ 2 bursts and the distribution of observed_ratio
    chk = [r for r in jsonl(os.path.join(rd, "loop_checks.jsonl")) if r.get("path") == "revisit"]
    att = jsonl(os.path.join(rd, "loop_attempts.jsonl"))
    mast = jsonl(os.path.join(rd, "mast3r_calls.jsonl"))
    order = {r["frame"]: i for i, r in enumerate(chk)}
    ratio = {r["frame"]: r["observed_ratio"] for r in chk}
    cand = {r["frame"]: r.get("cand_kf") for r in chk}
    acc = {int(x["cur_uid"]): x for x in att}
    res["n_checked_frames"] = len(chk)
    v = np.array([r["observed_ratio"] for r in chk]) if chk else np.zeros(0)
    if len(v):
        res["observed_ratio"] = {"quantiles": {str(q): float(np.quantile(v, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}, "max": float(v.max()),
                                 "hist_edges": [round(0.05 * i, 2) for i in range(21)], "hist": np.histogram(v, bins=np.linspace(0, 1, 21))[0].tolist()}
    res["bursts_by_threshold"] = {}
    for thr in (0.3, 0.4, 0.5):
        fr = sorted(f for f, x in ratio.items() if x > thr)
        bs = bursts_of(fr, order)
        res["bursts_by_threshold"][str(thr)] = {"n_frames": len(fr), "n_bursts": len(bs), "n_bursts_without_loop": sum(1 for b in bs if not any(f in acc for f in b)),
                                                "bursts": [[b[0], b[-1], len(b)] for b in bs]}
    table = []
    by_frame = collections.defaultdict(list)
    for m in mast:
        by_frame[m["frame"]].append(m)
    for b in bursts_of(sorted(f for f, r in zip(order, chk) if r["pass"]), order):
        cands = collections.Counter(cand[f] for f in b if cand[f] is not None)
        mm = [m for f in b for m in by_frame.get(f, []) if m["path"] == "revisit"]
        dumps = [gaps[r["dump"]] for r in chk if r["frame"] in set(b) and r.get("dump") and r["dump"] in gaps]
        loops = [(f, int(acc[f]["loop_uid"]), bool(acc[f].get("pgo_ran"))) for f in b if f in acc]
        table.append({"start": b[0], "end": b[-1], "n_frames": len(b), "cand_kf": [k for k, _ in cands.most_common(3)], "cand_kf_range": [min(cands), max(cands)] if cands else None,
                      "max_ratio": max(ratio[f] for f in b), "reloc": dict(collections.Counter(m["result"] for m in mm)),
                      "overlap_max": max([m["overlap_ratio"] for m in mm if m.get("overlap_ratio") is not None], default=None),
                      "loops": loops, "dumps": [{k: dmp.get(k) for k in ("tag", "frame", "cand_kf", "gated", "raw", "gated_over_floor", "raw_over_floor")} for dmp in dumps]})
    res["bursts"] = table

    # ------------------------------------------------ 3 keyframe 0
    allc = jsonl(os.path.join(rd, "loop_checks.jsonl"))
    res["kf0"] = {"revisit_candidate_frames": [r["frame"] for r in chk if r.get("cand_kf") == 0 and r["pass"]],
                  "featquery_candidate_frames": [r["frame"] for r in allc if r.get("path") == "featquery" and r.get("cand_kf") == 0],
                  "mast3r": [m for m in mast if m["loop_kf"] == 0],
                  "loops": [{k: x.get(k) for k in ("cur_uid", "loop_uid", "pgo_ran", "loop_factor_in_graph", "pgo_err_before")} for x in att if int(x["loop_uid"]) == 0]}
    res["loop_attempts"] = [{k: x.get(k) for k in ("cur_uid", "loop_uid", "pgo_ran", "loop_factor_in_graph")} for x in att]

    # ------------------------------------------------ 4 ATE
    from v6_ate_split import ate_report, load_final

    def ate(run):
        out = {}
        for name in ("metrics_prerefine.csv", "metrics.csv"):
            pth = os.path.join(run, name)
            if os.path.exists(pth):
                r = list(csv.DictReader(open(pth)))[0]
                out[name] = {"ate_kf_cm": 100 * float(r["ate_rmse_keyframes_m"]), "ate_all_cm": 100 * float(r["ate_rmse_all_tracked_m"]), "psnr": float(r["mean_psnr"])}
        if os.path.exists(os.path.join(run, "final_state.pt")):
            _, est, gt = load_final(run)
            s = ate_report(est, gt, 1144)
            out["split_1144"] = {"all_cm": 100 * s["all"], "upto_cm": 100 * s["upto"], "after_cm": None if s["after"] is None else 100 * s["after"], "n_after": s["n_after"]}
        return out

    res["ate"] = {"E5": ate(rd)}
    if a.baseline:
        res["ate"]["baseline"] = ate(a.baseline.rstrip("/"))
    json.dump(res, open(os.path.join(a.out, "e5.json"), "w"), indent=1, default=str)

    print(f"\nchecked frames {res['n_checked_frames']}; dumps {res['dump_time']}")
    for thr, x in res["bursts_by_threshold"].items():
        print(f"  threshold {thr}: {x['n_frames']} frames, {x['n_bursts']} bursts ({x['n_bursts_without_loop']} without an accepted loop)")
    for t in table:
        dm = "; ".join(f"{x['tag']}: gated {x['gated'].get('gap_med_mm') and round(x['gated']['gap_med_mm'], 1)} mm/{x['gated'].get('pairs')} px, raw {x['raw'].get('gap_med_mm') and round(x['raw']['gap_med_mm'], 1)} mm/{x['raw'].get('pairs')} px"
                       for x in t["dumps"] if x.get("gated"))
        print(f"  burst {t['start']:5d}-{t['end']:5d} ({t['n_frames']:3d} fr) cand {t['cand_kf']} max ratio {t['max_ratio']:.3f} reloc {t['reloc']} overlap max {t['overlap_max']} loops {t['loops']}  | {dm}")
    print("  kf0:", json.dumps(res["kf0"])[:600])
    print("  ATE:", json.dumps(res["ate"]))


if __name__ == "__main__":
    main()
