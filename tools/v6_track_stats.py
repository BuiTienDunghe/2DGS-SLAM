"""plan v6 Q1 + Q4: where a frame spends its time, and what the tracking iterations are spent on.

usage: python tools/v6_track_stats.py --out DIR RUN_DIR [RUN_DIR ...]
Reads timing.jsonl, timing_backend.jsonl and track_log.jsonl of each run. Writes DIR/track_stats.json.

Pose distances are taken on the relative transform D = T_b T_a^-1 of two w2c poses: |D.t| is the displacement of the
camera centre and the rotation vector of D.R the relative rotation (tracking updates T <- Exp(tau) T, so D is the
composition of the applied increments; differencing the w2c translation vectors would mix the rotation in).

Per tracking call (purpose "track"):
  k_min   iterations needed just to travel from the initial to the final pose if every iteration moves at most one
          learning rate per axis = max_axis(|D.t| / lr_t, |rotvec| / lr_r), rounded up to the next 5-iteration mark
  k_star  earliest 5-iteration mark from which every later mark and the final pose stay within 1 mm and 0.05 deg
          of the final pose
  travel = k_min, refine = max(0, k_star - k_min), tail = iters_done - k_star
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

TOL_T, TOL_R = 1e-3, math.radians(0.05)


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def T44(v):
    T = np.eye(4)
    T[:3, :] = np.asarray(v, dtype=np.float64).reshape(3, 4)
    return T


def rel_vec(Ta, Tb):
    """6-vector (translation, rotvec) of D = T_b T_a^-1."""
    D = Tb @ np.linalg.inv(Ta)
    return np.concatenate([D[:3, 3], Rot.from_matrix(D[:3, :3]).as_rotvec()])


def pct(x, qs=(50, 90)):
    x = np.asarray([v for v in x if v is not None], dtype=np.float64)
    if x.size == 0:
        return {f"p{q}": None for q in qs} | {"mean": None, "n": 0}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"mean": float(x.mean()), "n": int(x.size)}


def timing_stats(rd):
    rows = [r for r in load_jsonl(os.path.join(rd, "timing.jsonl")) if not r.get("first_after_init")]
    if not rows:
        return None
    parts = ["t_stall", "t_data", "t_track", "t_pad", "t_loop", "t_kf", "t_eval", "t_other", "t_total"]
    sub = ["t_sync", "t_eval_stall", "t_log", "t_track_iters", "t_reloc_iters"]
    out = {"n_frames": len(rows), "sum_total_s": float(sum(r["t_total"] for r in rows))}
    for k in parts + sub:
        out[k] = pct([r.get(k) for r in rows]) | {"sum_s": float(sum((r.get(k) or 0.0) for r in rows))}
        out[k]["share_of_total"] = out[k]["sum_s"] / out["sum_total_s"]
    out["t_wait"] = {"sum_s": out["t_stall"]["sum_s"] - out["t_sync"]["sum_s"] - out["t_eval_stall"]["sum_s"] - out["t_log"]["sum_s"]}
    out["t_wait"]["share_of_total"] = out["t_wait"]["sum_s"] / out["sum_total_s"]
    out["t_track_per_iter_ms"] = pct([1e3 * r["t_track_iters"] / r["iters_done"] for r in rows if r.get("iters_done")])
    out["iters_done"] = pct([r.get("iters_done") for r in rows])
    kfr = [r for r in rows if r.get("kf_src")]
    # on a keyframe t_kf holds the insertion into the MASt3R retrieval database (loop closure on) and the request
    out["t_kf_on_keyframes"] = pct([r["t_kf"] for r in kfr]) | {"sum_s": float(sum(r["t_kf"] for r in kfr))}
    out["t_kf_on_keyframes"]["share_of_total"] = out["t_kf_on_keyframes"]["sum_s"] / out["sum_total_s"]
    out["t_loop_on_non_reloc"] = pct([r["t_loop"] for r in rows if not r.get("n_reloc")])
    out["n_kf_frames"] = sum(1 for r in rows if r.get("kf_src"))
    out["n_loop_frames"] = sum(1 for r in rows if r.get("kf_src") == "loop")
    out["n_reloc"] = int(sum(r.get("n_reloc", 0) for r in rows))
    # frames that are not keyframes: where the time goes when the backend is not asked for anything
    return out


def backend_stats(rd):
    rows = load_jsonl(os.path.join(rd, "timing_backend.jsonl"))
    if not rows:
        return None
    per_kf, acc_it, acc_t = [], 0, 0.0
    for r in rows:
        acc_it += r["idle_iters_since_last"]
        acc_t += r["t_idle_s"]
        if r["type"] in ("keyframe", "pgo"):  # both insert a keyframe
            per_kf.append((acc_it, acc_t))
            acc_it, acc_t = 0, 0.0
    it = [a for a, _ in per_kf[1:]]  # the first keyframe follows the init map, not an idle period
    return {
        "n_keyframe_msgs": sum(1 for r in rows if r["type"] == "keyframe"),
        "n_pgo_msgs": sum(1 for r in rows if r["type"] == "pgo"),
        "idle_iters_per_keyframe": pct(it),
        "idle_iters_total": int(sum(r["idle_iters_since_last"] for r in rows)),
        "idle_time_total_s": float(sum(r["t_idle_s"] for r in rows)),
        "t_keyframe_msg_s": pct([r["t_s"] for r in rows if r["type"] == "keyframe"]),
        "t_pgo_msg_s": pct([r["t_s"] for r in rows if r["type"] == "pgo"]),
        "ms_per_idle_iter": pct([1e3 * r["t_idle_s"] / r["idle_iters_since_last"] for r in rows if r["idle_iters_since_last"] > 0]),
    }


def track_stats(rd, lr_t, lr_r, kf_uids=None):
    rows = [r for r in load_jsonl(os.path.join(rd, "track_log.jsonl")) if r["purpose"] == "track"]
    if not rows:
        return None
    lr = np.array([lr_t] * 3 + [lr_r] * 3)
    per = []
    win_ratio = []  # per 5-iteration window: max-axis travel / (5 lr)
    for r in rows:
        Ti, Tf = T44(r["T_init"]), T44(r["T_final"])
        marks = list(r["ckpt_iters"])
        Ts = [T44(v) for v in r["T_ckpt"]]
        n = r["iters_done"]
        if not marks or marks[-1] != n:
            marks.append(n)
            Ts.append(Tf)
        d0 = rel_vec(Ti, Tf)
        k_min_raw = float(np.max(np.abs(d0) / lr))
        k_min = min(n, int(math.ceil(k_min_raw / 5.0 - 1e-9) * 5))
        ok = []
        for T in Ts:
            d = rel_vec(T, Tf)
            ok.append(np.linalg.norm(d[:3]) <= TOL_T and np.linalg.norm(d[3:]) <= TOL_R)
        k_star = n
        for i in range(len(marks) - 1, -1, -1):
            if ok[i]:
                k_star = marks[i]
            else:
                break
        prev_T, prev_k = Ti, 0
        last_move = None
        for k, T in zip(marks, Ts):
            d = rel_vec(prev_T, T)
            if k - prev_k == 5:
                win_ratio.append(float(np.max(np.abs(d) / (5 * lr))))
            last_move = (float(np.linalg.norm(d[:3])), float(np.linalg.norm(d[3:])), k - prev_k)
            prev_T, prev_k = T, k
        capped = (n >= r["n_iter"]) and not r["converged"]
        still = capped and last_move is not None and last_move[2] == 5 and (last_move[0] > TOL_T or last_move[1] > TOL_R)
        per.append({"uid": r["uid"], "n": n, "capped": capped, "k_min_raw": k_min_raw, "k_min": k_min, "k_star": k_star,
                    "refine_raw": k_star - k_min, "travel_mm": 1e3 * float(np.linalg.norm(d0[:3])),
                    "travel_deg": math.degrees(float(np.linalg.norm(d0[3:]))), "still_moving": bool(still),
                    "last5_mm": None if last_move is None else 1e3 * last_move[0],
                    "last5_deg": None if last_move is None else math.degrees(last_move[1]), "init_mode": r.get("init_mode")})

    def split(sel):
        tot = sum(p["n"] for p in sel)
        if tot == 0:
            return None
        travel = sum(min(p["k_min"], p["k_star"]) for p in sel)
        refine = sum(max(0, p["k_star"] - p["k_min"]) for p in sel)
        tail = sum(p["n"] - p["k_star"] for p in sel)
        return {"n_calls": len(sel), "iters_total": tot, "travel": travel / tot, "refine": refine / tot, "tail": tail / tot,
                "clamped_frac": float(np.mean([p["k_star"] < p["k_min"] for p in sel])),
                "k_min": pct([p["k_min_raw"] for p in sel], (50, 80, 95)), "k_star": pct([p["k_star"] for p in sel], (50, 80, 95)),
                "iters_done": pct([p["n"] for p in sel], (50, 80, 95))}

    capped = [p for p in per if p["capped"]]
    early = [p for p in per if not p["capped"]]
    still = [p for p in capped if p["still_moving"]]
    wr = np.asarray(win_ratio)
    kf = set(kf_uids or [])
    return {
        "n_calls": len(per), "init_modes": {m: sum(1 for p in per if p["init_mode"] == m) for m in sorted({str(p["init_mode"]) for p in per})},
        "split_all": split(per), "split_early_stopped": split(early), "split_capped": split(capped),
        "frac_capped": len(capped) / len(per),
        "travel_mm": pct([p["travel_mm"] for p in per], (50, 80, 95)), "travel_deg": pct([p["travel_deg"] for p in per], (50, 80, 95)),
        "conv_at": {str(k): float(np.mean([p["k_star"] <= k for p in per])) for k in (15, 30, 60, 90, 120)},
        "i_capped_still_moving_frac": (len(still) / len(capped)) if capped else None,
        "i_last5_mm_of_capped": pct([p["last5_mm"] for p in capped]), "i_last5_deg_of_capped": pct([p["last5_deg"] for p in capped]),
        "ii_window_over_5lr_frac": float(np.mean(wr > 1.0)) if wr.size else None,
        "ii_window_over_5p5lr_frac": float(np.mean(wr > 1.1)) if wr.size else None,
        "ii_window_ratio": pct(wr, (50, 95, 99)) | {"max": float(wr.max()) if wr.size else None},
        "iii_keyframes_from_capped_still_moving": sum(1 for p in still if p["uid"] in kf) if kf else None,
        "per_call": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {}
    for rd in a.runs:
        rd = rd.rstrip("/")
        import yaml
        cfg = yaml.safe_load(open(os.path.join(rd, "config_resolved.yaml")))
        lr = cfg["Training"]["lr"]
        kf = [r["uid"] for r in load_jsonl(os.path.join(rd, "timing.jsonl")) if r.get("kf_src")]
        name = os.path.basename(rd)
        out[name] = {"timing": timing_stats(rd), "backend": backend_stats(rd),
                     "tracking": track_stats(rd, lr["cam_trans_delta"], lr["cam_rot_delta"], kf)}
        t, b, k = out[name]["timing"], out[name]["backend"], out[name]["tracking"]
        print(f"== {name}")
        if t:
            print("  frames %d  t_total med %.0f ms  shares: " % (t["n_frames"], 1e3 * t["t_total"]["p50"]) +
                  "  ".join(f"{p[2:]} {100 * t[p]['share_of_total']:.1f}%" for p in ("t_data", "t_track", "t_loop", "t_kf", "t_eval", "t_stall", "t_pad", "t_other")))
            print("  stall split: wait %.1f%%  sync %.1f%%  eval %.1f%%  log %.2f%%;  ms/iter med %.1f;  iters med %.0f mean %.1f" % (
                100 * t["t_wait"]["share_of_total"], 100 * t["t_sync"]["share_of_total"], 100 * t["t_eval_stall"]["share_of_total"],
                100 * t["t_log"]["share_of_total"], t["t_track_per_iter_ms"]["p50"], t["iters_done"]["p50"], t["iters_done"]["mean"]))
        if b:
            print("  backend: idle iters per keyframe med %.0f mean %.1f (n %d), total idle iters %d in %.0f s, kf msg med %.2f s" % (
                b["idle_iters_per_keyframe"]["p50"] or 0, b["idle_iters_per_keyframe"]["mean"] or 0, b["idle_iters_per_keyframe"]["n"],
                b["idle_iters_total"], b["idle_time_total_s"], b["t_keyframe_msg_s"]["p50"] or 0))
        if k:
            for lab in ("split_all", "split_early_stopped", "split_capped"):
                s = k[lab]
                if s:
                    print("  %-20s calls %4d  travel %.1f%%  refine %.1f%%  tail %.1f%%  (clamped %.1f%%)  k_min med %.1f  k* med %.0f" % (
                        lab, s["n_calls"], 100 * s["travel"], 100 * s["refine"], 100 * s["tail"], 100 * s["clamped_frac"], s["k_min"]["p50"], s["k_star"]["p50"]))
            print("  converged-by: " + "  ".join(f"k<={kk}: {100 * v:.0f}%" for kk, v in k["conv_at"].items()))
            print("  (i) capped still moving: %s   (ii) windows > 5 lr: %.2f%%, > 5.5 lr: %.2f%%, ratio p50/p95/max %.2f/%.2f/%.2f   (iii) keyframes from such calls: %s" % (
                "n/a" if k["i_capped_still_moving_frac"] is None else f"{100 * k['i_capped_still_moving_frac']:.1f}%",
                100 * (k["ii_window_over_5lr_frac"] or 0), 100 * (k["ii_window_over_5p5lr_frac"] or 0),
                k["ii_window_ratio"]["p50"] or 0, k["ii_window_ratio"]["p95"] or 0, k["ii_window_ratio"]["max"] or 0, k["iii_keyframes_from_capped_still_moving"]))
    json.dump(out, open(os.path.join(a.out, "track_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
