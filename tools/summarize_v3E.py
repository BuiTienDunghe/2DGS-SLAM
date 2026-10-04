"""handoffPlan_v3_solver step E: end-of-run summary of the method runs (#3, #5) next to the baseline (#1).

usage: python tools/summarize_v3E.py --out results_exp/reports/v3/E BASELINE_RUN RUN_3 RUN_5
Per run: ATE (keyframes / all frames), PSNR before / after refine, SSIM, LPIPS, loop attempts and events,
per event: accepted / fallback reason, pairs, nodes with pairs, solver iterations, time, node motion;
SLAM time, FPS, VRAM, number of Gaussians. Output: OUT/E_summary.json + a console table.
"""
import argparse
import csv
import json
import os


def read_csv(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rd):
    m, mp = read_csv(os.path.join(rd, "metrics.csv")), read_csv(os.path.join(rd, "metrics_prerefine.csv"))
    res = {}
    if os.path.exists(os.path.join(rd, "resources.json")):
        res = json.load(open(os.path.join(rd, "resources.json")))
    ev = read_jsonl(os.path.join(rd, "loop_events.jsonl"))
    att = read_jsonl(os.path.join(rd, "loop_attempts.jsonl"))
    f = lambda d, k: None if d is None or d.get(k) in (None, "") else float(d[k])
    out = {"run": os.path.basename(rd),
           "ate_kf_cm": None if m is None else 100 * f(m, "ate_rmse_keyframes_m"),
           "ate_all_cm": None if m is None else 100 * f(m, "ate_rmse_all_tracked_m"),
           "psnr_pre": f(mp, "mean_psnr"), "psnr_post": f(m, "mean_psnr"),
           "ssim_post": f(m, "mean_ssim"), "lpips_post": f(m, "mean_lpips"),
           "ate_kf_pre_cm": None if mp is None else 100 * f(mp, "ate_rmse_keyframes_m"),
           "n_attempts": len(att), "n_events": len(ev),
           "n_accepted": sum(1 for e in ev if e.get("accepted") is True),
           "n_fallback": sum(1 for e in ev if e.get("accepted") is False),
           "slam_time_s": res.get("slam_time_s"), "fps_hz": res.get("fps_hz"), "total_time_s": res.get("total_time_s"),
           "vram_fe_gb": res.get("frontend_peak_vram_gb"),
           "vram_be_gb": (res.get("backend") or {}).get("backend_peak_vram_gb"),
           "n_gaussians": (res.get("backend") or {}).get("n_gaussians"), "n_keyframes": res.get("n_keyframes"),
           "events": []}
    for e in ev:
        ts = e.get("t_stage_s") or {}
        ac = e.get("accept_checks") or {}
        out["events"].append({
            "event_id": e.get("event_id"), "cur": e.get("cur_uid"), "loop": e.get("loop_uid"), "mode": e.get("mode"),
            "accepted": e.get("accepted"), "reason": e.get("fallback_reason"),
            "pgo_err_before": e.get("pgo_err_before"), "pgo_err_after": e.get("pgo_err_after"),
            "pairs": e.get("corr_capped"), "nodes_with_pairs": e.get("n_nodes_with_pairs"), "n_nodes": e.get("n_nodes"),
            "lbfgs_iters": e.get("lbfgs_iters"), "grad_ratio": (e.get("solver_grad") or {}).get("grad_ratio"),
            "edl_opt_mm": (e.get("edl_opt_init_mm"), e.get("edl_opt_final_mm")), "A2_ratio": ac.get("A2_ratio"),
            "max_node_disp_mm": None if e.get("max_node_disp_m") is None else 1e3 * e["max_node_disp_m"],
            "max_node_rot_deg": e.get("max_node_rot_deg"), "wrong_pairs": e.get("frac_res_gt30mm"),
            "t_event_s": e.get("t_event_s", ts.get("total")), "t_total_s": ts.get("total")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = [summarize(rd) for rd in a.runs]
    with open(os.path.join(a.out, "E_summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    keys = ("ate_kf_cm", "ate_all_cm", "psnr_pre", "psnr_post", "n_attempts", "n_events", "n_accepted", "n_fallback",
            "slam_time_s", "fps_hz", "total_time_s", "vram_fe_gb", "vram_be_gb", "n_gaussians")
    for s in S:
        print(s["run"], {k: (round(s[k], 3) if isinstance(s[k], float) else s[k]) for k in keys})
        for e in s["events"]:
            print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})


if __name__ == "__main__":
    main()
