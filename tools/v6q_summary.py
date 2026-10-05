"""plan v6 quick: Markdown tables of E5 / E3 / E1 / E4 from results_exp/reports/v6q.

usage: python tools/v6q_summary.py [REPORT_DIR] [--md OUT.md] [--json OUT.json]
"""
import argparse
import json
import os
import statistics as S

RUN = {"20261002013837_rigid_s0": "rigid s0", "20261003123233_rigid_s1": "rigid s1", "20261003105219_deform5_s0": "v5 s0",
       "20261003114156_deform5_s1": "v5 s1", "20261004075609_deform6_s0": "v6 s0", "20261004085056_deform6_s1": "v6 s1"}
RES = {"reject_overlap": "loại ở chồng lấn", "reject_conf": "loại ở độ tin cậy", "reject_tracking": "loại ở tracking", "accepted": "nhận"}


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def name(run):
    return RUN.get(run, "E5" if run.endswith("_E5") else run)


def rows_of(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def e5(O, out, info):
    """Burst table of the E5 run (tools/e5_revisit_gap.py + tools/e5_gt_drift.py)."""
    p = os.path.join(O, "e5", "e5.json")
    if not os.path.exists(p):
        return
    r = json.load(open(p))
    pg = os.path.join(O, "e5", "e5_gt_drift.json")
    gt = json.load(open(pg)) if os.path.exists(pg) else {}
    out.append("| Đợt (frame) | Số frame qua cổng | Keyframe ứng viên | observed_ratio lớn nhất | MASt3R: kết quả; chồng lấn lớn nhất | Dump (frame) | Hai lớp cùng thấy (% lưới) "
               "| Khe có lọc 50 mm: trung vị (mm), số cặp | Khe bỏ lọc 50 mm: trung vị / p90 (mm), số cặp | Sai số pose tương đối theo GT (mm / °) | Loop được nhận |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for b in r["bursts"]:
        rel = ", ".join(f"{RES.get(k, k)} {v}" for k, v in b["reloc"].items())
        loops = "; ".join(f"{x[0]}↔{x[1]}" for x in b["loops"]) or "không"
        for i, d in enumerate(b["dumps"] or [{}]):
            g, w, q = d.get("gated") or {}, d.get("raw") or {}, gt.get(d.get("tag"), {})
            head = [f"{b['start']}–{b['end']}", str(b["n_frames"]), ", ".join(str(k) for k in b["cand_kf"]), f(b["max_ratio"], 2), f"{rel}; {f(b['overlap_max'], 2)}"] if i == 0 else [""] * 5
            cells = head + [str(d.get("frame", "—")), f(100 * g["both_layers_frac"], 0) if g else "—", f"{f(g.get('gap_med_mm'))}, {g.get('pairs', '—')}",
                            f"{f(w.get('gap_med_mm'))} / {f(w.get('gap_p90_mm'))}, {w.get('pairs', '—')}",
                            f"{f(q.get('rel_err_t_mm'), 0)} / {f(q.get('rel_err_rot_deg'), 1)}" if q else "—", loops if i == 0 else ""]
            out.append("| " + " | ".join(cells) + " |")
    info["e5"] = {"ate": r["ate"], "bursts_by_threshold": {k: [v["n_frames"], v["n_bursts"], v["n_bursts_without_loop"]] for k, v in r["bursts_by_threshold"].items()},
                  "ratio_quantiles": r["observed_ratio"]["quantiles"], "ratio_hist": r["observed_ratio"]["hist"], "dump_time": r["dump_time"], "n_checked": r["n_checked_frames"],
                  "kf0_frames": r["kf0"]["revisit_candidate_frames"], "kf0_featquery": r["kf0"]["featquery_candidate_frames"],
                  "kf0_overlap": [m["overlap_ratio"] for m in r["kf0"]["mast3r"]], "kf0_loops": r["kf0"]["loops"], "attempts": r["loop_attempts"]}


def e3(O, out, info):
    rows = rows_of(os.path.join(O, "e3", "e3.jsonl")) + rows_of(os.path.join(O, "e3_e5", "e3.jsonl"))
    it3 = {(r["run"], r["state"]): r for r in rows_of(os.path.join(O, "e3_it3", "e3.jsonl"))}
    out.append("| Run | Trạng thái | Sàn, chia theo keyframe: trung vị / p90 (mm) | Sàn, chia theo băm: trung vị / p90 (mm) | Độ lệch theo keyframe: trung vị / p90 / max (mm) | Sàn sau bù (keyframe / băm) | Sàn sau bù 3 lượt (keyframe) |")
    out.append("|---|---|---|---|---|---|---|")
    ev_r, ev_h, fin_r, fin_h, gain, pre, post = [], [], [], [], [], [], []
    for r in rows:
        st = r["state"].replace("event ", "sau ").replace("<->", "↔").replace("final", "bản đồ cuối").replace("revisit ", "đợt revisit ")
        if "floor" not in r or len(r["views"]) < 10:
            out.append(f"| {name(r['run'])} | {st} | — | — | — | — | — (vùng 300–600 chưa dựng xong: {len(r['views'])} khung nhìn) |")
            continue
        fl, fc, of = r["floor"], r["floor_comp"], r["offsets"]
        r3 = it3.get((r["run"], r["state"]))
        out.append(f"| {name(r['run'])} | {st} | {f(fl['rank']['median_mm'])} / {f(fl['rank']['p90_mm'])} | {f(fl['hash']['median_mm'])} / {f(fl['hash']['p90_mm'])} "
                   f"| {f(of['median_abs_mm'], 2)} / {f(of['p90_abs_mm'], 2)} / {f(of['max_abs_mm'], 2)} | {f(fc['rank']['median_mm'])} / {f(fc['hash']['median_mm'])} | {f(r3['floor_comp']['rank']['median_mm']) if r3 else '—'} |")
        (fin_r if r["state"] == "final" else ev_r).append(fl["rank"]["median_mm"])
        (fin_h if r["state"] == "final" else ev_h).append(fl["hash"]["median_mm"])
        gain.append(fl["rank"]["median_mm"] - fc["rank"]["median_mm"])
        # before the first large loop (short loop 70x<->63x, revisit dumps up to frame 970) / after a large loop event
        frame = int(r["state"].split()[1].split("<->")[0]) if r["state"].startswith("event") else (int(r["state"].split("f")[-1]) if r["state"].startswith("revisit") else None)
        if frame is not None:
            is_pre = (r["state"].startswith("event") and frame < 800) or (r["state"].startswith("revisit") and frame <= 970)
            (pre if is_pre else post).append(fl["rank"]["median_mm"])
            info.setdefault("e3_hash_groups", {"pre": [], "post": []})["pre" if is_pre else "post"].append(fl["hash"]["median_mm"])
            info.setdefault("e3_offsets", []).append(of["median_abs_mm"])
            info.setdefault("e3_offsets_max", []).append(of["max_abs_mm"])
    if ev_r:
        info["e3"] = {"event_rank": [min(ev_r), S.median(ev_r), max(ev_r)], "event_hash": [min(ev_h), S.median(ev_h), max(ev_h)],
                      "final_rank": [min(fin_r), max(fin_r)] if fin_r else None, "final_hash": [min(fin_h), max(fin_h)] if fin_h else None,
                      "comp_gain": [min(gain), S.median(gain), max(gain)], "n_states": len(ev_r) + len(fin_r), "n_event_states": len(ev_r),
                      "before_big_loop_rank": [min(pre), S.median(pre), max(pre), len(pre)] if pre else None,
                      "after_big_loop_rank": [min(post), S.median(post), max(post), len(post)] if post else None}


def e1(O, out, info):
    rows = [r for r in rows_of(os.path.join(O, "e1", "e1.jsonl")) + rows_of(os.path.join(O, "e1_e5", "e1.jsonl")) if "regions" in r]
    out.append("| Run | Vùng loop: PSNR trước → sau (Δ dB) | Vùng một lượt: PSNR trước → sau (Δ dB) | Δ_loop − Δ_một_lượt (dB): gộp pixel / trung bình theo keyframe / trung vị theo keyframe "
               "| Depth L1 vùng loop (mm) | Depth L1 vùng một lượt (mm) | Mức giảm L1: loop / một lượt | Theo quy tắc của plan (số gộp pixel) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        L, Sg, a = r["regions"]["loop"], r["regions"]["single"], r["alt"]
        out.append(f"| {name(r['run'])} | {f(L['psnr_pre'], 2)} → {f(L['psnr_post'], 2)} ({f(L['d_psnr'], 2)}) | {f(Sg['psnr_pre'], 2)} → {f(Sg['psnr_post'], 2)} ({f(Sg['d_psnr'], 2)}) "
                   f"| {f(r['d_psnr_loop_minus_single'], 2)} / {f(a['kfmean_loop_minus_single'], 2)} / {f(a['kfmedian_loop_minus_single'], 2)} "
                   f"| {f(L['l1_pre_mm'])} → {f(L['l1_post_mm'])} (−{f(100 * L['l1_drop_rel'], 0)} %) | {f(Sg['l1_pre_mm'])} → {f(Sg['l1_post_mm'])} (−{f(100 * Sg['l1_drop_rel'], 0)} %) "
                   f"| {f(r['l1_drop_ratio_loop_over_single'], 2)} | {'bù loop' if r['rule']['verdict'] == 'loop' else 'bù tối ưu online'} |")
    info["e1"] = {"pooled": [r["d_psnr_loop_minus_single"] for r in rows], "kfmean": [r["alt"]["kfmean_loop_minus_single"] for r in rows],
                  "kfmedian": [r["alt"]["kfmedian_loop_minus_single"] for r in rows], "l1_ratio": [r["l1_drop_ratio_loop_over_single"] for r in rows],
                  "gated_pooled": [r["alt"]["pooled_loop_minus_gated"] for r in rows], "gated_kfmean": [r["alt"]["kfmean_loop_minus_gated"] for r in rows],
                  "gated_kfmedian": [r["alt"]["kfmedian_loop_minus_gated"] for r in rows], "gated_l1_ratio": [r["alt"]["l1_drop_ratio_loop_over_gated"] for r in rows],
                  "loop_px": [r["regions"]["loop"]["n_pix"] for r in rows], "loop_kf": [r["regions"]["loop"]["n_kf"] for r in rows],
                  "l1_rel_loop": [r["regions"]["loop"]["l1_drop_rel"] for r in rows], "l1_rel_single": [r["regions"]["single"]["l1_drop_rel"] for r in rows],
                  "pose_diff": [r["pose_diff_max_m"] for r in rows]}


def e4(O, out, info):
    rows = [r for r in rows_of(os.path.join(O, "e4", "e4.jsonl")) if "variants" in r]
    out.append("| Run | Sự kiện | Gaussian vùng Π* (cũ / mới) | Biến thể | Cặp gộp | Giảm trong vùng / toàn bản đồ (%) | ΔPSNR: J_eval / mọi keyframe (dB) | ΔDepth L1: J_eval / mọi keyframe (mm) "
               "| Pixel Π* còn hai lớp: trước → sau | Khe trên pixel còn hai lớp: trước → sau (mm) | Khả thi |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    allv, fun = [], []
    for r in rows:
        for k, v in r["variants"].items():
            allv.append(v)
            tau, mode = k.split("_")
            out.append(f"| {name(r['run'])} | {r['event'].replace('<->', '↔')} | {r['n_region_old']} / {r['n_region_new']} | τ_d {tau[3:]} mm, {mode} | {v['n_pairs']} "
                       f"| {f(100 * v['removed_frac_region'])} / {f(100 * v['removed_frac_map'], 2)} | {f(v['d_psnr_eval'], 3)} / {f(v['d_psnr_all'], 3)} | {f(v['d_l1_eval_mm'], 2)} / {f(v['d_l1_all_mm'], 2)} "
                       f"| {f(r['before']['two_layer_frac'], 2)} → {f(v['after']['two_layer_frac'], 2)} | {f(v['gap_before_same_px_mm'])} → {f(v['after']['gap_mm'])} | {'có' if v['feasible'] else 'không'} |")
            if mode == "A" and v.get("funnel"):
                x = v["funnel"]
                fun.append({"event": r["event"], "tau": tau, **{kk: 100 * x[kk] / x["new"] for kk in ("normal_dist", "normal_angle", "tangential", "one_per_old")},
                            "nn_mm": x["nn_dist_med_mm"], "dn_mm": x["normal_dist_med_mm"], "dt_mm": x["tangential_med_mm"]})
    if allv:
        def rng(k, s=1.0):
            return [s * min(v[k] for v in allv), s * max(v[k] for v in allv)]

        info["e4"] = {"removed_region_pct": rng("removed_frac_region", 100), "removed_map_pct": rng("removed_frac_map", 100), "d_psnr_eval": rng("d_psnr_eval"),
                      "d_psnr_all": rng("d_psnr_all"), "d_l1_eval": rng("d_l1_eval_mm"), "d_l1_all": rng("d_l1_all_mm"),
                      "feasible": [sum(v["feasible"] for v in allv), len(allv)], "funnel": fun}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results_exp/reports/v6q")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out, info = [], {}
    for title, fn in (("E5", e5), ("E3", e3), ("E1", e1), ("E4", e4)):
        out.append(f"\n### {title}\n")
        fn(a.dir, out, info)
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(txt + "\n")
    if a.json:
        json.dump(info, open(a.json, "w"), indent=1, default=lambda x: round(x, 4))


if __name__ == "__main__":
    main()
