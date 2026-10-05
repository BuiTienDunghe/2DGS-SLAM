"""Tables and pass rule of the P vs D experiment from pd.jsonl (tools/pd_experiment.py).

usage: python tools/pd_summary.py PD_JSONL [--md OUT.md]
Pass rule for X in {D, D+fine} (a rejected D counts with P0's numbers):
  1 keyframe ATE(X) <= ATE(P1) + max(5 %, 3 mm)      at >= 2/3 of the events
  2 gap on Pi*(X)   <= gap(P1)                         at >= 2/3 of the events that have Pi*
  3 time(X) <= 120 s at every event
  4 no torn event: largest E_reg edge residual <= 0.10 m and largest relative rotation of adjacent nodes <= 5 deg
    (the absolute node displacement is reported next to it: it equals the loop correction itself)
Diagnostic rows (not part of the rule): D@500 / D@1500 (iterates of the same solve), D+fineP (fine stage with the
prior of selected_v6), D+fine/raw (fine result refused by A1-A4), D/poseG (poses from the keyframe's Gaussians).
"""
import argparse
import json

RUN = {"20261002013837_rigid_s0": "rigid s0", "20261003123233_rigid_s1": "rigid s1", "20261003105219_deform5_s0": "v5 s0", "20261003114156_deform5_s1": "v5 s1"}


def f(x, nd=1, scale=1.0):
    return "—" if x is None else f"{scale * x:.{nd}f}".replace(".", ",")


def rule(ev, name, base=None):
    """Pass-rule counters of branch `name`; `base` = the coarse branch whose reject / stretch applies (D for D+fine)."""
    base = base or name
    s = {"ate_ok": 0, "ate_n": 0, "gap_ok": 0, "gap_n": 0, "t_max": 0.0, "t_over": 0, "torn": 0, "torn_abs": 0, "reject": 0, "n": 0,
         "gap_le_p0": 0, "ate_fail": [], "gap_fail": [], "t_fail": []}
    for r in ev:
        b = r["branches"]
        if base not in b:
            continue
        s["n"] += 1
        tag = f"{RUN.get(r['run'], r['run'])} {r['cur']}↔{r['loop']}"
        P0, P1, x, c = b["P0"], b["P1"], b.get(name, {}), b[base]
        rej = bool(c.get("reject"))
        src = P0 if rej else (x if x.get("ate_kf_m") is not None else c)  # a rejected D counts with P0; a refused fine stage with D
        s["reject"] += int(rej)
        if P1.get("ate_kf_m") is not None and src.get("ate_kf_m") is not None:
            ok = src["ate_kf_m"] <= P1["ate_kf_m"] + max(0.05 * P1["ate_kf_m"], 0.003)
            s["ate_n"] += 1
            s["ate_ok"] += int(ok)
            if not ok:
                s["ate_fail"].append(tag)
        if P1.get("gap_mm") is not None and src.get("gap_mm") is not None:
            ok = src["gap_mm"] <= P1["gap_mm"]
            s["gap_n"] += 1
            s["gap_ok"] += int(ok)
            s["gap_le_p0"] += int(src["gap_mm"] <= P0["gap_mm"] * 1.05)
            if not ok:
                s["gap_fail"].append(tag)
        t = x.get("t_total_s") if x.get("t_total_s") is not None else c.get("t_total_s")
        if t is not None:
            s["t_max"] = max(s["t_max"], t)
            s["t_over"] += int(t > 120.0)
            if t > 120.0:
                s["t_fail"].append(tag)
        if c.get("edge_res_max_m") is not None and not rej:
            s["torn"] += int(c["edge_res_max_m"] > 0.10 or c["edge_rot_max_deg"] > 5.0)
            s["torn_abs"] += int(c["node_disp_abs_max_m"] > 0.10 or c["node_rot_abs_max_deg"] > 5.0)
    s["c1"] = s["ate_n"] > 0 and s["ate_ok"] * 3 >= 2 * s["ate_n"]
    s["c2"] = s["gap_n"] > 0 and s["gap_ok"] * 3 >= 2 * s["gap_n"]
    s["c3"], s["c4"] = s["t_over"] == 0, s["torn"] == 0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path) if l.strip()]
    ev = [r for r in rows if "branches" in r]
    skipped = [r for r in rows if "branches" not in r]
    out = []
    names = ["P0", "P1", "D", "D+fine"]
    out.append("| Run | Sự kiện | H: dịch khung hiện tại / xoay | ATE keyframe (cm) P0 · P1 · D · D+fine | Khe Π* (mm) trước · P0 · P1 · D · D+fine | D lệch P0 sau L: tịnh tiến tv/max (mm) · xoay tv/max (°) "
               "| Giãn D: cạnh max (mm / °) · node max (mm) | Giải D: s (vòng) | Tổng thời gian (s) P1 · D · D+fine | Ghi chú |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in ev:
        b = r["branches"]
        D = b.get("D", {})
        rej = bool(D.get("reject"))
        notes = []
        if not r["H_check"]["ok"]:
            notes.append("H không giảm khe")
        if b["P1"].get("reason"):
            notes.append(f"P1 lùi về rigid ({b['P1']['reason']})")
        if rej:
            notes.append(f"**D reject** (kiểm 6: {D.get('check_pairs')} cặp, khe {f(D.get('check_gap_mm'))} mm)" if D.get("check_pairs") is not None else f"**D reject** ({D['reject']})")
        if b.get("D+fine", {}).get("reason"):
            notes.append(f"fine bị loại ({b['D+fine']['reason']}) → D+fine = D")
        dv = D.get("dev_vs_P0_afterL") or {}
        star = "*" if rej else ""
        cells = [RUN.get(r["run"], r["run"]), f"{r['cur']}↔{r['loop']}", f"{f(r['H']['cur_shift_mm'], 0)} mm / {f(r['H']['rot_deg'], 2)}°",
                 " · ".join(f(b.get(n, {}).get("ate_kf_m"), 2, 100) + (star if n in ("D", "D+fine") else "") for n in names),
                 " · ".join([f(b["pre"].get("gap_own_mm"))] + [f(b.get(n, {}).get("gap_mm")) + (star if n in ("D", "D+fine") else "") for n in names]),
                 f"{f(dv.get('t_med_mm'))}/{f(dv.get('t_max_mm'))} · {f(dv.get('r_med_deg'), 2)}/{f(dv.get('r_max_deg'), 2)}",
                 f"{f(D.get('edge_res_max_m'), 1, 1e3)} / {f(D.get('edge_rot_max_deg'), 2)} · {f(D.get('node_disp_abs_max_m'), 0, 1e3)}",
                 f"{f(D.get('solve_s'), 0)} ({D.get('lbfgs_iters', '—')}" + ("" if (D.get("solver_grad") or {}).get("converged", True) else ", chưa hội tụ") + ")",
                 " · ".join(f(b.get(n, {}).get("t_total_s"), 0) for n in names[1:]), "; ".join(notes)]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("`*` = D bị reject ở bước kiểm (6): số trong ô là kết quả D tự đo, thống kê dùng P0. Sự kiện bỏ: "
               + ("; ".join(f"{RUN.get(r['run'], r['run'])} {r['cur']}↔{r['loop']}" for r in skipped) or "không") + " (factor loop bị backend gỡ khỏi đồ thị, không có phép đo loop).")
    out.append("")
    out.append("| Nhánh | (1) ATE ≤ P1 + max(5 %, 3 mm) | (2) khe Π* ≤ P1 | (3) thời gian ≤ 120 s | (4) không rách (cạnh ≤ 0,10 m và ≤ 5°) | D reject | Đạt |")
    out.append("|---|---|---|---|---|---|---|")
    diag = [("D", None), ("D+fine", "D"), ("D@500", "D@500"), ("D@1500", "D@1500"), ("D+fineP", "D"), ("D-active", None)]
    stats = {}
    for n, base in diag:
        s = rule(ev, n, base)
        if s["n"] == 0:
            continue
        stats[n] = s
        label = n + ("" if n in ("D", "D+fine") else " (chẩn đoán)")
        out.append(f"| {label} | {s['ate_ok']}/{s['ate_n']} {'✓' if s['c1'] else '✗'} | {s['gap_ok']}/{s['gap_n']} {'✓' if s['c2'] else '✗'} (≤ 1,05 × P0: {s['gap_le_p0']}/{s['gap_n']}) "
                   f"| max {f(s['t_max'], 0)} s, vượt {s['t_over']}/{s['n']} {'✓' if s['c3'] else '✗'} | rách {s['torn']}/{s['n'] - s['reject']} {'✓' if s['c4'] else '✗'} (node tuyệt đối > 0,10 m hoặc 5°: {s['torn_abs']}) "
                   f"| {s['reject']}/{s['n']} | {'ĐẠT' if s['c1'] and s['c2'] and s['c3'] and s['c4'] else 'KHÔNG'} |")
    # ---- diagnostics table
    out.append("")
    out.append("| Run | Sự kiện | D@500: ATE (cm) · khe (mm) · cách nghiệm cuối (mm) · tổng (s) | D@1500: như trái | D tỉ lệ gradient cuối | fine không prior: vòng · node max (mm / °) · khe thô (mm) | D+fineP (có prior): chấp nhận · ATE (cm) · khe (mm) · tổng (s) | D/poseG ATE (cm) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in ev:
        b = r["branches"]
        D = b.get("D", {})

        def snap(n):
            x = b.get(n)
            if not x:
                return "—"
            return f"{f(x.get('ate_kf_m'), 2, 100)} · {f(x.get('gap_mm'))} · {f(x.get('dtheta_to_final_max_m'), 1, 1e3)} · {f(x.get('t_total_s'), 0)}"

        F, FP, raw, rawP = b.get("D+fine", {}), b.get("D+fineP", {}), b.get("D+fine/raw", {}), b.get("D+fineP/raw", {})
        gF = raw.get("gap_mm") if raw else (F.get("gap_mm") if F.get("accepted") else None)
        fp = "—"
        if FP:
            x = FP if FP.get("accepted") else rawP
            fp = f"{'có' if FP.get('accepted') else 'không (' + str(FP.get('reason')) + ')'} · {f(x.get('ate_kf_m'), 2, 100)} · {f(x.get('gap_mm'))} · {f(FP.get('t_total_s'), 0)}"
        out.append("| " + " | ".join([RUN.get(r["run"], r["run"]), f"{r['cur']}↔{r['loop']}", snap("D@500"), snap("D@1500"),
                                       ("—" if not D.get("solver_grad") else f"{D['solver_grad'].get('grad_ratio', 0):.1e}".replace(".", ",")),
                                       ("—" if not F else f"{F.get('iters')} · {f(F.get('max_node_disp_m'), 0, 1e3)} / {f(F.get('max_node_rot_deg'), 2)} · {f(gF)}"),
                                       fp, f(b.get("D/poseG", {}).get("ate_kf_m"), 2, 100)]) + " |")
    if any("D-active" in r["branches"] for r in ev):
        out.append("")
        out.append("| Run | Sự kiện | D-active: ATE (cm) · khe (mm) · cạnh max (mm / °) · lệch P0 tv/max (mm) · tổng (s) |")
        out.append("|---|---|---|")
        for r in ev:
            x = r["branches"].get("D-active")
            if x:
                dv = x.get("dev_vs_P0_afterL") or {}
                out.append(f"| {RUN.get(r['run'], r['run'])} | {r['cur']}↔{r['loop']} | {f(x.get('ate_kf_m'), 2, 100)} · {f(x.get('gap_mm'))} · {f(x.get('edge_res_max_m'), 1, 1e3)} / {f(x.get('edge_rot_max_deg'), 2)} · "
                           f"{f(dv.get('t_med_mm'))}/{f(dv.get('t_max_mm'))} · {f(x.get('t_total_s'), 0)}" + (" · reject" if x.get("reject") else "") + " |")
    txt = "\n".join(out)
    print(txt)
    for n, s in stats.items():
        print(f"[{n}] ATE fail: {s['ate_fail']}  gap fail: {s['gap_fail']}  time fail: {len(s['t_fail'])}")
    if a.md:
        open(a.md, "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
