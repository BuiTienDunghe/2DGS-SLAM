"""PLAN MONOGS-PW2 §10: docs/monogs/REPORT_MONOGS_PW2_R2.md (+ R2_run1.md with --run1) from metrics_pw2.json.
usage (repo root, env monogs): python tools/monogs/pw2/make_report_pw2.py [--run1] [--no-eval]
Free text comes from docs/monogs/pw2/notes_*.md when present.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pw2 import SPEC  # noqa: E402

D = "docs/monogs/pw2"


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "có" if x else "không"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{nd}f}".replace(".", ",")


def g(m, *k):
    for x in k:
        if not isinstance(m, dict):
            return None
        m = m.get(x)
    return m


def note(n):
    p = os.path.join(D, n)
    return open(p).read() if os.path.exists(p) else ""


def tag(a, b):
    if a is None or b is None:
        return None
    return "tốt hơn" if a <= 0.7 * b else ("tệ hơn" if a >= 1.3 * b else "trong nhiễu")


def ok_run(m):
    return m is not None and "missing" not in m


def seq_table(M, s):
    L = [f"### {SPEC[s]['title']}\n",
         "| Method | ATE keyframe | ATE mọi frame (hiệu chỉnh*) | RPE Δ1 hiệu chỉnh* / thô | Chưa căn cuối / lớn nhất | Keyframe (mono/T1/T2) | \"Đủ\" % | % cặp bị kernel loại | % cặp đúng nhãn bị loại | Ứng viên tạo / xóa / kích hoạt | Anchor / mặt thật | Bản trùng | % ghép nhầm | Đổi ID đồng thời / lật | PGO áp | ms/frame | s tổng |",
         "|---|" + "---|" * 17]
    for meth in SPEC[s]["runs"]:
        m = M[s].get(meth)
        if not ok_run(m):
            L.append(f"| {meth} | không có run |" + " |" * 16)
            continue
        w, gt = m.get("pw", {}), m.get("gt", {}) or {}
        kr = w.get("kf_reasons") or {}
        kfs = f"{m['n_kf']}" + (f" ({kr.get('mono', 0)}/{kr.get('T1', 0)}/{kr.get('T2', 0)})" if kr else "")
        corr = m.get("ate_all_corr_cm")
        rc = m.get("rpe_d1_corr_cm")
        L.append(f"| {meth} | {f(m['ate_kf_cm'])} | {f(corr if corr is not None else m['ate_all_cm'])}{'' if corr is not None else '*'} | "
                 f"{f(rc if rc is not None else m['rpe_d1_cm'], 3)}{'' if rc is not None else '*'} / {f(m['rpe_d1_cm'], 3)} | "
                 f"{f(m['unaligned_end_cm'], 1)} / {f(m['unaligned_max_cm'], 1)} | {kfs} | {f(w.get('pct_enough'), 1)} | {f(w.get('pct_rejected'), 1)} | "
                 f"{f(gt.get('pct_correct_rejected'), 1)} | {w.get('anchors_created', '—')} / {w.get('anchors_deleted', '—')} / {w.get('anchors_activated', '—')} | "
                 f"{f(gt.get('anchors_per_true_plane'))} | {gt.get('duplicates', '—')} | {f(gt.get('pct_mismatch'), 1)} | "
                 f"{gt.get('id_simultaneous', '—')} / {gt.get('id_flip', '—')} | {g(w, 'pgo', 'applied') if g(w, 'pgo', 'n') else '—'} | "
                 f"{f(m['ms_track'], 0)} | {f(m['total_s'], 0)} |")
    L.append("\n\\* run không có `traj_all_corr.txt` (run cũ, không PGO): dùng quỹ đạo thô.\n")
    return L


def pgo_rows(M, s, meth):
    m = M[s].get(meth)
    if not ok_run(m) or not m.get("pgo_table"):
        return []
    L = [f"**{s} / {meth}** — độ chính xác closure (closure được nhận, có nhãn): {f(m.get('closure_precision'), 1)} % ({m.get('closure_n_accepted_labeled')})\n",
         "| Frame k | Trạng thái | Tier 1 / tier 2 | Closure (anchor, tier, đúng nhãn, bị bỏ, w IRLS) | ATE tập keyframe trước → sau (cm) | Dịch lớn nhất (cm) | s |",
         "|---|---|---|---|---|---|---|"]
    for r in m["pgo_table"]:
        cl = "; ".join(f"a{c['anchor']} t{c['tier']} {'đúng' if c['correct'] else ('sai' if c['correct'] is False else '?')}"
                       f"{' BỎ' if c['dropped'] else ''} w={f(c.get('irls_w'))}" for c in r["closures"]) or "—"
        L.append(f"| {r['frame']} | {r['status']} | {r['n_t1']} / {r['n_t2']} | {cl} | {f(r['ate_before_cm'])} → {f(r['ate_after_cm'])} | "
                 f"{f(r['max_shift_cm'])} | {f(r['solve_s'], 1)} |")
    L.append("")
    return L


def decisions(M):
    out, R = [], {}
    seqs = [s for s in M if ok_run(M[s].get("pw2"))]
    # D1
    d1 = []
    for s in seqs:
        p, e = M[s]["pw2"], M[s].get("e1c")
        pe, ee = g(p, "pw", "pct_enough"), g(e, "pw", "pct_enough")
        cr = g(p, "gt", "pct_correct_rejected")
        mp, me = g(p, "gt", "pct_mismatch"), g(e, "gt", "pct_mismatch")
        dp, de = g(p, "gt", "duplicates"), g(e, "gt", "duplicates")
        ok = None not in (pe, ee, cr, mp, me, dp, de) and pe >= 0.5 * ee and cr <= 15 and mp <= me and dp <= de
        d1.append(ok)
        out.append(f"  - {s}: \"đủ\" {f(pe, 1)} / E1c {f(ee, 1)} %; cặp đúng nhãn bị loại {f(cr, 1)} %; ghép nhầm {f(mp, 1)} / E1c {f(me, 1)} %; "
                   f"bản trùng {dp} / E1c {de} → {'đạt' if ok else 'không đạt'}.")
    R["D1"] = bool(d1) and all(d1)
    out.insert(0, f"- **D1 (plane world sống lại?): {'ĐẠT' if R['D1'] else 'KHÔNG ĐẠT'}** (cần trên mọi chuỗi đã chạy pw2)")
    # D2
    tags, all_ok, lines = [], True, []
    for s in seqs:
        p, e = M[s]["pw2"], M[s].get("e1c")
        t = tag(p["ate_kf_cm"], g(e, "ate_kf_cm"))
        pa = p.get("ate_all_corr_cm") or p["ate_all_cm"]
        ea = g(e, "ate_all_corr_cm") or g(e, "ate_all_cm")
        ta = tag(pa, ea)
        tags.append(t)
        all_ok &= ta != "tệ hơn"
        lines.append(f"  - {s}: ATE keyframe {f(p['ate_kf_cm'])} / E1c {f(g(e, 'ate_kf_cm'))} cm → {t}; ATE mọi frame (hiệu chỉnh) {f(pa)} / {f(ea)} cm → {ta}.")
    R["D2"] = "tốt hơn" in tags and "tệ hơn" not in tags and all_ok
    out += [f"- **D2 (hơn E1c?): {'ĐẠT' if R['D2'] else 'KHÔNG ĐẠT'}**"] + lines
    # D3
    t3, lines = [], []
    for s in ("snt_far", "snt_near"):
        p, n = M.get(s, {}).get("pw2"), M.get(s, {}).get("pw2_nopgo")
        if ok_run(p) and ok_run(n):
            t = tag(p["ate_kf_cm"], n["ate_kf_cm"])
            t3.append(t)
            lines.append(f"  - {s}: pw2 {f(p['ate_kf_cm'])} / pw2_nopgo {f(n['ate_kf_cm'])} cm → {t}; độ chính xác closure {f(p.get('closure_precision'), 1)} %.")
    precs = [M[s]["pw2"].get("closure_precision") for s in ("snt_far", "snt_near") if ok_run(M.get(s, {}).get("pw2"))]
    prec_ok = all(x is not None and x >= 90 for x in precs) if precs else False
    R["D3"] = len(t3) == 2 and "tốt hơn" in t3 and "tệ hơn" not in t3 and prec_ok
    out += [f"- **D3 (PGO có ích?): {'ĐẠT' if R['D3'] else 'KHÔNG ĐẠT'}**" + ("" if len(t3) == 2 else " (thiếu run)")] + lines
    # D4
    p, k = M.get("snt_far", {}).get("pw2"), M.get("snt_far", {}).get("base_kf")
    if ok_run(p) and ok_run(k):
        t = tag(p["ate_kf_cm"], k["ate_kf_cm"])
        R["D4"] = t == "tốt hơn"
        out.append(f"- **D4 (lợi không chỉ do keyframe?): {'ĐẠT' if R['D4'] else 'KHÔNG ĐẠT'}** — snt_far pw2 {f(p['ate_kf_cm'])} / base_kf "
                   f"{f(k['ate_kf_cm'])} cm ({k['n_kf']} keyframe) → {t}.")
    else:
        R["D4"] = False
        out.append("- **D4:** thiếu run.")
    # D5
    lines = []
    for s in seqs:
        p, e, b = M[s]["pw2"], M[s].get("e1c"), M[s].get("base")
        rp = p.get("rpe_d1_corr_cm") or p["rpe_d1_cm"]
        lines.append(f"  - {s}: RPE Δ1 (hiệu chỉnh) pw2 {f(rp, 3)}, e1c {f(g(e, 'rpe_d1_cm'), 3)}, base {f(g(b, 'rpe_d1_cm'), 3)} cm.")
    out += ["- **D5 (giật, chỉ báo):**"] + lines
    if R["D1"] and R["D2"] and R["D3"] and R.get("D4"):
        concl = "Tiếp tục: lượt sau thêm plane vào local BA, fr1/room đủ chuỗi, nhiều seed."
    elif not R["D1"]:
        concl = ("D1 không đạt → theo bảng mục 8: báo phân phối e của cặp đúng nhãn và nguyên nhân, không kết luận về D2–D4 "
                 "(D1 trượt ở điều kiện nào: xem các dòng D1 bên dưới và phần diễn giải)")
    elif R["D3"] and not R["D2"]:
        concl = "Chuyển hướng: mặt phẳng là mốc tầm dài (PGO + local BA), plane nhẹ ở tracking."
    elif not R["D2"] and not R["D3"]:
        concl = "Dừng PW, giữ E1c, tìm hướng khác."
    else:
        concl = "Kết quả không rơi đúng vào một dòng của bảng mục 8 (D1, D2 đạt nhưng D3 hoặc D4 không đạt): xem diễn giải."
    return out, concl, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    out = os.path.join(D, "metrics_pw2.json")
    if not a.no_eval:
        args = [sys.executable, "tools/monogs/pw2/eval_pw2.py", "--out", out]
        if a.run1:
            args += ["--only", "snt_far"]
        subprocess.run(args, check=True)
    M = json.load(open(out))
    if a.run1:
        L = ["# R2 — run 1 (pw2, snt_far đủ chuỗi), mốc P1.5\n", "Sinh bằng `tools/monogs/pw2/make_report_pw2.py --run1`.\n"]
        L += seq_table({"snt_far": M["snt_far"]}, "snt_far")
        L += pgo_rows(M, "snt_far", "pw2")
        open(os.path.join(D, "R2_run1.md"), "w").write("\n".join(L) + "\n")
        print("\n".join(L))
        return
    dec, concl, R = decisions(M)
    cal = json.load(open(os.path.join(D, "calib2.json")))
    n = cal["noise"]
    L = ["# REPORT MonoGS-PW2, lượt 2 (R2)\n",
         "Sinh bằng `tools/monogs/pw2/make_report_pw2.py`. Nhánh `exp/monogs-pw2`. 1 seed (0); A **tốt hơn** B khi A ≤ 0,7·B, **tệ hơn** khi "
         "A ≥ 1,3·B. Run base / e1c của snt_far, snt_near lấy từ `results_monogs/B` (snt_near cắt về frame 0–700); st_far base / e1c và `full` "
         "(R1) lấy từ `results_monogs/PW`.\n",
         "## 1. Kết luận lượt\n", f"**Kết luận (luật mục 8):** {concl}\n"] + dec + ["", note("notes_pw2_summary.md"),
         "## 2. Hiệu chỉnh (chi tiết `pw2/CALIB_PW2.md`)\n", "| Thang | σ_n | σ_d(z) |", "|---|---|---|",
         f"| σ_pgo | {f(np.degrees(n['sigma_n']), 3)}° | {f(100 * n['sigma_d_a'], 3)} + {f(100 * n['sigma_d_b'], 4)}·z² cm |",
         f"| σ_eff | {f(np.degrees(n['sigma_eff_n']), 3)}° | {f(100 * n['sigma_eff_d_a'], 3)} + {f(100 * n['sigma_eff_d_b'], 4)}·z² cm |",
         f"| R1 | 0,240° | 0,761 + 0,0042·z² cm |", "",
         f"σ_odo = {f(100 * n['sigma_odo_t'])} cm / {f(np.degrees(n['sigma_odo_r']))}°; θ_m = {f(n['merge_deg'])}°, δ_m = {f(100 * n['merge_m'])} cm.\n",
         "**Kiểm offline kernel** (σ_eff so với σ R1): " + "; ".join(
             f"{s}: đúng nhãn có e ≤ 3: {f(v['eff'][0], 1)} % (R1 {f(v['r1'][0], 1)} %), anchor sai nhãn trong cổng có e ≤ 3: {f(v['eff'][2], 1)} % (R1 {f(v['r1'][2], 1)} %)"
             for s, v in cal["kernel_check"].items()) + ".\n",
         "## 3. κ theo tầm (chi tiết `pw2/KAPPA_HORIZON.md`)\n", note("notes_pw2_kappa.md"),
         "## 4. Test (chi tiết `pw2/TESTS_PW2.md`)\n", note("notes_pw2_tests.md"),
         "## 5. Bảng mỗi chuỗi\n"]
    for s in SPEC:
        L += seq_table(M, s)
    L += ["## 6. Từng lần PGO\n"]
    for s in SPEC:
        L += pgo_rows(M, s, "pw2")
    L += ["## 7. Gợi ý cho lượt sau\n", note("notes_pw2_suggestions.md")]
    for s in SPEC:
        eb = g(M[s].get("pw2"), "pw", "e_by_depth")
        if eb:
            L.append(f"- *{s}, pw2, e theo độ sâu:* " + "; ".join(f"z {k}: trung vị e {f(v['median_e'])}, loại {f(v['pct_rejected'], 1)} % ({v['n']} cặp)" for k, v in eb.items()))
    miss = [f"{s}/{k}" for s in M for k, v in M[s].items() if not ok_run(v)]
    L += ["", "## 8. Run bị bỏ\n", ("Không có." if not miss else "Thiếu: " + ", ".join(miss)) + "\n", note("notes_pw2_runs.md"),
          "## 9. Khác plan\n", note("notes_pw2_deviations.md")]
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figs = []
        for s, sv in SPEC.items():
            fig, ax = plt.subplots(figsize=(8, 3.4))
            for meth in sv["runs"]:
                m = M[s].get(meth)
                if ok_run(m):
                    ax.plot(m["unaligned"], label=meth, lw=1)
            pf = g(M[s].get("pw2"), "pgo_frames") or []
            for i, x in enumerate(pf):
                ax.axvline(x, color="k", ls=":", lw=0.8, label="PGO pw2" if i == 0 else None)
            ax.set_title(f"{sv['title']}: sai số chưa căn theo frame")
            ax.set_xlabel("frame")
            ax.set_ylabel("cm")
            ax.legend(fontsize=7)
            fig.tight_layout()
            p = f"pw2_unaligned_{s}.png"
            fig.savefig(os.path.join(D, p), dpi=110)
            plt.close(fig)
            figs.append(p)
            an = g(M[s].get("pw2"), "plane", "anchors_per_kf")
            if an:
                fig, ax = plt.subplots(figsize=(8, 2.8))
                a_ = np.array(an)
                ax.plot(a_[:, 0], a_[:, 2], label="pw2 active")
                ae = g(M[s].get("e1c"), "plane", "anchors_per_kf")
                if ae:
                    b_ = np.array(ae)
                    ax.plot(b_[:, 0], b_[:, 2], label="e1c active")
                ax.set_title(f"{s}: số anchor active theo keyframe")
                ax.legend(fontsize=7)
                fig.tight_layout()
                p = f"pw2_anchors_{s}.png"
                fig.savefig(os.path.join(D, p), dpi=110)
                plt.close(fig)
                figs.append(p)
        L += ["## Hình\n"] + [f"![{p}](pw2/{p})\n" for p in figs]
    except Exception as e:  # noqa: BLE001
        L.append(f"(không vẽ được hình: {e})\n")
    open("docs/monogs/REPORT_MONOGS_PW2_R2.md", "w").write("\n".join(L) + "\n")
    json.dump(dict(R, conclusion=concl), open(os.path.join(D, "decisions.json"), "w"), indent=1)
    print("written docs/monogs/REPORT_MONOGS_PW2_R2.md;", concl)


if __name__ == "__main__":
    main()
