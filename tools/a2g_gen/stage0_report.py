"""PLAN A2G-P1 gen §3: STAGE0.md from the Step 0 outputs (a3.md / a3_summary.json, a12_summary.json, a12_diag.json).
usage: python tools/a2g_gen/stage0_report.py [--gate]     (--gate adds the §3.5 decision, computed here)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT0  # noqa: E402

GATE_CASES = ["snt_far/dec", "snt_far/core", "st_far/dec", "st_far/core"]
NAME = {"dec": "E1c", "core": "P1 core", "base": "baseline"}
RULE = {"closest": "P-closest", "first": "P-first", "closest_act": "P-closest-act", "first_act": "P-first-act"}


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{nd}f}".replace(".", ",")


def lamf(x):
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return (f"{m:.2g}".replace(".", ",") + f"·10^{e}") if abs(m - 1) > 1e-6 else f"10^{e}"


def gate(S):
    lam = S["locked"]["lam"]
    rows = []
    for c in GATE_CASES + ["room/dec", "room/core"]:
        r = S["runs"][c]
        e = r["erel"]
        g1 = (2.0 <= e["nis_mean"] <= 4.5) and e["pct_out"] <= 3.0
        nl = e["nll"]
        g2 = nl["G"] < min(nl["T"], nl["D"], nl["C"])
        g3 = lam / 2 <= r["lam_fit"] <= 2 * lam
        rows.append(dict(case=c, g1=g1, g2=g2, g3=g3, nis=e["nis_mean"], out=e["pct_out"], nll=nl, lam=r["lam_fit"]))
    n1 = sum(r["g1"] for r in rows if r["case"] in GATE_CASES)
    n2 = sum(r["g2"] for r in rows if r["case"] in GATE_CASES)
    return rows, n1, n2, (n1 >= 3 and n2 == 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    A3 = json.load(open(os.path.join(OUT0, "a3_summary.json")))
    S = json.load(open(os.path.join(OUT0, "a12_summary.json")))
    Dg = json.load(open(os.path.join(OUT0, "a12_diag.json"))) if os.path.exists(os.path.join(OUT0, "a12_diag.json")) else None
    L = S["locked"]
    out = ["# STAGE0 — A2G-P1 gen, Bước 0: kiểm giả định trên dữ liệu thật\n",
           "Sinh từ dữ liệu (`tools/a2g_gen/stage0_report.py`). CPU, không chạy SLAM. GT chỉ dùng trong `tools/`. "
           "Nhánh `exp/a2g-p1gen` (từ `exp/a2g-p1core` @ 8518042).\n",
           "**Run dùng lại** (không chạy lại; đủ file theo `check_run_files.py`, riêng baseline snt_far thiếu 2 cột timing không dùng): "
           "snt_near E1c `results_exp/e1f/fr3_snt_near/dec/seed0` (cả chuỗi, ATE 9,59 cm); snt_far / st_far / room: base, E1c, P1 core như mục 3.1 của plan.\n",
           "## 1. A3 — mô hình cảm biến\n", open(os.path.join(OUT0, "a3.md")).read(),
           "## 2. A1 / A2 — gia phả sai số\n",
           "ε thật = Log(T_cw⁻¹ T̂_cw) theo frame, không căn quỹ đạo. Pixel thưa bước 8 px (80×60), depth 0,1–4 m. "
           f"Chủ sở hữu theo 4 luật; Q_mot theo M5 (30 chuyển động gần nhất); λ tìm trên lưới log 10^-1 … 10^5 (bước 0,1 thập phân) bằng ML của E-local. "
           "E-local: NIS₆ (kỳ vọng 6). E-rel: d = J_ε(ε_k − ε_t(j)) cho mọi cặp pose của frame k; NIS₃ (kỳ vọng 3, % > χ²₃(0,99) kỳ vọng 1 %) và NLL "
           "trung bình mỗi cặp (thấp hơn là tốt hơn) của gia phả G và ba mô hình T (random walk theo thời gian), D (theo quãng đường), C (hằng số), "
           "fit ML trên snt_near.\n",
           "### 2.1 Chọn biến thể trên snt_near (run E1c)\n",
           "| Luật chủ | Ảnh màu | Cha mẹ anchor | λ fit (E-local) | NIS₆ TB | NIS₆ trung vị | E-rel NIS₃ TB | % > χ² | NLL G |",
           "|---|---|---|---|---|---|---|---|---|"]
    for v in S["variants"]:
        e = v["erel"]
        out.append(f"| {RULE[v['rule']]} | {'có' if v['photo'] else 'không'} | {'có' if v['anchor'] else 'không'} | "
                   f"{lamf(v['lam_fit'])}{' (biên)' if v['lam_edge'] else ''} | {f(v['local_nis_mean'])} | {f(v['local_nis_median'])} | "
                   f"{f(e.get('nis_mean'))} | {f(e.get('pct_out'), 1)} | {f(e.get('nll', {}).get('G'), 3)} |")
    out += ["", f"**Khóa:** luật **{RULE[L['rule']]}**, ảnh màu **{'có' if L['photo'] else 'không'}**, cha mẹ anchor "
                f"**{'có' if L['anchor'] else 'không'}**, **λ = {lamf(L['lam'])}** (NLL G thấp nhất của E-rel trên snt_near). "
                "λ nằm ở **biên trên của lưới plan**" + (f"; mở rộng lưới tới 10^9 (chỉ chẩn đoán) thì cực trị ở λ ≈ "
                                                        f"{lamf(Dg['runs']['snt_near/dec']['lambda_extended']['lam_opt'])}." if Dg else ".") + "\n",
            "Mô hình so sánh fit trên snt_near: Γ (chéo, mỗi frame track) = " + ", ".join(f"{x:.2e}" for x in L["models"]["Gamma"])
            + f"; κ_t² = {L['models']['kappa_t2']:.3e} m²/m, κ_r² = {L['models']['kappa_r2']:.3e} rad²/rad; Σ₀ đầy đủ 6×6.\n",
            "### 2.2 Áp biến thể và λ đã khóa cho mọi run\n",
            "| Chuỗi / run | Frame | KF | σ_c | % pixel có chủ | λ fit riêng (G3) | NIS₆ TB | NIS₆ trung vị | Cặp E-rel | NIS₃ TB | NIS₃ trung vị | % > χ² | NLL G | NLL T | NLL D | NLL C |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in S["runs"].items():
        e = r["erel"]
        nl = e.get("nll") or {}
        seq, m = k.split("/")
        out.append(f"| {seq} / {NAME[m]} | {r['n_frames']} | {r['n_kf']} | {f(r['sigma_c'], 4)} | {f(100 * r['owned_frac'], 0)} | "
                   f"{lamf(r['lam_fit'])}{' (biên)' if r['lam_edge'] else ''} | {f(r['local_nis_mean'])} | {f(r['local_nis_median'])} | "
                   f"{e['n']} | {f(e.get('nis_mean'))} | {f(e.get('nis_median'))} | {f(e.get('pct_out'), 1)} | "
                   + " | ".join(f(nl.get(x), 3) for x in "GTDC") + " |")
    out += ["", "![sd dự đoán so với |d| thật trên snt_far](sd_vs_absd_snt_far.png)\n",
            "*Hình: snt_far, hàng offset của d. Chấm đen là |d| thật; các dải màu là sd dự đoán của G, T, D, C (thang log).*\n"]
    if Dg:
        out += ["### 2.3 Chẩn đoán (chỉ báo cáo, không dùng cho cổng): thang riêng từng run\n",
                "λ của G fit lại trên E-local của chính run; T, D, C fit lại trên chính các cặp E-rel của run (lợi thế cho T/D/C, vì fit và chấm trên cùng dữ liệu). "
                "Câu hỏi: khi thang đã đúng, **cấu trúc** gia phả có tốt hơn không?\n",
                "| Run | λ riêng | NIS₃ TB | % > χ² | NLL G | NLL T | NLL D | NLL C | Thắng |", "|---|---|---|---|---|---|---|---|---|"]
        for k, r in Dg["runs"].items():
            o = r["own_scale"]
            nl = o["nll"]
            win = min(nl, key=nl.get)
            seq, m = k.split("/")
            out.append(f"| {seq} / {NAME[m]} | {lamf(o['lam_run'])} | {f(o['nis_mean'])} | {f(o['pct_out'], 1)} | "
                       + " | ".join(f(nl[x], 3) for x in "GTDC") + f" | {win} |")
        out.append("")
    if a.gate:
        rows, n1, n2, ok = gate(S)
        out += ["## 3. Cổng (mục 3.5)\n",
                f"Biến thể và λ khóa trên snt_near. G1: NIS₃ TB trong [2; 4,5] và % > χ²₃(0,99) ≤ 3 % (cần ≥ 3/4). "
                f"G2: NLL TB của G thấp hơn cả T, D, C (cần 4/4). G3: λ fit riêng trong [λ/2; 2λ] (chỉ báo cáo). room chỉ báo cáo.\n",
                "| Trường hợp | NIS₃ TB | % > χ² | G1 | NLL G / T / D / C | G2 | λ riêng | G3 |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            seq, m = r["case"].split("/")
            tag = " (chỉ báo cáo)" if seq == "room" else ""
            out.append(f"| {seq} / {NAME[m]}{tag} | {f(r['nis'])} | {f(r['out'], 1)} | {'đạt' if r['g1'] else 'trượt'} | "
                       + " / ".join(f(r["nll"][x], 3) for x in "GTDC") + f" | {'đạt' if r['g2'] else 'trượt'} | {lamf(r['lam'])} | "
                       f"{'đạt' if r['g3'] else 'trượt'} |")
        out += ["", f"**G1: {n1}/4 (cần ≥ 3). G2: {n2}/4 (cần 4). → Cổng {'ĐẠT' if ok else 'TRƯỢT'}.** "
                + ("Sang Bước 1." if ok else "Bỏ Bước 1–2; report chỉ gồm Bước 0 (mục 3.5 của plan).") + "\n"]
        an = os.path.join(OUT0, "analysis.md")
        if os.path.exists(an):
            out.append(open(an).read())
    dv = os.path.join(OUT0, "deviations.md")
    if os.path.exists(dv):
        out.append(open(dv).read())
    open(os.path.join(OUT0, "STAGE0.md"), "w").write("\n".join(out) + "\n")
    print(f"written {os.path.join(OUT0, 'STAGE0.md')}")
    if a.gate:
        print(f"gate: G1 {n1}/4, G2 {n2}/4 -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
