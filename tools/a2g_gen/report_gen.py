"""PLAN A2G-P1 gen §7 / §3.5: REPORT_P1_GEN.md. Gate failed -> the report is Step 0 only (summary + STAGE0).
usage: python tools/a2g_gen/report_gen.py     -> results_exp/a2g_p1gen/REPORT_P1_GEN.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT0  # noqa: E402
from stage0_report import NAME, f, gate, lamf  # noqa: E402


def main():
    S = json.load(open(os.path.join(OUT0, "a12_summary.json")))
    A3 = json.load(open(os.path.join(OUT0, "a3_summary.json")))
    rows, n1, n2, ok = gate(S)
    L = S["locked"]
    out = ["# REPORT A2G-P1 gen — mô hình độ bất định \"gia phả sai số\"\n",
           "Sinh từ dữ liệu (`tools/a2g_gen/report_gen.py`). Nhánh `exp/a2g-p1gen` (từ `exp/a2g-p1core` @ 8518042).\n",
           f"**Kết quả cổng (mục 3.5): {'ĐẠT' if ok else 'TRƯỢT'}.** G1 {n1}/4 (cần ≥ 3), G2 {n2}/4 (cần 4). "
           + ("" if ok else "Theo plan, Bước 1 (code) và Bước 2 (chạy GPU) bị bỏ; report này chỉ gồm Bước 0. "
                            "Không có run SLAM mới, nên không có cột P1 gen trong bảng ATE.") + "\n",
           "## Tóm tắt\n",
           "**A3 (cảm biến, pose GT):** khối `PlaneWorld.sensor` khóa: "
           f"`undistort_depth: {str(A3['sensor_locked']['undistort_depth']).lower()}`, "
           f"`fit_excess: {str(A3['sensor_locked']['fit_excess']).lower()}`, `lambda_s: {A3['sensor_locked']['lambda_s']:.4g}`.\n",
           "| Chuỗi | Cặp | NIS TB thô (σ₀ = 0) | NIS TB sau khóa | NIS trung vị sau khóa | % > χ² sau khóa |", "|---|---|---|---|---|---|"]
    for s, q in A3["seq"].items():
        out.append(f"| {s} | {q['raw']['n']} | {f(q['raw']['mean'], 1)} | {f(q['final']['mean'])} | {f(q['final']['median'])} | "
                   f"{f(q['final']['pct_out'], 1)} |")
    out += ["", f"**A1/A2 (gia phả):** biến thể khóa trên snt_near: luật {L['rule']}, ảnh màu {'có' if L['photo'] else 'không'}, "
                f"cha mẹ anchor {'có' if L['anchor'] else 'không'}, λ = {lamf(L['lam'])} (biên lưới).\n",
            "| Trường hợp | NIS₆ TB (E-local) | NIS₃ TB (E-rel) | % > χ² | NLL G | NLL T | NLL D | NLL C | G1 | G2 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        seq, m = r["case"].split("/")
        loc = S["runs"][r["case"]]["local_nis_mean"]
        out.append(f"| {seq} / {NAME[m]}{' (chỉ báo cáo)' if seq == 'room' else ''} | {f(loc)} | {f(r['nis'])} | {f(r['out'], 1)} | "
                   + " | ".join(f(r["nll"][x], 3) for x in "GTDC") + f" | {'đạt' if r['g1'] else 'trượt'} | {'đạt' if r['g2'] else 'trượt'} |")
    out += ["", "**Kết luận ngắn:**\n",
            "1. Thang thông tin tính theo pixel độc lập (Σ 1/σ_z²) không chuyển được giữa các chuỗi: λ khóa trên snt_near (≥ 10^5) "
            "lớn gấp khoảng 10 lần λ của snt_far, st_far. G vì thế quá thận trọng trên hai chuỗi far (NIS₃ 0,48–0,78), nên G1 trượt 4/4. "
            "A3 cho thấy cùng hệ số khoảng 10 giữa near và far.",
            "2. Trên fr3, sai số tương đối frame–anchor gần như đứng yên: mô hình hằng C thắng ở 3/4 trường hợp, kể cả khi mọi mô hình được fit "
            "thang trên chính run (chẩn đoán 2.3). Trên room (khám phá, khoảng 300 keyframe), G là mô hình duy nhất còn hiệu chỉnh tốt và có NLL thấp nhất.",
            "3. Cấu trúc trộn cha mẹ (M3) khớp với đa số frame (NIS₆ trung vị 3–7), nhưng sai số có đuôi nặng; ảnh màu và cha mẹ anchor không giúp.\n",
            "Chi tiết, phân tích, đề xuất và danh sách khác plan ở phần STAGE0 dưới đây.\n", "---\n"]
    st = open(os.path.join(OUT0, "STAGE0.md")).read().replace("](sd_vs_absd_snt_far.png)", "](sd_vs_absd_snt_far.png)")
    out.append(st)
    p = os.path.join(os.path.dirname(OUT0), "REPORT_P1_GEN.md")
    open(p, "w").write("\n".join(out) + "\n")
    print(f"written {p}")


if __name__ == "__main__":
    main()
