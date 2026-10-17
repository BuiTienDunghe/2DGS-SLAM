"""PLAN E1c gate P: seg_metrics.json of base/ours seed 0 vs ANALYSIS (q123.json), tolerance 0,01 cm.
usage: python tools/e1c_analysis/check_p.py  -> results_exp/e1c/CHECK_P.md
"""
import json
import os

R = os.path.expanduser("~/2DGS-SLAM/results_exp")
q = json.load(open(os.path.join(R, "e1/analysis/q123.json")))
TOL = 0.01
rows, ok_all = [], True
pairs = [("ATE cục bộ (cm)", "ate_local_cm", lambda Q, m, s: Q["q1"][s][f"{m}_ate_local_cm"]),
         ("RPE dịch Δ1 (cm)", "rpe_trans_cm_d1", lambda Q, m, s: Q["q1"][s][f"{m}_rpe_trans_cm_d1"]),
         ("RPE dịch Δ15 trượt (cm)", "rpe_trans_cm_d15_slide", lambda Q, m, s: Q["q1"][s][f"{m}_rpe_trans_cm_d15_slide"]),
         ("Tỉ lệ quãng đường (×100)", "path_ratio", lambda Q, m, s: 100 * Q["q1"][s][f"{m}_path_ratio"]),
         ("Không căn đầu đoạn (cm)", "raw_start", lambda Q, m, s: Q["q1_raw"][s][m]["start"]),
         ("Không căn cuối đoạn (cm)", "raw_end", lambda Q, m, s: Q["q1_raw"][s][m]["end"])]
for m in ("base", "ours"):
    sm = json.load(open(os.path.join(R, f"e1c/reused/{m}_seed0/seg_metrics.json")))
    for s in ("S1", "S2", "S3", "S4", "S5"):
        for name, key, ref in pairs:
            new = sm["segments"][s][key] * (100 if key == "path_ratio" else 1)
            old = ref(q, m, s)
            d = abs(new - old)
            ok = d <= TOL
            ok_all &= ok
            rows.append(f"| {m} | {s} | {name} | {old:.4f} | {new:.4f} | {d:.2e} | {'✓' if ok else '✗'} |".replace(".", ","))
L = ["# Cổng P — seg_metrics.py so với ANALYSIS (E1 seed 0)\n",
     f"Ngưỡng lệch: {str(TOL).replace('.', ',')} (cm; tỉ lệ quãng đường so theo ×100). Kết quả: **{'ĐẠT' if ok_all else 'KHÔNG ĐẠT'}** ({len(rows)} số).\n",
     "| Run | Đoạn | Chỉ số | ANALYSIS | seg_metrics | Lệch | Đạt |", "|---|---|---|---|---|---|---|"] + rows
open(os.path.join(R, "e1c/CHECK_P.md"), "w").write("\n".join(L) + "\n")
print(L[1])
