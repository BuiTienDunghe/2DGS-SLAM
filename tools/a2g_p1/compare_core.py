"""PLAN A2G-P1 core §8: REPORT_P1_CORE.md from data (one block per sequence, base / E1c / P1 v1 / P1 core).
usage: python tools/a2g_p1/compare_core.py --spec SPEC.json --out results_exp/a2g_p1core/REPORT_P1_CORE.md
       [--appendix APPENDIX.md]
SPEC: {"seq": {"base": dir, "e1c": dir, "p1v1": dir or null, "p1core": dir}, ...}
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute  # noqa: E402

COLS = (("base", "baseline"), ("e1c", "E1c"), ("p1v1", "P1 v1"), ("p1core", "P1 core"))


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{nd}f}".replace(".", ",")


def pct(a, b):
    return 100.0 * (a - b) / b


def verdict(d):
    return "trong nhiễu" if abs(d) < 15 else ("tốt hơn" if d < 0 else "tệ hơn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--appendix", default=None)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    outdir = os.path.dirname(a.out)
    R = {}
    for seq, runs in spec.items():
        R[seq] = {}
        for k, _ in COLS:
            d = runs.get(k)
            if d and os.path.exists(os.path.join(d, "e1_metrics.json")):
                try:
                    R[seq][k] = compute(d)
                except Exception as e:  # noqa: BLE001
                    R[seq][k] = dict(error=str(e))
    json.dump({s: {k: {kk: vv for kk, vv in v.items() if kk != "err_unaligned_cm"} for k, v in d.items()}
               for s, d in R.items()}, open(os.path.join(outdir, "p1core_metrics.json"), "w"), indent=1)
    L = ["# REPORT A2G-P1 core: ATE của plane world (một mô hình độ bất định) so với baseline và E1c\n",
         "Sinh tự động từ dữ liệu (`tools/a2g_p1/compare_core.py`). 1 seed; hai run cùng code đã lệch nhau 5–12 cm (T7), "
         "hai run baseline trên snt_far cho 14,64 / 12,63 cm; chênh lệch ATE dưới khoảng 15 % ghi là **trong nhiễu**. "
         "P1 v1 và P1 core khác nhau ở plane world nhưng cùng frontend, nên so được trực tiếp.\n"]
    rows = (("ATE mọi frame (cm)", "ate_all_cm", 2), ("ATE keyframe (cm)", "ate_kf_cm", 2), ("RPE Δ1 (cm)", "rpe_d1_cm", 2),
            ("Sai số chưa căn cuối (cm)", "end_err_cm", 1), ("Sai số chưa căn lớn nhất (cm)", "max_err_cm", 1),
            ("% frame khởi tạo bằng plane", "pct_init_plane", 1), ("ms/frame tracking", "ms_track", 0),
            ("ms/frame plane world", "ms_pw", 1))
    summary = []
    for seq in spec:
        L.append(f"## {seq}\n")
        L.append("| Chỉ số | " + " | ".join(n for _, n in COLS) + " |")
        L.append("|---|" + "---|" * len(COLS))
        for name, key, nd in rows:
            L.append(f"| {name} | " + " | ".join(f(R[seq].get(k, {}).get(key), nd) for k, _ in COLS) + " |")
        L.append("")
        p, e, b = (R[seq].get(k, {}).get("ate_all_cm") for k in ("p1core", "e1c", "base"))
        if p is not None and e is not None and b is not None:
            de, db = pct(p, e), pct(p, b)
            summary.append(f"- **{seq}:** P1 core {f(p)} cm; so với E1c {f(e)} cm: {de:+.1f} % ({verdict(de)}); "
                           f"so với baseline {f(b)} cm: {db:+.1f} % ({verdict(db)}).".replace(".", ",", 0))
        elif p is None:
            summary.append(f"- **{seq}:** không có run P1 core hoàn chỉnh.")
    L.append("## Tóm tắt\n")
    L += summary
    L.append("")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for seq in spec:
            fig, ax = plt.subplots(figsize=(7, 3.4))
            for k, name in COLS:
                if "err_unaligned_cm" in R[seq].get(k, {}):
                    ax.plot(R[seq][k]["err_unaligned_cm"], label=name)
            ax.set_title(f"{seq}: sai số chưa căn theo frame")
            ax.set_xlabel("frame track")
            ax.set_ylabel("cm")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"err_unaligned_{seq}.png"), dpi=110)
            plt.close(fig)
            L.append(f"![{seq}](err_unaligned_{seq}.png)\n")
    except Exception as e:  # noqa: BLE001
        L.append(f"(Không vẽ được hình: {e})\n")
    L.append("## Phụ lục: chỉ số plane world không cần nhãn\n")
    L.append("| Chuỗi | Phương pháp | Anchor primary cuối | Active cuối | Cặp gần trùng ở cuối | Đổi ID: bản trùng | Đổi ID: nghi ghép sai | Gộp |")
    L.append("|---|---|---|---|---|---|---|---|")
    for seq in spec:
        for k, name in COLS[1:]:
            m = R[seq].get(k)
            if not m or "error" in m:
                continue
            L.append(f"| {seq} | {name} | {m.get('n_anchor_final')} | {m.get('n_live_final')} | {m.get('near_dup_end')} | "
                     f"{m.get('switch_dup')} | {m.get('switch_suspect')} | {m.get('merges', '—')} |")
    L.append("")
    L.append("| Chuỗi | P1 core: NIS IC dữ liệu mới (TB / trung vị, n) | NIS IC dữ liệu cũ (TB, n) | NIS ghép lại keyframe dữ liệu mới (TB, n) | % cặp không mơ hồ | b̂ trung vị (góc °, offset mm) | Hoãn | revisit_candidate | Mẫu b̂ nhận / từ chối / mơ hồ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for seq in spec:
        m = R[seq].get("p1core")
        if not m or not m.get("core"):
            continue
        L.append(f"| {seq} | {f(m['nis_ic_fresh_mean'])} / {f(m['nis_ic_fresh_median'])} ({m['n_ic_fresh']}) | "
                 f"{f(m['nis_ic_stale_mean'])} ({m['n_ic_stale']}) | {f(m['nis_reassign_fresh_mean'])} ({m['n_reassign_fresh']}) | "
                 f"{f(m['pct_unambiguous'], 1)} | {f(m['bhat_theta_deg_median'], 3)}, {f(m['bhat_d_mm_median'], 2)} | "
                 f"{m['deferred']} | {m['revisit_candidates']} | {m['bias_accept']} / {m['bias_reject']} / {m['bias_ambiguous']} |")
    L.append("")
    if a.appendix and os.path.exists(a.appendix):
        L.append(open(a.appendix).read())
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
