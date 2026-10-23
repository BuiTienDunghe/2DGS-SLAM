"""PLAN A2G-P1 (rút gọn) §9: REPORT_P1_QUICK.md from data (B1–B4, figure, decision table).
usage: python tools/a2g_p1/compare.py --seqs snt_far,st_far --base snt_far=DIR,st_far=DIR --e1c ... --p1 ... \
       --out results_exp/a2g_p1q/REPORT_P1_QUICK.md [--notes NOTES.md]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute  # noqa: E402


def kv(s):
    return dict(x.split("=", 1) for x in s.split(","))


def f(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--e1c", required=True)
    ap.add_argument("--p1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default=None)
    a = ap.parse_args()
    seqs = a.seqs.split(",")
    B, E, P = kv(a.base), kv(a.e1c), kv(a.p1)
    R = {s: dict(base=compute(B[s]), e1c=compute(E[s]), p1=compute(P[s])) for s in seqs}
    outdir = os.path.dirname(a.out)
    json.dump({s: {k: {kk: vv for kk, vv in v.items() if kk != "err_unaligned_cm"} for k, v in d.items()}
               for s, d in R.items()}, open(os.path.join(outdir, "p1q_metrics.json"), "w"), indent=1)
    L = []
    if a.notes:
        L.append(open(a.notes).read().rstrip() + "\n")
    L.append("## B1. Tracking (1 seed)\n")
    L.append("| Chuỗi | Phương pháp | ATE mọi frame (cm) | ATE keyframe (cm) | RPE Δ1 (cm) | Sai số chưa căn cuối / lớn nhất (cm) | % frame khởi tạo bằng plane | ms/frame tracking | ms/frame plane world |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in seqs:
        for k, name in (("base", "baseline"), ("e1c", "E1c"), ("p1", "P1")):
            m = R[s][k]
            L.append(f"| {s} | {name} | {f(m['ate_all_cm'])} | {f(m['ate_kf_cm'])} | {f(m['rpe_d1_cm'])} | "
                     f"{f(m['end_err_cm'], 1)} / {f(m['max_err_cm'], 1)} | {f(m['pct_init_plane'], 1)} | {f(m['ms_track'], 0)} | "
                     f"{f(m['ms_pw'], 1)} |")
    L.append("\n## B2. Plane world (chỉ số không cần nhãn)\n")
    L.append("| Chuỗi | Phương pháp | Anchor primary cuối | Anchor active/frozen cuối | Cặp gần trùng ở cuối | Đổi ID: bản trùng | Đổi ID: nghi ghép sai | Số liên kết segment | Gộp | Nghi gộp nhầm | Xung đột | Sự kiện quay lại | Candidate bị xóa |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in seqs:
        for k, name in (("e1c", "E1c"), ("p1", "P1")):
            m = R[s][k]
            L.append(f"| {s} | {name} | {m['n_anchor_final']} | {m['n_live_final']} | {m['near_dup_end']} | {m['switch_dup']} | "
                     f"{m['switch_suspect']} | {m['id_links']} | {m.get('merges', '—')} | {m.get('merges_suspect', '—')} | "
                     f"{m.get('conflicts', '—')} | {m.get('revisits', '—')} | {m.get('cand_deleted', '—')} |")
    L.append("\n## B3. JC và khởi tạo (P1)\n")
    L.append("| Chuỗi | Frame có JC | % hạng 6 | % frame bỏ cặp (shape) | % jump trượt | Cặp bị bỏ | Nguồn khởi tạo (plane / jc_jump / rank_lt6 / none_pairs) |")
    L.append("|---|---|---|---|---|---|---|")
    for s in seqs:
        m = R[s]["p1"]
        src = m.get("init_sources", {})
        L.append(f"| {s} | {m.get('jc_frames')} | {f(m.get('pct_rank6'), 1)} | {f(m.get('pct_shape_drop'), 1)} | "
                 f"{f(m.get('pct_jump'), 1)} | {m.get('pairs_dropped')} | {src.get('plane')} / {src.get('jc_jump')} / "
                 f"{src.get('rank_lt6')} / {src.get('none_pairs')} |")
    L.append("\n## B4. NIS (P1)\n")
    L.append("| Chuỗi | EKF (K3): trung bình / trung vị (số lần) | IC (K2) của cặp được gán: trung bình / trung vị (số cặp) |")
    L.append("|---|---|---|")
    for s in seqs:
        m = R[s]["p1"]
        L.append(f"| {s} | {f(m.get('nis_ekf_mean'))} / {f(m.get('nis_ekf_median'))} ({m.get('n_ekf')}) | "
                 f"{f(m.get('nis_ic_mean'))} / {f(m.get('nis_ic_median'))} ({m.get('n_ic')}) |")
    sus = [(s, x) for s in seqs for x in R[s]["p1"].get("merges_suspect_list", [])]
    if sus:
        L.append("\nNghi gộp nhầm (lệch > 3° hoặc > 3 cm tại o của anchor bị gộp), cần soát tay:\n")
        for s, x in sus:
            L.append(f"- {s}, keyframe {x['k']}: giữ {x['keep']}, gộp {x['drop']}, lệch {f(x['angle_deg'])}°, "
                     f"{f(x['dist_cm'])} cm")
    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(seqs), figsize=(6 * len(seqs), 3.6))
        ax = np.atleast_1d(ax)
        for i, s in enumerate(seqs):
            for k, name in (("base", "baseline"), ("e1c", "E1c"), ("p1", "P1")):
                ax[i].plot(R[s][k]["err_unaligned_cm"], label=name)
            ax[i].set_title(f"fr3 {s}: sai số chưa căn")
            ax[i].set_xlabel("frame track")
            ax[i].set_ylabel("cm")
            ax[i].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "err_unaligned.png"), dpi=110)
        L.append("\n![Sai số chưa căn theo frame](err_unaligned.png)\n")
    except Exception as e:  # noqa: BLE001
        L.append(f"\n(Không vẽ được hình: {e})\n")
    # decision
    st = R.get("st_far", {})
    dec = {}
    if st:
        e1, p1 = st["e1c"], st["p1"]
        dec["a"] = (p1["switch_dup"] <= 0.10 * max(e1["switch_dup"], 0) and p1["near_dup_end"] <= 1,
                    f"st_far: đổi ID bản trùng P1 {p1['switch_dup']} so với E1c {e1['switch_dup']} (ngưỡng ≤ 10 %), "
                    f"cặp gần trùng cuối P1 {p1['near_dup_end']} (ngưỡng ≤ 1)")
    n_sus = sum(R[s]["p1"].get("merges_suspect", 0) for s in seqs)
    dec["b"] = (n_sus == 0, f"{n_sus} lần nghi gộp nhầm" + ("" if n_sus == 0 else ", chưa soát tay"))
    ok_c = all(R[s]["p1"]["ate_all_cm"] <= 1.15 * R[s]["e1c"]["ate_all_cm"] for s in seqs)
    dec["c"] = (ok_c, "; ".join(f"{s}: P1/E1c = {f(R[s]['p1']['ate_all_cm'] / R[s]['e1c']['ate_all_cm'])}" for s in seqs))
    nis = [R[s]["p1"].get("nis_ekf_mean") for s in seqs]
    dec["d"] = (all(x is not None and 2 <= x <= 4 for x in nis), "NIS EKF trung bình " + ", ".join(f(x) for x in nis))
    inv = [R[s]["p1"].get("invariant_violations") for s in seqs]
    nan = [R[s]["p1"].get("has_nan") for s in seqs]
    dec["e"] = (all(x == 0 for x in inv) and not any(nan), f"vi phạm bất biến {inv}, NaN {nan}")
    L.append("\n## Quyết định (theo mục 9 của plan, khóa trước run chính)\n")
    L.append("| Tiêu chí | Đạt? | Số liệu |")
    L.append("|---|---|---|")
    for k in "abcde":
        if k in dec:
            L.append(f"| ({k}) | {'đạt' if dec[k][0] else 'trượt'} | {dec[k][1]} |")
    allok = all(v[0] for v in dec.values())
    part = dec.get("b", (False,))[0] and dec.get("e", (False,))[0]
    ratio = max(R[s]["p1"]["ate_all_cm"] / R[s]["e1c"]["ate_all_cm"] for s in seqs)
    if allok:
        verdict = "**Đạt → đi tiếp** (sang Phần 2)."
    elif not dec["e"][0] or ratio > 1.5:
        verdict = "**Trượt**: dừng, làm phần chẩn đoán ở mục 10 trước khi sửa."
    elif part:
        verdict = "**Đạt một phần**: làm chẩn đoán mục 10 cho chuỗi bị trượt, sửa, rồi chạy lại riêng chuỗi đó."
    else:
        verdict = "**Chưa kết luận**: có lần nghi gộp nhầm chưa soát tay; các tiêu chí khác xem bảng."
    if not allok and 1.15 < ratio <= 1.5:
        verdict += " ATE nằm giữa 1,15 và 1,5 lần E1c: theo plan cần thêm seed 1 cho chuỗi đó trước khi kết luận."
    L.append(f"\n{verdict}\n")
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
