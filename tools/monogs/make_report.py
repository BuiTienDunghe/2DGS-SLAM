"""PLAN MONOGS-E1C §4, §6, §7: tables, figures and reports, generated from the run outputs.
usage (repo root, env monogs):
  python tools/monogs/make_report.py --stage A        -> docs/monogs/PART_A.md (+ metrics_A.json, pair_A.json)
  python tools/monogs/make_report.py --stage full     -> docs/monogs/REPORT_MONOGS_E1C.md (+ metrics_all.json, figures)
Free-text sections (Khác plan, sự cố) are read from docs/monogs/notes_*.md when present.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_runs import pair_metrics, run_metrics  # noqa: E402

M = "results_monogs"
D = "docs/monogs"
PAPER_FR1_DESK = 1.50          # MonoGS arXiv 2312.06741 v3, Table 1, RGB-D w/o loop closure, ATE RMSE cm (keyframes, mean of 3 runs, multi-process, RTX 4090)
A_RUNS = [("sp_r1", "đơn luồng, seed 0"), ("sp_r2", "đơn luồng, seed 0, chạy lại"), ("mt_r1", "song song, seed 0")]
B_SEQ = [("snt_far", "fr3/structure_notexture_far"), ("snt_near", "fr3/structure_notexture_near")]
B_METH = [("base", "base"), ("e1c", "e1c (λ_p = 1)"), ("e1c_s", "e1c_s (λ_p = 0,1)")]
REF_2DGS = {"snt_far": (14.64, 3.80), "snt_near": (58.27, 9.59)}
REF_PUB = {"snt_far": "MonoGS 16,63 (theo StructureGS), 10,39 (theo Spectral); StructureGS-SLAM 7,47",
           "snt_near": "MonoGS 55,59 (theo StructureGS), 26,91 (theo Spectral); StructureGS-SLAM 16,36"}


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{nd}f}".replace(".", ",")


def note(name):
    p = os.path.join(D, name)
    return open(p).read() if os.path.exists(p) else ""


def verdict_r(r):
    return "khớp" if r <= 1.5 else ("gần" if r <= 2 else "chưa khớp")


def part_a():
    runs = {k: os.path.join(M, "A/fr1_desk", k) for k, _ in A_RUNS}
    met = {k: run_metrics(p) for k, p in runs.items() if os.path.exists(os.path.join(p, "traj_all.txt"))}
    pair = None
    if "sp_r1" in met and "sp_r2" in met:
        pair = pair_metrics(runs["sp_r1"], runs["sp_r2"])
    os.makedirs(D, exist_ok=True)
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "unaligned_cm"} for k, v in met.items()},
              open(os.path.join(D, "metrics_A.json"), "w"), indent=1)
    if pair:
        json.dump(pair, open(os.path.join(D, "pair_A.json"), "w"), indent=1)
    L = ["### Phần A — MonoGS trên fr1/desk\n",
         f"Số paper: **{f(PAPER_FR1_DESK)} cm** (MonoGS, arXiv 2312.06741, Table 1, RGB-D, không loop closure; ATE RMSE trên keyframe, "
         "trung bình 3 run, chế độ song song, RTX 4090). Ở đây mỗi run là 1 lần chạy, RTX 5060 Ti.\n",
         "| Run | Chế độ | ATE keyframe (cm) | ATE mọi frame (cm) | Số keyframe | Số frame | ms/frame tracking | s/frame tổng | r so với paper | Đánh giá |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, lab in A_RUNS:
        m = met.get(k)
        if not m:
            L.append(f"| {k} | {lab} | không có run hoàn chỉnh | | | | | | | |")
            continue
        r = m["ate_kf_cm"] / PAPER_FR1_DESK
        L.append(f"| {k} | {lab} | {f(m['ate_kf_cm'])} | {f(m['ate_all_cm'])} | {m['n_kf']} | {m['n_frames']} | {f(m['ms_track'], 0)} | "
                 f"{f(m['s_per_frame'])} | {f(r)} | {verdict_r(r)} |")
    L.append("")
    n_rel = None
    if pair:
        n_rel = pair["rel_noise_kf"]
        L += ["**sp_r1 so với sp_r2 (cùng code, cùng seed 0, chạy lại):**\n",
              "| Chỉ số | Giá trị |", "|---|---|",
              f"| Danh sách keyframe trùng nhau | {'có' if pair['kf_identical'] else 'không'} ({pair['n_kf'][0]} / {pair['n_kf'][1]} keyframe, chung {pair['n_kf_common']}) |",
              f"| Lệch tâm camera lớn nhất ở keyframe chung, không căn (cm) | {f(pair['max_kf_centre_diff_cm'], 3)} |",
              f"| Lệch tâm camera lớn nhất trên mọi frame, không căn (cm) | {f(pair['max_frame_centre_diff_cm'], 3)} |",
              f"| Frame đầu tiên lệch > 1 mm | {pair['first_frame_diff_gt_1mm'] if pair['first_frame_diff_gt_1mm'] is not None else 'không có'} |",
              f"| ATE keyframe r1 / r2 (cm) | {f(pair['ate_kf_cm'][0])} / {f(pair['ate_kf_cm'][1])} |",
              f"| ATE mọi frame r1 / r2 (cm) | {f(pair['ate_all_cm'][0])} / {f(pair['ate_all_cm'][1])} |",
              f"| Mức nhiễu tương đối n = \\|ATE r1 − ATE r2\\| / ATE r1 (keyframe) | {f(100 * n_rel, 1)} % |", ""]
    return met, pair, n_rel, "\n".join(L) + "\n"


def env_table():
    p = os.path.join(D, "env.json")
    if not os.path.exists(p):
        return ""
    e = json.load(open(p))
    rows = [("GPU, driver, VRAM", e["gpu"]), ("CPU", e["cpu"]), ("CUDA toolkit", e["cuda_toolkit"]),
            ("PyTorch / CUDA / cuDNN", f"{e['torch']} / {e['torch_cuda']} / {e['cudnn']}"), ("Open3D", e["open3d"]),
            ("NumPy", e["numpy"]), ("evo / matplotlib", f"{e['evo']} / {e['matplotlib']}"), ("Python", e["python"]),
            ("WSL kernel", e["wsl_kernel"]), ("MonoGS", e["monogs_commit"]),
            ("Submodule diff-gaussian-rasterization", e["submodules"]["diff-gaussian-rasterization"][:7]),
            ("Submodule simple-knn", e["submodules"]["simple-knn"][:7]),
            ("TF32 mặc định (matmul / cuDNN)", f"{e['tf32_default_matmul']} / {e['tf32_default_cudnn']}")]
    L = ["| Mục | Giá trị |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    L += ["", "| Chuỗi | Đường dẫn | Dòng trong rgb.txt |", "|---|---|---|"]
    for k, v in e["datasets"].items():
        L.append(f"| {k} | `{v['path']}` | {v['rgb_lines']} |")
    return "\n".join(L) + "\n"


def part_b(n_rel):
    n = max(n_rel if n_rel is not None else 0.0, 0.15)
    met = {}
    for s, _ in B_SEQ:
        for k, _ in B_METH:
            p = os.path.join(M, "B", s, k)
            if os.path.exists(os.path.join(p, "traj_all.txt")):
                met[(s, k)] = run_metrics(p)
    L = [f"### Phần B — E1c trên MonoGS\n",
         f"Mức nhiễu dùng để đọc kết quả: n = max(mức nhiễu phần A, 15 %) = **{f(100 * n, 1)} %**. "
         f"\"E1c tốt hơn base\" khi ATE_e1c ≤ (1 − 2n) × ATE_base = {f(1 - 2 * n)} × ATE_base, xét trên ATE keyframe (chỉ số chính) "
         "và ghi thêm cho ATE mọi frame. \"Không làm giật hơn\" khi RPE Δ1 ≤ 1,1 × base. 1 seed.\n"]
    concl = []
    for s, title in B_SEQ:
        L.append(f"#### {title}\n")
        L.append("| Method | ATE keyframe (cm) | ATE mọi frame (cm) | RPE Δ1 (cm) | Sai số không căn cuối / lớn nhất (cm) | Keyframe | "
                 "% khởi tạo bằng plane | % σ_min ≥ 0,3 | Anchor tổng / active | Cặp gần trùng | Đổi ID: bản trùng / nghi ghép sai | "
                 "ms/frame tracking | ms/frame plane |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, lab in B_METH:
            m = met.get((s, k))
            if not m:
                L.append(f"| {lab} | không có run hoàn chỉnh |" + " |" * 11)
                continue
            p = m["plane"] or {}
            L.append(f"| {lab} | {f(m['ate_kf_cm'])} | {f(m['ate_all_cm'])} | {f(m['rpe_d1_cm'], 3)} | {f(m['unaligned_end_cm'], 1)} / "
                     f"{f(m['unaligned_max_cm'], 1)} | {m['n_kf']} | {f(p.get('pct_init_plane'), 1)} | {f(p.get('pct_sigma_ge_03'), 1)} | "
                     f"{p.get('anchors_total', '—')} / {p.get('anchors_active', '—')} | {p.get('near_dup_pairs', '—')} | "
                     f"{p.get('id_switch_dup', '—')} / {p.get('id_switch_suspect', '—')} | {f(m['ms_track'], 0)} | {f(m['ms_plane'], 1)} |")
        L.append("")
        b = met.get((s, "base"))
        for k, lab in B_METH[1:]:
            e = met.get((s, k))
            if not b or not e:
                continue
            for key, name in (("ate_kf_cm", "ATE keyframe"), ("ate_all_cm", "ATE mọi frame")):
                d = 100 * (e[key] - b[key]) / b[key]
                ok = e[key] <= (1 - 2 * n) * b[key]
                concl.append(f"- **{title}, {lab}, {name}:** {f(e[key])} so với base {f(b[key])} cm ({f(d, 1)} %) → "
                             f"{'**E1c tốt hơn base**' if ok else ('trong nhiễu' if e[key] <= b[key] * (1 + 2 * n) else 'tệ hơn base (ngoài nhiễu)')}.")
            jit = e["rpe_d1_cm"] <= 1.1 * b["rpe_d1_cm"]
            concl.append(f"  - RPE Δ1 {f(e['rpe_d1_cm'], 3)} so với {f(b['rpe_d1_cm'], 3)} cm → {'không làm giật hơn' if jit else '**giật hơn**'}.")
        r2, r1 = REF_2DGS[s]
        L.append(f"*Tham chiếu (khác giao thức):* 2DGS-SLAM (ATE mọi frame, bước 2, loop tắt, 1 seed): base {f(r2)}, E1c {f(r1)} cm. "
                 f"Số công bố: {REF_PUB[s]} cm (\"notex_far/near\" trong các paper nhiều khả năng là structure_notexture_far/near).\n")
    # "similar to 2DGS-SLAM": >= 50 % reduction of all-frame ATE on both sequences
    sim = {}
    for k, lab in B_METH[1:]:
        red = []
        for s, _ in B_SEQ:
            b, e = met.get((s, "base")), met.get((s, k))
            red.append(None if not b or not e else 100 * (b["ate_all_cm"] - e["ate_all_cm"]) / b["ate_all_cm"])
        sim[k] = red
        if all(r is not None for r in red):
            ok = all(r >= 50 for r in red)
            concl.append(f"- **\"Tương tự 2DGS-SLAM\" ({lab}):** giảm ATE mọi frame {f(red[0], 1)} % (snt_far) và {f(red[1], 1)} % (snt_near); "
                         f"cần ≥ 50 % trên cả hai → {'**đạt**' if ok else '**không đạt**'} (2DGS-SLAM: −74 % / −84 %).")
    # e1c vs e1c_s
    for s, title in B_SEQ:
        a, c = met.get((s, "e1c")), met.get((s, "e1c_s"))
        if a and c:
            d = 100 * (c["ate_kf_cm"] - a["ate_kf_cm"]) / a["ate_kf_cm"]
            ok = c["ate_kf_cm"] <= (1 - 2 * n) * a["ate_kf_cm"]
            ok2 = a["ate_kf_cm"] <= (1 - 2 * n) * c["ate_kf_cm"]
            concl.append(f"- **{title}, e1c_s so với e1c (ATE keyframe):** {f(c['ate_kf_cm'])} so với {f(a['ate_kf_cm'])} cm ({f(d, 1)} %) → "
                         + ("e1c_s tốt hơn" if ok else ("e1c tốt hơn" if ok2 else "trong nhiễu")) + ".")
    return met, n, "\n".join(L) + "\n", concl


def figures(met_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    for s, title in B_SEQ:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        any_ = False
        for k, lab in B_METH:
            m = met_b.get((s, k))
            if m:
                ax.plot(m["unaligned_cm"], label=lab)
                any_ = True
        if not any_:
            plt.close(fig)
            continue
        ax.set_title(f"{title}: sai số không căn theo frame")
        ax.set_xlabel("frame")
        ax.set_ylabel("cm")
        ax.legend()
        fig.tight_layout()
        p = f"unaligned_{s}.png"
        fig.savefig(os.path.join(D, p), dpi=110)
        plt.close(fig)
        out.append(p)
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        for k, lab in B_METH[1:]:
            m = met_b.get((s, k))
            if m and m["plane"] and m["plane"].get("anchors_per_kf"):
                a = np.array(m["plane"]["anchors_per_kf"])
                ax.plot(a[:, 0], a[:, 1], label=f"{lab}: tổng")
                ax.plot(a[:, 0], a[:, 2], "--", label=f"{lab}: active")
        ax.set_title(f"{title}: số anchor theo keyframe")
        ax.set_xlabel("frame của keyframe")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = f"anchors_{s}.png"
        fig.savefig(os.path.join(D, p), dpi=110)
        plt.close(fig)
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "full"], required=True)
    a = ap.parse_args()
    met_a, pair, n_rel, txt_a = part_a()
    if a.stage == "A":
        doc = ["# MonoGS-E1c — cài đặt và Phần A (mốc P1)\n", "Sinh từ dữ liệu (`tools/monogs/make_report.py --stage A`).\n",
               "## Môi trường\n", env_table(), note("notes_setup.md"), txt_a, note("notes_determinism.md")]
        open(os.path.join(D, "PART_A.md"), "w").write("\n".join(doc) + "\n")
        print("written docs/monogs/PART_A.md")
        return
    met_b, n, txt_b, concl = part_b(n_rel)
    figs = figures(met_b)
    allm = {f"A/{k}": {kk: vv for kk, vv in v.items() if kk != "unaligned_cm"} for k, v in met_a.items()}
    allm.update({f"B/{s}/{k}": {kk: vv for kk, vv in v.items() if kk != "unaligned_cm"} for (s, k), v in met_b.items()})
    json.dump(allm, open(os.path.join(D, "metrics_all.json"), "w"), indent=1)
    r1 = met_a.get("sp_r1")
    sumA = (f"- **Phần A:** sp_r1 ATE keyframe {f(r1['ate_kf_cm'])} cm, r = {f(r1['ate_kf_cm'] / PAPER_FR1_DESK)} so với paper "
            f"{f(PAPER_FR1_DESK)} cm → **{verdict_r(r1['ate_kf_cm'] / PAPER_FR1_DESK)}**." if r1 else "- **Phần A:** không có sp_r1.")
    if met_a.get("mt_r1"):
        m = met_a["mt_r1"]
        sumA += f" mt_r1 (song song): {f(m['ate_kf_cm'])} cm, r = {f(m['ate_kf_cm'] / PAPER_FR1_DESK)}."
    if pair:
        sumA += (f" Chạy lại cùng seed: keyframe {'trùng' if pair['kf_identical'] else 'khác'}, ATE {f(pair['ate_kf_cm'][0])} / "
                 f"{f(pair['ate_kf_cm'][1])} cm, mức nhiễu {f(100 * n_rel, 1)} %.")
    doc = ["# REPORT MonoGS-E1c — tái lập MonoGS trên fr1/desk, E1c trên MonoGS (snt_far, snt_near)\n",
           "Sinh từ dữ liệu (`tools/monogs/make_report.py --stage full`). Nhánh `exp/monogs-e1c`. 1 seed mỗi run.\n",
           "## 1. Tóm tắt\n", sumA, *concl, "", note("notes_summary.md"),
           "## 2. Cài đặt\n", env_table(), note("notes_setup.md"),
           "## 3. Phần A\n", txt_a, note("notes_determinism.md"),
           "## 4. Test và smoke\n", note("TESTS.md"),
           "## 5. Phần B\n", txt_b,
           "## 6. Hình\n", *[f"![{p}]({p})\n" for p in figs],
           "## 7. Diễn giải\n", note("notes_interpretation.md"),
           "## 8. Khác plan\n", note("notes_deviations.md")]
    open(os.path.join(D, "REPORT_MONOGS_E1C.md"), "w").write("\n".join(doc) + "\n")
    print("written docs/monogs/REPORT_MONOGS_E1C.md")


if __name__ == "__main__":
    main()
