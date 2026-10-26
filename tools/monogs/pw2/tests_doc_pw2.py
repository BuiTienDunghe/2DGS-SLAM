"""PLAN MONOGS-PW2 §5: docs/monogs/pw2/TESTS_PW2.md from tests_pw2.json (T8-T11), the T12 run, the R1 reproducibility
run and the pw2 smoke. usage (repo root, env monogs): python tools/monogs/pw2/tests_doc_pw2.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "tools", "monogs"))
from eval_runs import jsonl  # noqa: E402

D = "docs/monogs/pw2"
M = "results_monogs/PW2"


def f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "có" if x else "không"
    if isinstance(x, (int, np.integer)):
        return str(x)
    try:
        return f"{float(x):.{nd}g}".replace(".", ",")
    except (TypeError, ValueError):
        return str(x)


def main():
    t = json.load(open(os.path.join(D, "tests_pw2.json")))
    L = ["# TESTS_PW2 — test và smoke (PLAN MONOGS-PW2 §5)\n",
         "Sinh bằng `tools/monogs/pw2/tests_doc_pw2.py`. T8–T11 chạy bằng `python tests/test_pw2.py` (trong `monogs/`) với bộ nhiễu "
         "đã khóa trong `pw2_calib.yaml`.\n", "| Test | Kết quả | Số liệu chính |", "|---|---|---|"]
    r = t["T8"]
    L.append(f"| T8 hiệu chỉnh | {'đạt' if r.get('passed') else 'trượt'} | (a) σ ước lượng {f(r.get('a_sigma_est_deg'))}° / thật 0,5°; bộ ước lượng R1 cho {f(r.get('a_r1_estimator_deg'))}° "
             f"({f(r.get('a_r1_ratio'))}σ); (b) b trung vị {f(r.get('b_median'))} / thật {f(r.get('b_true'))} |")
    r = t["T9"]
    L.append(f"| T9 ghép `e_first` | {'đạt' if r.get('passed') else 'trượt'} | (a) chọn ứng viên {r['a'].get('pairs')} (e active {f(r['a'].get('e_active'))}, e ứng viên {f(r['a'].get('e_cand'))}); "
             f"(b) ứng viên thành active, số anchor {r['b'].get('n_anchors')}; (c) cả hai inlier → chọn active {r['c'].get('pairs')}; (d) chế độ `cost` giống R1 trên 10 frame thật: {f(r['d'].get('identical'))} |"
             if "a" in r else f"| T9 | trượt | {r.get('error')} |")
    r = t["T10"]
    L.append(f"| T10 cơ sở | {'đạt' if r.get('passed') else 'trượt'} | (a) không cặp inlier: lệch so với Adam của base {f(r['a'].get('max_diff'))} (100 bước, gradient ngẫu nhiên); "
             f"(b) cơ sở dựng từ {r['b'].get('n_inlier')} cặp inlier, trùng cơ sở E1c của đúng các cặp đó |" if "a" in r else f"| T10 | trượt | {r.get('error')} |")
    r = t["T11"]
    L.append(f"| T11 PGO | {'đạt' if r.get('passed') else 'trượt'} | (a) residual vector hóa lệch {f(r['a'].get('max_diff'))}; (b) T4 R1: {f(r['b'].get('err_before_cm'))} → "
             f"{f(r['b'].get('err_after_cm'))} cm; (c) factor sai: w IRLS {f(r['c'].get('irls_w'))}, bị bỏ {r['c'].get('dropped')}; (d) odometry không đổi trọng số: "
             f"{f(r['d'].get('passed'))}; (e) {r['e'].get('n_kf')} keyframe, {r['e'].get('n_plane')} factor: {f(r['e'].get('seconds'))} s |" if "a" in r else f"| T11 | trượt | {r.get('error')} |")
    # T12
    run = f"{M}/tests/t12_base_kf11"
    if os.path.exists(f"{run}/kf_indices.txt"):
        kf = [int(x) for x in open(f"{run}/kf_indices.txt").read().split()]
        gaps = np.diff(kf)
        L.append(f"| T12 `kf_force_every` | {'đạt' if gaps.max() <= 11 else 'trượt'} | base, N = 11, snt_far 0–200: {len(kf)} keyframe, khoảng cách lớn nhất {int(gaps.max())} frame |")
    else:
        L.append("| T12 | trượt | không có run |")
    # R1 reproducibility
    run = f"{M}/tests/r1_full_r100"
    if os.path.exists(f"{run}/plot/stats_final.json"):
        a = 100 * json.load(open(f"{run}/plot/stats_final.json"))["rmse"]
        L.append(f"| R1 tái lập | {'đạt' if abs(a - 0.998) / 0.998 <= 0.15 else 'trượt'} | `full` snt_far 0–100: ATE keyframe {f(a)} cm, smoke 1 của R1: 0,998 cm "
                 f"(lệch {f(100 * abs(a - 0.998) / 0.998, 2)} %) |")
    else:
        L.append("| R1 tái lập | trượt | không có run |")
    L.append("")
    # smoke pw2
    run = f"{M}/smoke/pw2_r200"
    L.append("## Smoke `pw2` (snt_far, `--range 0 200`)\n")
    if os.path.exists(f"{run}/traj_all.txt"):
        fr = jsonl(f"{run}/pw_frames.jsonl")
        tim = json.load(open(f"{run}/timing.json"))
        ev = collections.Counter(e["type"] for e in jsonl(f"{run}/events.jsonl"))
        cand = [p for r_ in fr if r_["init_source"] != "gt" for p in r_["planes"] if p["anchor"] is not None and p["e"] is not None]
        e1 = [r_ for r_ in jsonl(f"{run}/e1c_frames.jsonl") if r_["init_source"] != "gt"]
        enough = 100 * np.mean([(r_["n_active_pairs"] >= 3) and ((r_["sigma_min"] or 0) >= 0.3) for r_ in e1]) if e1 else None
        need = ["pgo_poses.jsonl", "traj_all_corr.txt", "pw_frames.jsonl", "stiffness.jsonl", "pgo.jsonl"]
        miss = [n for n in need if not os.path.exists(f"{run}/{n}")]
        st = json.load(open(f"{run}/plot/stats_final.json"))
        L += ["| Chỉ số | Giá trị |", "|---|---|",
              f"| Chạy hết | có ({tim['n_frames']} frame, {tim['total_s']:.0f} s) |",
              f"| NaN | {f('nan' in open(f'{run}/traj_all.txt').read().lower())} |",
              f"| Log thiếu | {', '.join(miss) if miss else 'không'} (pgo_poses.jsonl chỉ có dòng khi có PGO) |",
              f"| % cặp bị kernel loại | {f(100 * np.mean([not p['inlier'] for p in cand]) if cand else None)} % |",
              f"| Ứng viên tạo / xóa / kích hoạt | {ev.get('create', 0)} / {ev.get('delete', 0)} / {ev.get('activate', 0)} |",
              f"| \"Đủ\" | {f(enough)} % |",
              f"| Keyframe theo lý do | {dict(collections.Counter(r_['kf_reason'] for r_ in fr if r_['is_kf']))} |",
              f"| ATE keyframe | {f(100 * st['rmse'])} cm |", ""]
    else:
        L.append("Không có kết quả smoke.\n")
    L.append(open(os.path.join(D, "notes_pw2_tests.md")).read() if os.path.exists(os.path.join(D, "notes_pw2_tests.md")) else "")
    open(os.path.join(D, "TESTS_PW2.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
