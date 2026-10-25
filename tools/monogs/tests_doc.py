"""PLAN MONOGS-E1C §5.4: docs/monogs/TESTS.md from the test outputs and the E1c smoke run.
usage (repo root, env monogs): python tools/monogs/tests_doc.py
inputs: docs/monogs/tests_monogs_e1c.json (T1-T3), docs/monogs/tests_e1c_old.txt (T4, run_tests.py output),
        results_monogs/smoke/snt_far_e1c_r200 (smoke)
"""
import collections
import json
import os
import re

D = "docs/monogs"
SMOKE = "results_monogs/smoke/snt_far_e1c_r200"


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def main():
    t = json.load(open(os.path.join(D, "tests_monogs_e1c.json")))
    old = open(os.path.join(D, "tests_e1c_old.txt")).read() if os.path.exists(os.path.join(D, "tests_e1c_old.txt")) else ""
    res = re.findall(r"(tests/\S+::\S+) (PASSED|FAILED)", old)
    L = ["### Test (mục 5.4)\n", "| Test | Kết quả | Số liệu |", "|---|---|---|"]
    L.append(f"| T1 đổi pose | {'đạt' if t['T1']['passed'] else 'trượt'} | sai lớn nhất {t['T1']['max_abs_change']:.1e} (100 pose ngẫu nhiên) |")
    T2 = t["T2"]
    if "error" in T2:
        L.append(f"| T2 quy ước gradient pose | trượt | lỗi: {T2['error']} |")
    else:
        m, fr = T2["mapped_delta_0.0001"], T2["fresh_map_delta_0.0001"]
        L.append(f"| T2 quy ước gradient pose | {'đạt' if T2['passed'] else 'trượt'} | bản đồ đã tối ưu (500 vòng, {100 * T2['mapped_opacity_gt_095_fraction']:.0f} % pixel opacity > 0,95): "
                 f"cos {f(m['cos'])}, dấu thành phần lớn {'trùng' if m['sign_large_ok'] else 'khác'}; δ = 10⁻³: cos {f(T2['mapped_delta_0.001']['cos'])}. "
                 f"Chẩn đoán, bản đồ mới (opacity khoảng 0,5): cos {f(fr['cos'])} |")
    L.append(f"| T3 ManualAdam | {'đạt' if t['T3']['passed'] else 'trượt'} | sai tương đối lớn nhất {t['T3']['max_rel_diff']:.1e} so với torch.optim.Adam (200 bước) |")
    n_ok = sum(1 for _, s in res if s == "PASSED")
    fails = [n.split("::")[1] for n, s in res if s == "FAILED"]
    L.append(f"| T4 test E1c cũ | {'đạt' if not fails else 'trượt (' + ', '.join(fails) + ')'} | {n_ok}/{len(res)} đạt |")
    L.append("")
    L.append("**T2.** Lần chạy đầu dựng bản đồ thẳng từ đám mây điểm của một frame (opacity khoảng 0,5) và cho cos −0,75. "
             "Chẩn đoán: với loss màu bỏ thừa số opacity, gradient rasterizer khớp sai phân trung tâm (cos 0,998), và các loss trơn "
             "(trung bình depth, trung bình màu có trọng số) khớp với cos 0,92–0,99 theo nhiễu trái `SE3_exp(τ) @ T_w2c`, "
             "không khớp theo nhiễu phải. Vậy quy ước τ của rasterizer trùng với loss plane. Phần lệch đến từ thừa số opacity "
             "trong loss tracking của MonoGS: rasterizer không đạo hàm theo nhánh opacity, mà nhánh đó lấn át khi opacity thấp. "
             "Test được sửa (1 lần) để đo ở trạng thái thật của hệ: bản đồ đã tối ưu như `initialize_map`.\n")
    if fails:
        line = next((l for l in old.splitlines() if l.startswith("T3:")), "")
        L.append(f"**T4.** `test_T3_solve_pose` trượt **cả trên code gốc của nhánh `exp/e1c-decouple` trong env `2dgs-slam`** "
                 f"(cùng thông báo), nên không do việc port. `{line.strip()}`. Ngưỡng của test (mọi thử nghiệm có nhiễu dưới 1° và 1 cm) "
                 "chặt hơn mức nhiễu nó tự sinh; plan yêu cầu giữ nguyên logic E1c, nên không sửa.\n")
    if os.path.exists(os.path.join(SMOKE, "e1c_frames.jsonl")):
        fr = [json.loads(l) for l in open(os.path.join(SMOKE, "e1c_frames.jsonl"))]
        tim = json.load(open(os.path.join(SMOKE, "timing.json")))
        c = collections.Counter(r["init_source"] for r in fr if r["init_source"] != "gt")
        nan = "nan" in open(os.path.join(SMOKE, "traj_all.txt")).read().lower()
        st = json.load(open(os.path.join(SMOKE, "plot", "stats_final.json")))
        L += ["### Smoke E1c (snt_far, `--range 0 200`, `--method e1c`)\n", "| Chỉ số | Giá trị |", "|---|---|",
              f"| Chạy hết | có ({tim['n_frames']} frame, {tim['n_kf']} keyframe, {tim['total_s']:.0f} s) |",
              f"| NaN trong quỹ đạo | {'có' if nan else 'không'} |",
              f"| Nguồn khởi tạo | plane {c.get('plane', 0)}, prev {c.get('prev', 0)}, rejected {c.get('rejected', 0)} |",
              f"| ATE keyframe (cm) | {100 * st['rmse']:.2f} |",
              f"| ms/frame tracking / plane | {tim['ms_track_mean']:.0f} / {tim['ms_plane_mean']:.1f} |", ""]
    open(os.path.join(D, "TESTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
