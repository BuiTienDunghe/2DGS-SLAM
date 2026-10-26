"""PLAN MONOGS-PW §6: docs/monogs/pw/TESTS_PW.md from tests_pw.json and the two smoke runs (§7.2).
usage (repo root, env monogs): python tools/monogs/pw/tests_doc_pw.py
"""
import collections
import json
import os

import numpy as np

D = "docs/monogs/pw"
SMOKES = [("results_monogs/PW/smoke/full_r100", "Smoke 1: `full`, snt_far, `--range 0 100`"),
          ("results_monogs/PW/smoke/full_forcepgo_r200", "Smoke 2: `full`, snt_far, `--range 0 200`, ép PGO mỗi 5 keyframe")]


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


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def smoke_rows(run, title):
    if not os.path.exists(os.path.join(run, "traj_all.txt")):
        return [f"**{title}:** không có kết quả.", ""]
    fr = jl(os.path.join(run, "pw_frames.jsonl"))
    st = jl(os.path.join(run, "stiffness.jsonl"))
    pg = jl(os.path.join(run, "pgo.jsonl"))
    tim = json.load(open(os.path.join(run, "timing.json")))
    nan = "nan" in open(os.path.join(run, "traj_all.txt")).read().lower()
    need = ["stiffness.jsonl", "e1c_frames.jsonl", "anchors.jsonl", "pgo.jsonl", "traj_all_corr.txt", "pw_frames.jsonl", "events.jsonl"]
    miss = [n for n in need if not os.path.exists(os.path.join(run, n))]
    reasons = collections.Counter(r["kf_reason"] for r in fr if r["is_kf"])
    act = [p for r in fr for p in r["planes"] if p["anchor"] is not None]
    sh = [s for r in st for s in (r.get("share") or []) if s is not None]
    st_ = json.load(open(os.path.join(run, "plot", "stats_final.json")))
    L = [f"**{title}**\n", "| Chỉ số | Giá trị |", "|---|---|",
         f"| Chạy hết | có ({tim['n_frames']} frame, {tim['total_s']:.0f} s, {1000 * tim['total_s'] / tim['n_frames']:.0f} ms/frame) |",
         f"| NaN trong quỹ đạo | {f(nan)} |", f"| Log thiếu | {', '.join(miss) if miss else 'không'} |",
         f"| Keyframe theo lý do | {', '.join(f'{k}: {v}' for k, v in reasons.items())} |",
         f"| % cặp ứng viên bị kernel loại | {f(100 * np.mean([not p['inlier'] for p in act]) if act else None)} % |",
         f"| Trung vị phần chia s_k | {f(np.median(sh) if sh else None)} |",
         f"| Số lần PGO | {len(pg)} ({', '.join(r['status'] for r in pg) or '—'}) |",
         f"| Độ dịch keyframe lớn nhất sau mỗi PGO (cm) | {', '.join(f(100 * (r.get('max_kf_shift_m') or 0)) for r in pg) or '—'} |",
         f"| ATE keyframe (cm) | {f(100 * st_['rmse'])} |", ""]
    big = [r for r in pg if (r.get("max_kf_shift_m") or 0) > 0.05]
    if big:
        L.append(f"**Cảnh báo:** có {len(big)} lần PGO dịch keyframe > 5 cm.\n")
    return L


def main():
    t = json.load(open(os.path.join(D, "tests_pw.json")))
    L = ["# TESTS_PW — test và smoke (PLAN MONOGS-PW §6, §7.2)\n",
         "Sinh bằng `tools/monogs/pw/tests_doc_pw.py` từ `tests_pw.json` (chạy `python tests/test_pw.py` trong `monogs/`) và log smoke. "
         "Mô hình nhiễu dùng khi test: bộ số đã khóa trong `pw_calib.yaml`.\n",
         "| Test | Kết quả | Số liệu chính |", "|---|---|---|"]
    desc = {
        "T1": lambda r: (f"(a) 2 mảnh → anchor {r['a'].get('pairs')}; (b) mặt song song 6 cm: e = {f(r['b'].get('e'))}, w = {f(r['b'].get('w'))}; "
                         f"(c) phần xa của mặt lớn: ghép, inlier; (d) khởi tạo bền bỏ {r['d'].get('n_dropped')} cặp, sai {f(100 * (r['d'].get('err_m') or 0))} cm "
                         f"(E1c: {f(100 * (r['d'].get('err_e1c_m') or 0))} cm); (e) chế độ e1c giống code cũ: ghép {f(r['e'].get('match_identical'))}, lệch loss {f(r['e'].get('max_loss_diff'))}"),
        "T2": lambda r: f"(a) sai refresh {f(r['a']['max_err'])}; (b) n_obs_kf {r['b']['n_obs_kf']}; (c) còn sống sau 0/1/2 keyframe: {r['c']['alive']}; (d) {r['d']['created']}",
        "T3": lambda r: f"(a) tier 2 = {r['a']['tier2']}; (b) hai anchor cũ trong cổng: {r['b']['tier2']}; (c) {r['c']}",
        "T4": lambda r: (f"(a) phần dư lớn nhất {f(r['a']['max_residual'])}; (b) sai số TB {f(r['b']['err_before_cm'])} → {f(r['b']['err_after_cm'])} cm; "
                         f"(c) factor bị bỏ {r['c']['dropped']}"),
        "T5": lambda r: (f"sai dịch {f(r['trans_err_cm'])} cm, sai quay {f(r['rot_err_deg'])}°; thành phần dịch thật dọc giao tuyến hai mặt "
                         f"{f(r['true_t_along_intersection_cm'])} cm"),
        "T6": lambda r: (f"(a) sai xyz {f(r['a_xyz_err'])}, sai quay {f(r['a_rot_err'])}; (b) depth lệch lớn nhất {f(1000 * r['b_depth_err_m'])} mm "
                         f"trên {r['b_valid_px']} pixel hợp lệ (p99 {f(1000 * r['b_depth_p99_m'])} mm; mọi pixel có depth: lớn nhất {f(1000 * r['b_any_depth_max_m'])} mm)"),
        "T7": lambda r: (f"(a) R, T khôi phục từng bit: {f(r['a']['passed'])}; (b) Δ_k nhỏ nhất {f(r['b']['min_delta'])}; "
                         f"(c) cân bằng {f(r['c']['x'])} / đích {f(r['c']['target'])}; (d) κ ước lượng {f(r['d']['kappa_est'])} / thật {r['d']['kappa_true']}; "
                         f"(e) loss {f(r['e']['loss'], 10)} / công thức {f(r['e']['formula'], 10)}"),
    }
    for k in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        r = t.get(k, {})
        if "error" in r:
            L.append(f"| {k} | trượt | lỗi: {r['error']} |")
            continue
        try:
            s = desc[k](r)
        except Exception as e:  # noqa: BLE001
            s = f"(không đọc được: {e})"
        L.append(f"| {k} | {'đạt' if r.get('passed') in (True, 'True') else 'trượt'} | {s} |")
    L += ["",
          "**Ghi chú:**",
          "- **T4:** lần chạy đầu dùng drift nhân trái lên w2c (drift quay theo hướng camera). Khi đó chính quỹ đạo thật có cost odometry 108, cao hơn nghiệm tối ưu 10,7, nên PGO không thể đưa về quỹ đạo thật. Test được sửa theo mô hình drift odometry chuẩn: mỗi chuyển động tương đối mang cùng một sai lệch 0,5 cm / 0,2° trong hệ thân. Solver không đổi.",
          "- **T4c, solver:** `least_squares` dừng vì chạm giới hạn số lần đánh giá (status 0), mà lúc đầu code coi đó là lỗi. Plan đặt \"tối đa 100 vòng\", nên đã sửa để chấp nhận status 0.",
          "- **T5 trượt:** với **đúng 2 mặt**, dịch dọc giao tuyến hai mặt không quan sát được, cả bằng nghiệm dạng đóng (nghiệm chuẩn nhỏ nhất) lẫn bằng ICP điểm–mặt. Sai số đo được gần bằng thành phần dịch thật dọc giao tuyến. Đây là giới hạn của closure kiểu StructureGS khi chỉ có 2 mặt; không sửa.",
          "- **T6:** lần đầu so mọi pixel có depth và lệch 1,3 mm ở viền vật. Đã sửa 1 lần: chỉ so pixel hợp lệ theo quy ước MonoGS (opacity > 0,95 ở cả hai lần render). Kèm số trên mọi pixel để đối chiếu.",
          "- **T7:** hai lỗi của chính test: thiếu `compute_grad_mask` cho camera, và sai dấu bước của `ManualAdam`. `ManualAdam` trả về chính bước cập nhật.",
          ""]
    L.append("## Smoke (mục 7.2)\n")
    for run, title in SMOKES:
        L += smoke_rows(run, title)
    open(os.path.join(D, "TESTS_PW.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
