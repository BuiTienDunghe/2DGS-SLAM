"""PLAN MONOGS-PW §10 (this round): docs/monogs/REPORT_MONOGS_PW_R1.md, docs/monogs/pw/metrics_pw.json, figures.
usage (repo root, env monogs): python tools/monogs/pw/make_report_pw.py
Free text (interpretation, deviations) comes from docs/monogs/pw/notes_*.md when present.
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "monogs"))
from eval_pw import gt_obs  # noqa: E402
from eval_runs import jsonl, load_tum  # noqa: E402

D = "docs/monogs/pw"
B, P = "results_monogs/B", "results_monogs/PW"
SPEC = {
    "snt_far": dict(max_frame=None, title="fr3/structure_notexture_far (đủ chuỗi)",
                    runs=dict(base=f"{B}/snt_far/base", e1c=f"{B}/snt_far/e1c", full=f"{P}/snt_far/full")),
    "st_far": dict(max_frame=500, title="fr3/structure_texture_far (frame 0–500)",
                   runs=dict(base=f"{P}/st_far/base", e1c=f"{P}/st_far/e1c", full=f"{P}/st_far/full", full_nob=f"{P}/st_far/full_nob")),
    "snt_near": dict(max_frame=700, title="fr3/structure_notexture_near (frame 0–700)",
                     runs=dict(base=f"{B}/snt_near/base", e1c=f"{B}/snt_near/e1c", full=f"{P}/snt_near/full")),
}
AX = ["tx", "ty", "tz", "rx", "ry", "rz"]


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{nd}f}".replace(".", ",")


def note(n):
    p = os.path.join(D, n)
    return open(p).read() if os.path.exists(p) else ""


def ratio_tag(a, b):
    if a is None or b is None:
        return "—"
    r = a / b
    return "tốt hơn" if r <= 0.7 else ("tệ hơn" if r >= 1.3 else "trong nhiễu")


def get(m, *keys):
    for k in keys:
        if m is None:
            return None
        m = m.get(k) if isinstance(m, dict) else None
    return m


def main():
    spec_json = {s: dict(max_frame=v["max_frame"], runs=v["runs"]) for s, v in SPEC.items()}
    json.dump(spec_json, open("/tmp/pw_spec.json", "w"))
    subprocess.run([sys.executable, "tools/monogs/pw/eval_pw.py", "--spec", "/tmp/pw_spec.json", "--out",
                    os.path.join(D, "metrics_pw.json")], check=True)
    M = json.load(open(os.path.join(D, "metrics_pw.json")))
    cal = json.load(open(os.path.join(D, "calib.json")))
    have = {s: [k for k, v in M[s].items() if "missing" not in v] for s in M}
    L = ["# REPORT MonoGS-PW, lượt 1 (R1)\n",
         "Sinh bằng `tools/monogs/pw/make_report_pw.py`. Nhánh `exp/monogs-pw`. 1 seed (0) mỗi run; mức nhiễu n = 15 %: "
         "A **tốt hơn** B khi A ≤ 0,7·B, **tệ hơn** khi A ≥ 1,3·B, còn lại trong nhiễu. Run base và e1c của snt_far, snt_near lấy "
         "từ `results_monogs/B` (lượt MonoGS-E1c); snt_near được cắt về frame 0–700 (`--max-frame`). Với run đủ chuỗi bị cắt, "
         "khoảng 3 keyframe cuối trước frame cắt có thể còn được local BA chỉnh sau đó.\n"]
    # ---------------------------------------------------------------- 1. answers
    ans = []
    sf, st, sn = M.get("snt_far", {}), M.get("st_far", {}), M.get("snt_near", {})

    def ate(m):
        return get(m, "ate_kf_cm")

    def stable(m, e):
        if not m or not e or "plane" not in m or not m.get("plane") or not e.get("plane"):
            return None
        dup = m["plane"].get("near_dup_pairs", 0) <= 0.5 * e["plane"].get("near_dup_pairs", 0)
        sw_m = m["plane"].get("id_switch_dup", 0) + m["plane"].get("id_switch_suspect", 0)
        sw_e = e["plane"].get("id_switch_dup", 0) + e["plane"].get("id_switch_suspect", 0)
        en = (get(m, "pw", "pct_enough") or 0) >= 0.9 * (get(e, "pw", "pct_enough") or 0)
        return dup and sw_m <= 0.5 * sw_e and en, dict(dup=(m["plane"].get("near_dup_pairs"), e["plane"].get("near_dup_pairs")),
                                                       sw=(sw_m, sw_e), enough=(get(m, "pw", "pct_enough"), get(e, "pw", "pct_enough")))

    for s, tag in (("snt_far", "Run 1"), ("snt_near", "Run 6")):
        m, e, b = M[s].get("full"), M[s].get("e1c"), M[s].get("base")
        if m and "missing" not in m:
            stb = stable(m, e)
            ans.append(f"- **{tag} ({SPEC[s]['title']}):** ATE keyframe `full` {f(ate(m))} cm, `e1c` {f(ate(e))} cm, `base` {f(ate(b))} cm → "
                       f"full so với e1c: **{ratio_tag(ate(m), ate(e))}**. Plane world ổn định hơn E1c (8.6): "
                       f"**{'có' if stb and stb[0] else 'không'}** (cặp gần trùng {stb[1]['dup'][0]} / {stb[1]['dup'][1]}, đổi ID {stb[1]['sw'][0]} / {stb[1]['sw'][1]}, "
                       f"\"đủ\" {f(stb[1]['enough'][0], 1)} / {f(stb[1]['enough'][1], 1)} %). Closure: tier 1 {get(m, 'pw', 'closure_t1')}, tier 2 "
                       f"{get(m, 'pw', 'closure_t2')}; PGO áp {get(m, 'pw', 'pgo', 'applied')}, hủy {get(m, 'pw', 'pgo', 'aborted')}.")
        else:
            ans.append(f"- **{tag}:** không có run `full` hoàn chỉnh.")
    if st.get("full") and "missing" not in st["full"]:
        m, e, b, nb = st["full"], st.get("e1c"), st.get("base"), st.get("full_nob")
        ans.append(f"- **Run 2–4 (st_far 0–500):** ATE keyframe full {f(ate(m))}, e1c {f(ate(e))}, base {f(ate(b))} cm → full so với e1c: "
                   f"**{ratio_tag(ate(m), ate(e))}**; RPE Δ1 full {f(get(m, 'rpe_d1_cm'), 3)}, e1c {f(get(e, 'rpe_d1_cm'), 3)}, base "
                   f"{f(get(b, 'rpe_d1_cm'), 3)} cm → full {'không giật hơn base' if get(m, 'rpe_d1_cm') and get(b, 'rpe_d1_cm') and get(m, 'rpe_d1_cm') <= 1.1 * get(b, 'rpe_d1_cm') else 'giật hơn base'} (ngưỡng 1,1×).")
        if nb and "missing" not in nb:
            rf, rn, rb = get(m, "rpe_d1_cm"), get(nb, "rpe_d1_cm"), get(b, "rpe_d1_cm")
            ok = (rf <= 0.7 * rn) or (rf <= 1.1 * rb and rn > 1.1 * rb)
            ans.append(f"- **Run 5 (1b giảm giật?):** RPE Δ1 full {f(rf, 3)}, full_nob {f(rn, 3)}, base {f(rb, 3)} cm → **{'có' if ok else 'không'}** "
                       f"(quy tắc: full ≤ 0,7 × full_nob, hoặc full ≤ 1,1 × base trong khi full_nob > 1,1 × base). ATE keyframe full_nob {f(ate(nb))} cm.")
    # method better than E1c (this round)
    tags = [ratio_tag(ate(M[s].get("full")), ate(M[s].get("e1c"))) for s in M if M[s].get("full") and "missing" not in M[s]["full"]]
    better = "tốt hơn" in tags and "tệ hơn" not in tags
    ans.append(f"- **Method mới hơn E1c (lượt này):** {', '.join(f'{s}: {t}' for s, t in zip([s for s in M if M[s].get('full') and 'missing' not in M[s]['full']], tags))} → "
               f"**{'đạt' if better else 'không đạt'}** (cần tốt hơn ở ≥ 1 chuỗi và không tệ hơn ở chuỗi nào).")
    # share direction
    shs = {s: get(M[s].get("full"), "pw", "stiff", "share_all_med") for s in M}
    if shs.get("st_far") is not None and (shs.get("snt_far") is not None or shs.get("snt_near") is not None):
        snt = [v for k, v in shs.items() if k != "st_far" and v is not None]
        ans.append(f"- **Phần chia đúng hướng:** trung vị s_k (trục plane quan sát được, tại keyframe) snt: {', '.join(f(v, 3) for v in snt)}; "
                   f"st_far: {f(shs['st_far'], 3)} → **{'đúng' if min(snt) > shs['st_far'] else 'không đúng'}** (cần snt > st_far).")
    # kernel useful + drift (GT)
    for s in M:
        g = get(M[s].get("full"), "gt")
        if g and "kernel_wrong_in_accepted" in g and g.get("kernel_wrong_in_all"):
            ok = g["kernel_wrong_in_accepted"] <= 0.5 * g["kernel_wrong_in_all"]
            ans.append(f"- **Kernel bền có ích ({s}, nhãn GT):** sai nhãn trong cặp được nhận {f(g['kernel_wrong_in_accepted'], 1)} %, trong mọi cặp ứng viên "
                       f"{f(g['kernel_wrong_in_all'], 1)} %, trong cặp bị loại {f(g.get('kernel_wrong_in_rejected'), 1)} % → **{'có' if ok else 'không'}** (cần ≤ 0,5×).")
        if g and g.get("change_ang_med_deg") is not None:
            ok = g["change_ang_med_deg"] <= 0.5 and g["change_d_med_cm"] <= 0.5
            ans.append(f"  - Không trôi ngoài PGO ({s}): trung vị {f(g['change_ang_med_deg'], 3)}° / {f(g['change_d_med_cm'], 3)} cm → **{'đạt' if ok else 'không đạt'}**.")
    L += ["## 1. Trả lời ngắn (theo mục 9)\n"] + ans + ["", note("notes_pw_summary.md")]
    # ---------------------------------------------------------------- 2. calibration
    n = cal["noise"]
    k = cal["kappa"]
    L += ["## 2. Hiệu chỉnh (mục 4, chi tiết ở `pw/CALIB_PW.md`)\n", "| Đại lượng | Giá trị |", "|---|---|",
          f"| σ_n | {f(np.degrees(n['sigma_n']), 3)}° |", f"| σ_d(z) | {f(100 * n['sigma_d_a'], 3)} + {f(100 * n['sigma_d_b'], 4)}·z² cm |",
          f"| θ_m / δ_m | {f(n['merge_deg'])}° / {f(100 * n['merge_m'])} cm |",
          f"| σ_odo | {f(100 * n['sigma_odo_t'])} cm / {f(np.degrees(n['sigma_odo_r']))}° |",
          f"| κ chung (dịch / quay) | {f(n['kappa'], 1)} ({f(k['kappa_t'], 1)} / {f(k['kappa_r'], 1)}) |"]
    for r, v in k["per_run"].items():
        L.append(f"| κ {r} (dịch / quay) | {f(v['all'], 1)} ({f(v['t'], 1)} / {f(v['r'], 1)}) |")
    L += ["", "κ của hai chuỗi validation chênh nhau hơn 40 lần (ngưỡng ghi nhận của plan là 3 lần). Theo plan, giữ một số chung, không đổi.\n"]
    # ---------------------------------------------------------------- 3. tests
    L += ["## 3. Test\n", "Xem `pw/TESTS_PW.md`: T1, T2, T3, T4, T6, T7 đạt; T5 trượt (2 mặt không xác định được dịch dọc giao tuyến).\n"]
    # ---------------------------------------------------------------- 4. tables 8.6
    L.append("## 4. Bảng mỗi chuỗi (mục 8.6)\n")
    for s, sv in SPEC.items():
        L.append(f"### {sv['title']}\n")
        L.append("| Method | ATE keyframe | ATE mọi frame | ATE mọi frame hiệu chỉnh | RPE Δ1 | Sai số chưa căn cuối / lớn nhất | Keyframe (mono/T1/T2) | ms/frame tracking | ms/frame plane | s tổng |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for meth in sv["runs"]:
            m = M[s].get(meth)
            if not m or "missing" in m:
                L.append(f"| {meth} | không có run |" + " |" * 8)
                continue
            kr = get(m, "pw", "kf_reasons") or {}
            kfs = f"{m['n_kf']}" + (f" ({kr.get('mono', 0)}/{kr.get('T1', 0)}/{kr.get('T2', 0)})" if kr else "")
            L.append(f"| {meth} | {f(m['ate_kf_cm'])} | {f(m['ate_all_cm'])} | {f(get(m, 'pw', 'ate_all_corr_cm'))} | {f(m['rpe_d1_cm'], 3)} | "
                     f"{f(m['unaligned_end_cm'], 1)} / {f(m['unaligned_max_cm'], 1)} | {kfs} | {f(m['ms_track'], 0)} | {f(m['ms_plane'], 1)} | {f(m['total_s'], 0)} |")
        L.append("")
        L.append("| Method | Anchor tổng / active cuối | Cặp gần trùng | Đổi ID (bản trùng / nghi ghép sai) | % khởi tạo bằng plane | \"Đủ\" (%) | % cặp bị kernel loại | w TB | Cặp bị bỏ ở khởi tạo bền | Ứng viên tạo / xóa / kích hoạt | Gộp | ms đo độ cứng |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for meth in sv["runs"]:
            m = M[s].get(meth)
            if not m or "missing" in m or not m.get("plane"):
                continue
            p, w = m["plane"], m.get("pw", {})
            L.append(f"| {meth} | {p.get('anchors_total')} / {p.get('anchors_active')} | {p.get('near_dup_pairs')} | {p.get('id_switch_dup')} / {p.get('id_switch_suspect')} | "
                     f"{f(p.get('pct_init_plane'), 1)} | {f(w.get('pct_enough'), 1)} | {f(w.get('pct_rejected'), 1)} | {f(w.get('mean_w'), 3)} | "
                     f"{w.get('init_dropped', '—')} | {w.get('anchors_created', '—')} / {w.get('anchors_deleted', '—')} / {w.get('anchors_activated', '—')} | "
                     f"{w.get('merges', '—')} | {f(get(w, 'stiff', 'ms_mean'), 0)} |")
        L.append("")
        for meth in sv["runs"]:
            eb = get(M[s].get(meth), "pw", "e_by_depth")
            if eb:
                L.append(f"*{meth}: e theo độ sâu:* " + "; ".join(f"z {k}: trung vị e {f(v['median_e'])}, loại {f(v['pct_rejected'], 1)} % ({v['n']} cặp)" for k, v in eb.items()) + "\n")
    # ---------------------------------------------------------------- 5. 1b
    L += ["## 5. 1b (mục 8.5)\n", "Trung vị (phân vị 25–75 %) tại keyframe; s_k chỉ ở trục có c_plane > 0.\n",
          "| Chuỗi / method | " + " | ".join(f"c_render {a}" for a in AX) + " | " + " | ".join(f"s_k {a}" for a in AX) + " | RPE Δ1 |",
          "|---|" + "---|" * 13]
    for s in M:
        for meth in ("full_nob", "full", "base"):
            m = M[s].get(meth)
            stf = get(m, "pw", "stiff")
            if not stf:
                continue
            cr = " | ".join(f"{f(a, 0)} ({f(b_, 0)}–{f(c, 0)})" for a, b_, c in zip(stf["c_render_med"], stf["c_render_q25"], stf["c_render_q75"]))
            sk = " | ".join(f"{f(a, 3)} ({f(b_, 3)}–{f(c, 3)})" if a is not None else "—" for a, b_, c in
                            zip(stf.get("share_med", [None] * 6), stf.get("share_q25", [None] * 6), stf.get("share_q75", [None] * 6)))
            L.append(f"| {s} / {meth} | {cr} | {sk} | {f(m['rpe_d1_cm'], 3)} |")
    L.append("")
    # ---------------------------------------------------------------- 6. suggestions from logs (data part)
    L += ["## 6. Gợi ý điều chỉnh (rút từ log)\n"]
    for s in M:
        m = M[s].get("full")
        if not m or "missing" in m:
            continue
        run = SPEC[s]["runs"]["full"]
        fr = [r for r in jsonl(os.path.join(run, "pw_frames.jsonl")) if SPEC[s]["max_frame"] is None or r["frame"] < SPEC[s]["max_frame"]]
        _, est = load_tum(os.path.join(run, "traj_all.txt"))
        _, gt = load_tum(os.path.join(run, "traj_gt_all.txt"))
        W = [np.linalg.inv(T) for T in est]
        G = [np.linalg.inv(T) for T in gt]
        rpe = {t: float(np.linalg.norm(((W[t] @ np.linalg.inv(W[t - 1])) @ np.linalg.inv(G[t] @ np.linalg.inv(G[t - 1])))[:3, 3]))
               for t in range(1, len(W))}
        top = sorted(rpe, key=lambda t: -rpe[t])[:3]
        byf = {r["frame"]: r for r in fr}
        items = []
        for t in top:
            r = byf.get(t, {})
            rej = sum(1 for p in r.get("planes", []) if p["anchor"] is not None and not p["inlier"])
            items.append(f"frame {t}: RPE {f(100 * rpe[t], 2)} cm, khởi tạo `{r.get('init_source')}`, kernel loại {rej} cặp, "
                         f"{r.get('n_active_pairs')} cặp active, keyframe {'có (' + str(r.get('kf_reason')) + ')' if r.get('is_kf') else 'không'}")
        w = m.get("pw", {})
        L.append(f"- **{s}:** kernel loại {f(w.get('pct_rejected'), 1)} % cặp ứng viên active; T1 sinh {get(w, 'kf_reasons', 'T1') or 0} keyframe, T2 "
                 f"{get(w, 'kf_reasons', 'T2') or 0}; ứng viên tạo {w.get('anchors_created')}, xóa {w.get('anchors_deleted')}, kích hoạt {w.get('anchors_activated')}; "
                 f"trung vị s_k {f(get(w, 'stiff', 'share_all_med'), 3)}. Ba frame giật nhất: " + "; ".join(items) + ".")
    L += ["", note("notes_pw_suggestions.md")]
    # ---------------------------------------------------------------- 7. GT metrics
    L += ["## 7. Chỉ số theo nhãn GT (mục 8.2)\n",
          "| Chuỗi / method | Mặt thật | Anchor / mặt thật | Bản trùng | % ghép nhầm | Kernel: % bị loại / sai nhãn trong bị loại / trong được nhận | Đổi ID đồng thời / lật | Thay đổi ngoài PGO: trung vị (lớn nhất) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in M:
        for meth in ("e1c", "full", "full_nob"):
            g = get(M[s].get(meth), "gt")
            if not g or "error" in g:
                continue
            kern = (f"{f(g.get('kernel_pct_rejected'), 1)} / {f(g.get('kernel_wrong_in_rejected'), 1)} / {f(g.get('kernel_wrong_in_accepted'), 1)}"
                    if "kernel_pct_rejected" in g else "—")
            L.append(f"| {s} / {meth} | {g['n_true_planes']} | {f(g.get('anchors_per_true_plane'))} | {g['duplicates']} | {f(g.get('pct_mismatch'), 1)} | {kern} | "
                     f"{g['id_simultaneous']} / {g['id_flip']} | {f(g.get('change_ang_med_deg'), 3)}° / {f(g.get('change_d_med_cm'), 3)} cm "
                     f"({f(g.get('change_ang_max_deg'), 2)}° / {f(g.get('change_d_max_cm'), 2)} cm) |")
    L.append("")
    # ---------------------------------------------------------------- 8. diag
    L += ["## 8. Chẩn đoán E1c (mục 5)\n", note("DIAG_E1C.md").split("\n", 2)[-1] if note("DIAG_E1C.md") else "(chưa chạy)\n"]
    # ---------------------------------------------------------------- 9. part 4
    L += ["## 9. Phần 4: closure và PGO\n", "| Chuỗi | Ứng viên tier 1 / tier 2 | PGO áp / hủy / bỏ qua | Factor bị bỏ | Closure được nhận | ms mỗi PGO | Dịch keyframe lớn nhất (cm) |", "|---|---|---|---|---|---|---|"]
    for s in M:
        for meth in ("full", "full_nob"):
            w = get(M[s].get(meth), "pw")
            if not w or "pgo" not in w:
                continue
            pg = w["pgo"]
            L.append(f"| {s} / {meth} | {w.get('closure_t1', 0)} / {w.get('closure_t2', 0)} | {pg['applied']} / {pg['aborted']} / {pg['skipped']} | {pg['dropped']} | "
                     f"{pg['closure_accepted']} | {f(pg['ms_mean'], 0)} | {f(pg['max_shift_cm'])} |")
    L += ["", "Smoke ép PGO và test T3–T6: xem `pw/TESTS_PW.md`.\n"]
    # ---------------------------------------------------------------- 10, 11
    missing = [f"{s}/{k}" for s in M for k, v in M[s].items() if "missing" in v]
    L += ["## 10. Run bị bỏ\n", ("Không có." if not missing else "Thiếu: " + ", ".join(missing)) + "\n",
          note("notes_pw_runs.md"), "## 11. Khác plan\n", note("notes_pw_deviations.md")]
    # figure: unaligned error per frame
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from eval_runs import run_metrics
        figs = []
        for s, sv in SPEC.items():
            fig, ax = plt.subplots(figsize=(7.5, 3.3))
            for meth, run in sv["runs"].items():
                if os.path.exists(os.path.join(run, "traj_all.txt")):
                    ax.plot(run_metrics(run, sv["max_frame"])["unaligned_cm"], label=meth)
            ax.set_title(f"{sv['title']}: sai số chưa căn theo frame")
            ax.set_xlabel("frame")
            ax.set_ylabel("cm")
            ax.legend()
            fig.tight_layout()
            p = f"pw_unaligned_{s}.png"
            fig.savefig(os.path.join(D, p), dpi=110)
            plt.close(fig)
            figs.append(p)
        L += ["## Hình\n"] + [f"![{p}](pw/{p})\n" for p in figs]
    except Exception as e:  # noqa: BLE001
        L.append(f"(không vẽ được hình: {e})\n")
    open("docs/monogs/REPORT_MONOGS_PW_R1.md", "w").write("\n".join(L) + "\n")
    print("written docs/monogs/REPORT_MONOGS_PW_R1.md")


if __name__ == "__main__":
    main()
