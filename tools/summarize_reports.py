"""Print markdown tables for the phase reports from P1/P2/P3 JSON (numbers only; prose is written by hand).

usage: python tools/summarize_reports.py results_exp/reports
"""
import json
import os
import sys

import numpy as np


def mm(x, nd=2):
    return "—" if x is None else f"{1e3 * x:.{nd}f}"


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}"


def pc0(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def sgn(x):
    return "—" if x is None else f"{100 * x:+.1f}%"


def p1(root):
    p = os.path.join(root, "P1", "P1_metrics.json")
    if not os.path.exists(p):
        return
    ev = json.load(open(p))["events"]
    print("\n### P1 — theo event\n")
    print("| Scene | Event | c | h | \\|J_L\\| | \\|J_eval\\| | #px | e_dl pre (mm) | e_dl rigid p50 / p90 / inlier<1cm | e_step rigid (mm) | PSNR rigid (dB) | ATE kf rigid (cm) | Acc rigid (mm) | nhiễu TF32 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in ev:
        r, pr = e["methods"]["rigid"], e["methods"]["pre"]
        ed = r["e_dl"] or {}
        print(f'| {e["scene"]} | {e["event_id"]} | {e["cur_uid"]} | {e["loop_uid"]} | {e["n_loop_kfs"]} | {len(e["J_eval"])} | '
              f'{e["n_eval_pix"]} | {mm((pr["e_dl"] or {}).get("median"))} | {mm(ed.get("median"))} / {mm(ed.get("p90"))} / '
              f'{f(ed.get("inlier_1cm"), 2)} | {mm((r["e_step"] or {}).get("e_step"))} | {f((r["render"] or {}).get("psnr"), 2)} | '
              f'{pct(r["ate_kf"])} | {mm((r["acc"] or {}).get("acc_median"))} | '
              f'{f((e.get("tf32_floor") or {}).get("median_mm"), 3)} mm |')


def p2(root):
    p = os.path.join(root, "P2", "P2_metrics.json")
    if not os.path.exists(p):
        return
    ev = json.load(open(p))["events"]
    print("\n### P2 — cấu hình chính (tọa độ V-B)\n")
    print("| Scene | Event | #node | filler | #cạnh | độ phủ | tách lần quét (#voxel trộn) | nhiễu node (mm) | t S+node (s) | %R | S p50 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in ev:
        b, rl = e["main_B"], e["reliability"]
        print(f'| {e["scene"]} | {e["event_id"]} | {b["n_nodes"]} | {100 * b["filler_frac"]:.0f}% | {b["n_edges"]} | '
              f'{100 * b["coverage"]:.2f}% | {pc0(b["separation"])} ({b["n_mixed_voxels"]}) | '
              f'{f(b["node_noise_mm"], 2)} | {b["time_total_s"]:.1f} | {100 * rl["frac_R"]:.1f}% | {rl["S_median"]:.4f} |')
    print("\n### P2 — thành phần của S (Gaussian được chấm điểm)\n")
    print("| Scene | Event | r_D p50 / p75 (mm) | e_n p50 / p75 | n p50 | f_n p50 | f_D p50 | f_e p50 |")
    print("|---|---|---|---|---|---|---|---|")
    for e in ev:
        s = e.get("S_factors")
        if not s:
            continue
        print(f'| {e["scene"]} | {e["event_id"]} | {s["rD_mm"]["p50"]:.1f} / {s["rD_mm"]["p75"]:.1f} | {s["en"]["p50"]:.3f} / {s["en"]["p75"]:.3f} | '
              f'{s["n"]["p50"]:.0f} | {s["f_n"]["p50"]:.3f} | {s["f_D"]["p50"]:.4f} | {s["f_e"]["p50"]:.4f} |')
    if ev and "ablation" in ev[0]:
        print("\n### P2 — ablation (trung bình qua event)\n")
        print("| Cách dựng node | #node | độ phủ | tách lần quét |")
        print("|---|---|---|---|")
        for k in ("i_systematic", "ii_voxel", "iii_voxel_time", "iv_main"):
            rows = [e["ablation"][k] for e in ev if k in e.get("ablation", {})]
            cov = [r["coverage"] for r in rows if r.get("coverage") is not None]
            sep = [r["separation"] for r in rows if r.get("separation") is not None]
            print(f'| {k} | {np.mean([r["n_nodes"] for r in rows]):.0f} | {100 * np.mean(cov):.2f}% | '
                  f'{pc0(np.mean(sep) if sep else None)} |')


def p3(root):
    p = os.path.join(root, "P3", "P3_metrics.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    res, ev = d["result"], d["events"]
    names = [n for n in ev[0]["methods"] if n not in ("pre", "rigid")]
    print("\n### P3 — tổng hợp theo cấu hình (trung vị thay đổi tương đối so với rigid, trên event có J_eval)\n")
    print("| Cấu hình | Scene | Δe_dl | Δe_step (mm) | ΔPSNR (dB) | ΔATE kf (mm) | ΔAcc | #cải thiện e_dl / #event | #chấp nhận / #event | lý do fallback |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        for sc in sorted({e["scene"] for e in ev}):
            rr = [e for e in ev if e["scene"] == sc]
            ee = [e for e in rr if e["J_eval"]]
            def rel(k):
                v = [(e["methods"][n][k] - e["methods"]["rigid"][k]) / e["methods"]["rigid"][k] for e in ee
                     if e["methods"][n][k] is not None and e["methods"]["rigid"][k]]
                return None if not v else float(np.median(v))
            def dif(k, s=1.0):
                v = [s * (e["methods"][n][k] - e["methods"]["rigid"][k]) for e in ee
                     if e["methods"][n][k] is not None and e["methods"]["rigid"][k] is not None]
                return None if not v else float(np.median(v))
            imp = sum(1 for e in ee if e["methods"][n]["edl"] is not None and e["methods"]["rigid"]["edl"] is not None
                      and e["methods"][n]["edl"] < e["methods"]["rigid"]["edl"])
            acc = sum(1 for e in rr if e["logs"][n]["accepted"])
            reasons = {}
            for e in rr:
                if not e["logs"][n]["accepted"]:
                    reasons[e["logs"][n]["reason"]] = reasons.get(e["logs"][n]["reason"], 0) + 1
            de = rel("edl")
            print(f'| {n} | {sc} | {sgn(de)} | {mm(dif("estep"), 3)} | {f(dif("psnr"), 3)} | '
                  f'{mm(dif("ate"), 2)} | {sgn(rel("acc"))} | {imp}/{len(ee)} | {acc}/{len(rr)} | {reasons or "—"} |')
    print("\n### P3 — kết quả chọn\n")
    print("```")
    print(json.dumps({k: v for k, v in res.items() if k != "criteria"}, indent=1, default=str))
    print("```")
    print("\n### P3 — tiêu chí §8.2 theo cấu hình (giá trị, đạt?)\n")
    for n, t in res["criteria"].items():
        for sc, crit in t.items():
            print(f"- {n} / {sc}: " + ", ".join(f"{k}={'—' if v[0] is None else round(v[0], 4)} {'✅' if v[1] else '❌'}" for k, v in crit.items()))
    print("\n### P3 — chẩn đoán cặp tương ứng (cấu hình được chọn)\n")
    ch = res["chosen"]
    print("| Scene | Event | lý do | thô | sau gating | sau giới hạn | gate α / dist / normal / S / edge | E_con init→final | iters | max dịch node (mm) | max quay node (°) | pose sync t / rot / fallback | t tổng (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in ev:
        g = e["logs"].get(ch)
        if not g:
            continue
        gf = g.get("gate_frac") or {}
        ps = g.get("pose_sync") or {}
        Ei, Ef = (g.get("E_init") or {}).get("con"), (g.get("E_final") or {}).get("con")
        print(f'| {e["scene"]} | {e["event_id"]} | {g["reason"] or "accepted"} | {g.get("corr_raw")} | {g.get("corr_gated")} | {g.get("corr_capped")} | '
              f'{" / ".join(f"{100 * gf.get(k, 0):.0f}%" for k in ("alpha", "dist", "normal", "S", "edge"))} | '
              f'{f(Ei, 1)}→{f(Ef, 1)} | {g.get("lbfgs_iters")} | {mm(g.get("max_node_disp_m"), 1)} | {f(g.get("max_node_rot_deg"), 2)} | '
              f'{f(ps.get("median_t_mm"), 2)} mm / {f(ps.get("median_rot_deg"), 3)}° / {ps.get("n_fallback")} | {f((g.get("t_stage_s") or {}).get("total"), 1)} |')


if __name__ == "__main__":
    root = sys.argv[1]
    p1(root)
    p2(root)
    p3(root)
