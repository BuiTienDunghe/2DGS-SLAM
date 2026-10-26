"""PLAN MONOGS-PW2 §4.6 (CPU, report only): does the render stiffness predict the tracking error at longer horizons?
usage (repo root, env monogs): python tools/monogs/pw2/kappa_horizon.py   -> docs/monogs/pw2/KAPPA_HORIZON.md, kappa_horizon.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "monogs"))
sys.path.insert(0, os.path.join(ROOT, "tools", "monogs"))
from eval_runs import load_tum  # noqa: E402
from utils.pgo_planes import se3_log  # noqa: E402

RUNS = {"snt_far_val": "results_monogs/PW/calib/snt_far_val/base", "st_far_val": "results_monogs/PW/calib/st_far_val/base"}
AX = ["tx", "ty", "tz", "rx", "ry", "rz"]
D = "docs/monogs/pw2"


def f(x, nd=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}g}".replace(".", ",")


def main():
    os.makedirs(D, exist_ok=True)
    out = {}
    for name, run in RUNS.items():
        _, est = load_tum(os.path.join(run, "traj_all.txt"))
        _, gt = load_tum(os.path.join(run, "traj_gt_all.txt"))
        W = [np.linalg.inv(T) for T in est]
        G = [np.linalg.inv(T) for T in gt]
        kf = [int(x) for x in open(os.path.join(run, "kf_indices.txt")).read().split()]
        st = {json.loads(l)["frame"]: json.loads(l) for l in open(os.path.join(run, "stiffness.jsonl"))}

        def eps(t, dlt):
            return se3_log((W[t] @ np.linalg.inv(W[t - dlt])) @ np.linalg.inv(G[t] @ np.linalg.inv(G[t - dlt])))

        fr = [t for t in sorted(st) if t >= 1]
        C = np.array([st[t]["c_render"] for t in fr])
        E1 = np.array([eps(t, 1) for t in fr]) ** 2
        r = dict(c_render_med=np.median(C, 0).tolist(), eps2_med=np.median(E1, 0).tolist(),
                 spearman=[float(spearmanr(C[:, k], E1[:, k]).correlation) for k in range(6)])
        hz = {}
        for dlt in (1, 5, 20):
            ts = [t for t in fr if t >= dlt]
            v = np.array([eps(t, dlt) ** 2 * np.array(st[t]["c_render"]) for t in ts])
            hz[str(dlt)] = (np.median(v, 0) / dlt).tolist()
        kv = []
        for i, j in zip(kf[:-1], kf[1:]):
            near = min(st, key=lambda t: abs(t - j))
            kv.append(eps(j, j - i) ** 2 * np.array(st[near]["c_render"]) / max(1, j - i))
        hz["kf"] = np.median(np.array(kv), 0).tolist() if kv else None
        r["horizon"] = hz
        out[name] = r
    json.dump(out, open(os.path.join(D, "kappa_horizon.json"), "w"), indent=1)
    L = ["# KAPPA_HORIZON — độ cứng render và sai số theo tầm (PLAN MONOGS-PW2 §4.6, chỉ báo)\n",
         "Sinh bằng `tools/monogs/pw2/kappa_horizon.py` trên hai run `base` hiệu chỉnh của R1 (frame 0–300, đo độ cứng mỗi 5 frame). "
         "ε_Δ = Log(M_est·M_gt⁻¹), M = T(t)·T(t−Δ)⁻¹, theo trục τ (m, rad).\n"]
    for name, r in out.items():
        L += [f"## {name}\n", "| Đại lượng | " + " | ".join(AX) + " |", "|---|" + "---|" * 6,
              "| trung vị c_render | " + " | ".join(f(x) for x in r["c_render_med"]) + " |",
              "| trung vị ε² (Δ = 1) | " + " | ".join(f(x) for x in r["eps2_med"]) + " |",
              "| Spearman(c_render, ε²) | " + " | ".join(f(x, 2) for x in r["spearman"]) + " |"]
        for k, v in r["horizon"].items():
            if v is not None:
                L.append(f"| trung vị(ε_Δ²·c)/Δ, Δ = {k} | " + " | ".join(f(x) for x in v) + " |")
        L.append("")
    L.append("**Đọc:** nếu dòng trung vị(ε_Δ²·c)/Δ tăng mạnh theo Δ, sai số render tương quan theo thời gian (trôi); khi đó κ đo "
             "từ hai frame liền nhau làm render được tin quá mức về lâu dài. Spearman dương nghĩa là trục render cứng hơn lại có sai số "
             "lớn hơn, ngược với giả định của 1b.\n")
    open(os.path.join(D, "KAPPA_HORIZON.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
