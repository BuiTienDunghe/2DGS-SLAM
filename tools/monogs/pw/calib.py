"""PLAN MONOGS-PW §4: one-time measurement of the plane noise (sigma_n, sigma_d(z), theta_m, delta_m), the odometry
noise and kappa (1b) on the fr3 validation sequences (never on the test sequences). GT is used here only.
usage (repo root, env monogs): python tools/monogs/pw/calib.py
writes monogs/configs/rgbd/tum/pw_calib.yaml, docs/monogs/pw/CALIB_PW.md, docs/monogs/pw/calib.json, figures.
"""
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.getcwd()
MONO = os.path.join(ROOT, "monogs")
sys.path.insert(0, MONO)
sys.path.insert(0, os.path.join(ROOT, "tools", "monogs"))
from eval_runs import load_tum  # noqa: E402

VAL = ["structure_notexture_far_validation", "structure_notexture_near_validation",
       "structure_texture_far_validation", "structure_texture_near_validation"]
KAPPA_RUNS = {"snt_far_val": "results_monogs/PW/calib/snt_far_val/base", "st_far_val": "results_monogs/PW/calib/st_far_val/base"}
STEP, GAP, LAB_DEG, LAB_M, ZBIN = 5, 30, 3.0, 0.03, 0.5
D = os.path.join(ROOT, "docs/monogs/pw")


def f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}".replace(".", ",")


def mad_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(2)) if len(x) else float("nan")


def label(obs):
    """PLAN §8.1: greedy clusters in frame order, compared with the first element of each cluster."""
    heads, lab = [], []
    for o in obs:
        hit = None
        for ci, h in enumerate(heads):
            c = float(o["n"] @ h["n"])
            if c > 0 and np.degrees(np.arccos(min(1.0, c))) <= LAB_DEG and abs(o["d"] - h["d"]) <= LAB_M:
                hit = ci
                break
        if hit is None:
            heads.append(o)
            hit = len(heads) - 1
        lab.append(hit)
    return lab


def plane_obs(seq):
    os.chdir(MONO)
    from munch import munchify
    from utils.config_utils import load_config
    from utils.dataset import load_dataset
    from utils.plane_anchor import PlaneExtractor, plane_c2w
    cfg = load_config("configs/rgbd/tum/fr3_snt_far_sp.yaml")
    cfg["Dataset"]["dataset_path"] = f"datasets/tum/rgbd_dataset_freiburg3_{seq}"
    mp_ = munchify(cfg["model_params"])
    ds = load_dataset(mp_, mp_.source_path, config=cfg)
    ds.device = "cpu"
    ex = PlaneExtractor(yaml.safe_load(open("configs/rgbd/tum/e1c.yaml"))["PlaneTracking"])
    obs = []
    for i in range(0, len(ds), STEP):
        _, depth, pose = ds[i]
        T = pose.cpu().numpy().astype(np.float64)          # GT w2c
        for p in ex(depth, ds.K):
            n_w, d_w = plane_c2w(p.n.numpy(), p.d, T)
            obs.append(dict(frame=i, n=n_w / np.linalg.norm(n_w), d=float(d_w), z=float(np.mean(p.pts_c[:, 2])),
                            n_px=int(p.n_px)))
    os.chdir(ROOT)
    return obs, len(ds)


def pairs(obs):
    lab = label(obs)
    by = {}
    for o, l in zip(obs, lab):
        by.setdefault(l, []).append(o)
    out = []
    for l, os_ in by.items():
        for i, a in enumerate(os_):
            for b in os_[i + 1:]:
                if b["frame"] - a["frame"] > GAP:
                    break
                if b["frame"] == a["frame"]:
                    continue
                c = float(np.clip(a["n"] @ b["n"], -1, 1))
                out.append((np.arccos(c), a["d"] - b["d"], 0.5 * (a["z"] + b["z"])))
    return np.array(out).reshape(-1, 3), len(by)


def fit_sd(z, dd):
    edges = np.arange(0, np.ceil(z.max() / ZBIN) * ZBIN + 1e-9, ZBIN)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        if m.sum() >= 30:
            rows.append((0.5 * (lo + hi), mad_sigma(dd[m]), int(m.sum())))
    rows = np.array(rows)
    from scipy.optimize import nnls
    w = np.sqrt(rows[:, 2])
    A = np.c_[np.ones(len(rows)), rows[:, 0] ** 2] * w[:, None]
    (a, b), _ = nnls(A, rows[:, 1] * w)
    return float(a), float(b), rows


def se3_log_w2c(Ta, Tb):
    sys.path.insert(0, MONO)
    from utils.pgo_planes import se3_log
    return se3_log(Ta @ np.linalg.inv(Tb))


def odo_and_kappa():
    from utils.pgo_planes import se3_log
    res = {}
    E_t, E_r = [], []
    Z = {"all": [], "t": [], "r": []}
    for name, run in KAPPA_RUNS.items():
        _, est = load_tum(os.path.join(run, "traj_all.txt"))
        _, gt = load_tum(os.path.join(run, "traj_gt_all.txt"))
        W = [np.linalg.inv(T) for T in est]
        G = [np.linalg.inv(T) for T in gt]
        kf = [int(x) for x in open(os.path.join(run, "kf_indices.txt")).read().split()]
        for i, j in zip(kf[:-1], kf[1:]):
            e = se3_log((W[j] @ np.linalg.inv(W[i])) @ np.linalg.inv(G[j] @ np.linalg.inv(G[i])))
            E_t.append(np.linalg.norm(e[:3]))
            E_r.append(np.linalg.norm(e[3:]))
        zs = {"all": [], "t": [], "r": []}
        for line in open(os.path.join(run, "stiffness.jsonl")):
            r = json.loads(line)
            t = r["frame"]
            if t < 1 or (r.get("iters") or 0) < 10:
                continue
            eps = se3_log((W[t] @ np.linalg.inv(W[t - 1])) @ np.linalg.inv(G[t] @ np.linalg.inv(G[t - 1])))
            for k in range(6):
                c = r["c_render"][k]
                if c <= 0:
                    continue
                v = eps[k] ** 2 * c
                zs["all"].append(v)
                zs["t" if k < 3 else "r"].append(v)
        res[name] = {g: (2 * 0.455 / np.median(v) if v else None) for g, v in zs.items()}
        res[name]["n"] = len(zs["all"])
        for g in Z:
            Z[g] += zs[g]
    kappa = {g: (2 * 0.455 / np.median(v) if v else None) for g, v in Z.items()}
    s_t = float(np.sqrt(np.mean(np.square(E_t))) / np.sqrt(3))
    s_r = float(np.sqrt(np.mean(np.square(E_r))) / np.sqrt(3))
    return dict(kappa=kappa["all"], kappa_t=kappa["t"], kappa_r=kappa["r"], per_run=res, sigma_odo_t=s_t,
                sigma_odo_r=s_r, n_kf_pairs=len(E_t))


def main():
    os.makedirs(D, exist_ok=True)
    allp, per = [], {}
    for s in VAL:
        p = f"monogs/datasets/tum/rgbd_dataset_freiburg3_{s}/groundtruth.txt"
        if not os.path.exists(p):
            per[s] = dict(skipped="no groundtruth.txt")
            continue
        obs, n = plane_obs(s)
        P, n_lab = pairs(obs)
        allp.append(P)
        a_, b_, rows = fit_sd(P[:, 2], P[:, 1]) if len(P) else (None, None, np.zeros((0, 3)))
        per[s] = dict(n_frames=n, n_obs=len(obs), n_labels=n_lab, n_pairs=int(len(P)), sigma_n_deg=float(np.degrees(mad_sigma(P[:, 0]))),
                      sigma_d_bins=[dict(z=float(r[0]), sigma_d_cm=100 * float(r[1]), n=int(r[2])) for r in rows],
                      fit_a=a_, fit_b=b_)
        print(s, {k: v for k, v in per[s].items() if k != "sigma_d_bins"}, flush=True)
    P = np.concatenate(allp)
    sn = mad_sigma(P[:, 0])
    a, b, rows = fit_sd(P[:, 2], P[:, 1])
    th_m = float(np.degrees(np.percentile(P[:, 0], 99)))
    de_m = float(np.percentile(np.abs(P[:, 1]), 99))
    sys.path.insert(0, MONO)
    ok = odo_and_kappa()
    cal = dict(source="tools/monogs/pw/calib.py on the 4 fr3 *_validation sequences (planes) and the two base "
                      "calibration runs (kappa, odometry)",
               sigma_n=float(sn), sigma_d_a=a, sigma_d_b=b, sigma_odo_t=ok["sigma_odo_t"], sigma_odo_r=ok["sigma_odo_r"],
               merge_deg=th_m, merge_m=de_m, kappa=float(ok["kappa"]) if ok["kappa"] else None)
    yaml.safe_dump({"Noise": cal}, open("monogs/configs/rgbd/tum/pw_calib.yaml", "w"), sort_keys=False)
    json.dump(dict(noise=cal, per_sequence=per, kappa=ok, pooled_bins=rows.tolist(), n_pairs=int(len(P))),
              open(os.path.join(D, "calib.json"), "w"), indent=1, default=float)
    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(np.degrees(P[:, 0]), bins=100, range=(0, 6))
    ax[0].axvline(np.degrees(sn), color="r", label=f"σ_n = {np.degrees(sn):.2f}°")
    ax[0].axvline(th_m, color="k", ls="--", label=f"θ_m (p99) = {th_m:.2f}°")
    ax[0].set_xlabel("Δgóc (°)")
    ax[0].legend()
    ax[1].scatter(rows[:, 0], 100 * rows[:, 1], s=10 + rows[:, 2] / 50, label="σ_d theo khoảng z")
    zz = np.linspace(0, rows[:, 0].max() + 0.5, 50)
    ax[1].plot(zz, 100 * (a + b * zz ** 2), "r", label=f"fit {100 * a:.2f} + {100 * b:.3f} z² (cm)")
    ax[1].set_xlabel("z (m)")
    ax[1].set_ylabel("σ_d (cm)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(D, "calib_noise.png"), dpi=110)
    # report
    L = ["# CALIB_PW — hiệu chỉnh một lần (PLAN MONOGS-PW §4)\n",
         "Sinh bằng `tools/monogs/pw/calib.py`. Chỉ dùng 4 chuỗi `*_validation` của fr3 (mặt phẳng) và hai run `base` "
         "hiệu chỉnh trên `structure_notexture_far_validation`, `structure_texture_far_validation` (frame 0–300). "
         f"Mặt phẳng lấy mỗi {STEP} frame bằng đúng extractor của SLAM, đổi sang world bằng pose GT; nhãn mặt thật: gom tham lam "
         f"(≤ {LAB_DEG:g}°, ≤ {100 * LAB_M:g} cm); cặp cùng nhãn cách nhau ≤ {GAP} frame.\n",
         "## Bộ số khóa (`pw_calib.yaml`)\n", "| Đại lượng | Giá trị |", "|---|---|",
         f"| σ_n | {f(np.degrees(sn))}° ({sn:.5f} rad) |",
         f"| σ_d(z) = a + b·z² | a = {f(100 * a)} cm, b = {f(100 * b, 4)} cm/m² |",
         f"| θ_m, δ_m (phân vị 99 %) | {f(th_m)}°, {f(100 * de_m)} cm |",
         f"| σ_odo (dịch, quay) | {f(100 * ok['sigma_odo_t'])} cm, {f(np.degrees(ok['sigma_odo_r']))}° ({ok['n_kf_pairs']} cặp keyframe liên tiếp) |",
         f"| κ (chung) | {f(ok['kappa'], 4)} |",
         f"| κ nhóm dịch / nhóm quay | {f(ok['kappa_t'], 4)} / {f(ok['kappa_r'], 4)} |", "",
         "## Từng chuỗi validation\n", "| Chuỗi | Frame | Mặt đo | Nhãn | Cặp | σ_n (°) | a (cm) | b (cm/m²) |",
         "|---|---|---|---|---|---|---|---|"]
    for s, v in per.items():
        if "skipped" in v:
            L.append(f"| {s} | bỏ: {v['skipped']} | | | | | | |")
            continue
        L.append(f"| {s} | {v['n_frames']} | {v['n_obs']} | {v['n_labels']} | {v['n_pairs']} | {f(v['sigma_n_deg'])} | "
                 f"{f(100 * v['fit_a']) if v['fit_a'] is not None else '—'} | {f(100 * v['fit_b'], 4) if v['fit_b'] is not None else '—'} |")
    L += ["", "**σ_d theo khoảng z (gộp mọi chuỗi):**\n", "| z (m) | σ_d (cm) | Cặp |", "|---|---|---|"]
    L += [f"| {f(r[0], 2)} | {f(100 * r[1])} | {int(r[2])} |" for r in rows]
    L += ["", "**κ từng run hiệu chỉnh:**\n", "| Run | κ chung | κ dịch | κ quay | Mẫu |", "|---|---|---|---|---|"]
    for n, v in ok["per_run"].items():
        L.append(f"| {n} | {f(v['all'], 4)} | {f(v['t'], 4)} | {f(v['r'], 4)} | {v['n']} |")
    sn_s = [v["sigma_n_deg"] for v in per.values() if "sigma_n_deg" in v]
    kap = [v["all"] for v in ok["per_run"].values() if v["all"]]
    L += ["", "**Kiểm khả năng chuyển (chỉ báo cáo, không đổi số):**\n",
          f"- σ_n các chuỗi: {', '.join(f(x) for x in sn_s)}°; lớn nhất / nhỏ nhất = {f(max(sn_s) / min(sn_s), 2)}"
          + (" (> 2: **ghi nhận**)" if max(sn_s) / min(sn_s) > 2 else " (≤ 2)") + ".",
          f"- κ hai run: {', '.join(f(x, 4) for x in kap)}; tỉ lệ = {f(max(kap) / min(kap), 2)}"
          + (" (> 3: **ghi nhận**)" if max(kap) / min(kap) > 3 else " (≤ 3)") + "." if len(kap) == 2 else "- κ: thiếu run.",
          "", "![nhiễu](calib_noise.png)", ""]
    open(os.path.join(D, "CALIB_PW.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
