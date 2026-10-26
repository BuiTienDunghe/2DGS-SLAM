"""PLAN MONOGS-PW2 §4: sigma_pgo (GT poses, every frame distance), sigma_eff (estimated poses, the way anchors are
used), sigma_odo, theta_m / delta_m, offline kernel check. Validation sequences only. GT is used here only.
usage (repo root, env monogs): python tools/monogs/pw2/calib2.py
writes monogs/configs/rgbd/tum/pw2_calib.yaml, docs/monogs/pw2/CALIB_PW2.md, calib2.json, figures.
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

SEQS = {"snt_far_val": "structure_notexture_far_validation", "st_far_val": "structure_texture_far_validation",
        "snt_near_val": "structure_notexture_near_validation"}
E1C_RUN = "results_monogs/PW2/calib/{}/e1c"
BASE_RUN = {"snt_far_val": "results_monogs/PW/calib/snt_far_val/base", "st_far_val": "results_monogs/PW/calib/st_far_val/base"}
MAXF, STEP, LAB_DEG, LAB_M, ZBIN, MAXPAIRS = 500, 5, 3.0, 0.03, 0.5, 200000
GAPS = [(1, 30), (31, 100), (101, 300), (301, 10 ** 9)]
D = os.path.join(ROOT, "docs/monogs/pw2")
R1 = dict(sn=0.0041907774738404934, a=0.00760616476617554, b=4.204491930675162e-05)


def f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}".replace(".", ",")


def mad(x, sqrt2=False):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan")
    s = 1.4826 * np.median(np.abs(x - np.median(x)))
    return float(s / np.sqrt(2) if sqrt2 else s)


def tbasis(n):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(n, a)
    b1 /= np.linalg.norm(b1)
    return np.stack([b1, np.cross(n, b1)], 0)          # (2, 3)


def fit_ab(z, s, n):
    """sigma(z) = a + b z^2, a, b >= 0, weights = pair counts (nnls)."""
    from scipy.optimize import nnls
    w = np.sqrt(np.asarray(n, dtype=np.float64))
    A = np.c_[np.ones(len(z)), np.asarray(z) ** 2] * w[:, None]
    (a, b), _ = nnls(A, np.asarray(s) * w)
    return float(a), float(b)


def zbins(z, v, sqrt2):
    rows = []
    hi = np.ceil(np.max(z) / ZBIN) * ZBIN if len(z) else 0
    for lo in np.arange(0, hi, ZBIN):
        m = (z >= lo) & (z < lo + ZBIN)
        if m.sum() >= 30:
            rows.append((lo + ZBIN / 2, mad(v[m], sqrt2), int(m.sum())))
    return rows


def c2w_to_w2c(Ts):
    return [np.linalg.inv(T) for T in Ts]


def extract(seq, frames):
    os.chdir(MONO)
    from munch import munchify
    from utils.config_utils import load_config
    from utils.dataset import load_dataset
    from utils.plane_anchor import PlaneExtractor
    cfg = load_config("configs/rgbd/tum/fr3_snt_far_sp.yaml")
    cfg["Dataset"]["dataset_path"] = f"datasets/tum/rgbd_dataset_freiburg3_{SEQS[seq]}"
    mp_ = munchify(cfg["model_params"])
    ds = load_dataset(mp_, mp_.source_path, config=cfg)
    ds.device = "cpu"
    ex = PlaneExtractor(yaml.safe_load(open("configs/rgbd/tum/e1c.yaml"))["PlaneTracking"])
    obs, G = [], {}
    for i in sorted(frames):
        _, depth, pose = ds[i]
        G[i] = pose.cpu().numpy().astype(np.float64)      # GT w2c
        for p in ex(depth, ds.K):
            obs.append(dict(frame=i, n=p.n.numpy().copy(), d=float(p.d), z=float(np.mean(p.pts_c[:, 2])), n_px=int(p.n_px)))
    os.chdir(ROOT)
    return obs, G


def label(obs, G):
    from utils.plane_anchor import plane_c2w
    heads, lab = [], []
    for o in obs:
        n_w, d_w = plane_c2w(o["n"], o["d"], G[o["frame"]])
        o["n_g"], o["d_g"] = n_w / np.linalg.norm(n_w), d_w
        hit = None
        for ci, h in enumerate(heads):
            c = float(o["n_g"] @ h["n_g"])
            if c > 0 and np.degrees(np.arccos(min(1.0, c))) <= LAB_DEG and abs(o["d_g"] - h["d_g"]) <= LAB_M:
                hit = ci
                break
        if hit is None:
            heads.append(o)
            hit = len(heads) - 1
        lab.append(hit)
        o["label"] = hit
    return lab


def pgo_pairs(obs, rng):
    by = {}
    for o in obs:
        by.setdefault(o["label"], []).append(o)
    pairs = []
    for l, os_ in by.items():
        for i in range(len(os_)):
            for j in range(i + 1, len(os_)):
                if os_[i]["frame"] != os_[j]["frame"]:
                    pairs.append((os_[i], os_[j]))
    if len(pairs) > MAXPAIRS:
        idx = rng.choice(len(pairs), MAXPAIRS, replace=False)
        pairs = [pairs[i] for i in idx]
    rows = []
    for a, b in pairs:
        B = tbasis(a["n_g"])
        comp = B @ (b["n_g"] - a["n_g"])
        ang = np.arccos(np.clip(a["n_g"] @ b["n_g"], -1, 1))
        rows.append((comp[0], comp[1], ang, a["d_g"] - b["d_g"], 0.5 * (a["z"] + b["z"]), abs(a["frame"] - b["frame"])))
    return np.array(rows).reshape(-1, 6)


def eff_residuals(obs, run, sigma=None):
    """Anchor = first keyframe observation of each label (largest piece), placed with the FINAL keyframe pose;
    residual at the estimated pose of every later frame with an observation of that label."""
    from utils.plane_anchor import plane_c2w
    kf = [int(x) for x in open(os.path.join(run, "kf_indices.txt")).read().split()]
    _, Tk = load_tum(os.path.join(run, "traj_kf.txt"))
    Wk = dict(zip(kf, c2w_to_w2c(Tk)))
    _, Ta = load_tum(os.path.join(run, "traj_all.txt"))
    Wa = c2w_to_w2c(Ta)
    kfs = set(kf)
    by = {}
    for o in obs:
        by.setdefault(o["label"], []).append(o)
    anchors = {}
    for l, os_ in by.items():
        ok = sorted([o for o in os_ if o["frame"] in kfs], key=lambda o: (o["frame"], -o["n_px"]))
        if not ok:
            continue
        c = ok[0]
        n_w, d_w = plane_c2w(c["n"], c["d"], Wk[c["frame"]])
        anchors[l] = (n_w / np.linalg.norm(n_w), d_w, c["frame"])
    rows = []
    for o in obs:
        if o["label"] not in anchors or o["frame"] >= len(Wa):
            continue
        n_w, d_w, f0 = anchors[o["label"]]
        if o["frame"] <= f0:
            continue
        T = Wa[o["frame"]]
        Rn = T[:3, :3] @ n_w
        rn = tbasis(o["n"]) @ (o["n"] - Rn)
        rd = o["d"] - (d_w - Rn @ T[:3, 3])
        rows.append((rn[0], rn[1], rd, o["z"], o["frame"] - f0))
    return np.array(rows).reshape(-1, 5), anchors, Wa


def kernel_check(obs, anchors, Wa, sn, a, b):
    """% correct-label pairs with e <= 3, % wrong-label anchors inside the E1c gate with e <= 3."""
    ok_true, ok_false = [], []
    for o in obs:
        if o["frame"] >= len(Wa):
            continue
        T = Wa[o["frame"]]
        for l, (n_w, d_w, f0) in anchors.items():
            if o["frame"] <= f0:
                continue
            Rn = T[:3, :3] @ n_w
            d_c = d_w - Rn @ T[:3, 3]
            c = float(np.clip(o["n"] @ Rn, -1, 1))
            if d_c <= 0 or np.degrees(np.arccos(c)) > 10 or abs(o["d"] - d_c) > 0.10:
                continue
            r_n = o["n"] - Rn
            e2 = float(r_n @ r_n) / sn ** 2 + (o["d"] - d_c) ** 2 / (a + b * o["z"] ** 2) ** 2
            (ok_true if l == o["label"] else ok_false).append(e2 <= 9.0)
    return (100 * float(np.mean(ok_true)) if ok_true else None, len(ok_true),
            100 * float(np.mean(ok_false)) if ok_false else None, len(ok_false))


def main():
    os.makedirs(D, exist_ok=True)
    rng = np.random.default_rng(0)
    from utils.pgo_planes import se3_log
    per, pg_all, eff_all, odo_t, odo_r = {}, [], [], [], []
    store = {}
    for seq in SEQS:
        run = E1C_RUN.format(seq)
        if not os.path.exists(os.path.join(run, "traj_all.txt")):
            per[seq] = dict(skipped=f"no run {run}")
            continue
        kf = [int(x) for x in open(os.path.join(run, "kf_indices.txt")).read().split()]
        frames = sorted({i for i in range(0, MAXF, STEP)} | {k for k in kf if k < MAXF})
        obs, G = extract(seq, frames)
        label(obs, G)
        P = pgo_pairs(obs, rng)
        comps = P[:, :2].ravel()
        s_n = mad(comps, True)
        zb = zbins(P[:, 4], P[:, 3], True)
        a_, b_ = fit_ab([r[0] for r in zb], [r[1] for r in zb], [r[2] for r in zb])
        near = P[:, 5] <= 30
        gaps = {f"{lo}-{hi}": dict(n=int(((P[:, 5] >= lo) & (P[:, 5] <= hi)).sum()),
                                   sigma_n_deg=float(np.degrees(mad(P[(P[:, 5] >= lo) & (P[:, 5] <= hi), :2].ravel(), True))),
                                   sigma_d_cm=100 * mad(P[(P[:, 5] >= lo) & (P[:, 5] <= hi), 3], True)) for lo, hi in GAPS}
        E, anchors, Wa = eff_residuals(obs, run)
        se_n = mad(E[:, :2].ravel())
        zbe = zbins(E[:, 3], E[:, 2], False)
        ae, be = fit_ab([r[0] for r in zbe], [r[1] for r in zbe], [r[2] for r in zbe]) if zbe else (float("nan"), float("nan"))
        egaps = {f"{lo}-{hi}": dict(n=int(((E[:, 4] >= lo) & (E[:, 4] <= hi)).sum()),
                                    sigma_n_deg=float(np.degrees(mad(E[(E[:, 4] >= lo) & (E[:, 4] <= hi), :2].ravel()))),
                                    sigma_d_cm=100 * mad(E[(E[:, 4] >= lo) & (E[:, 4] <= hi), 2])) for lo, hi in GAPS}
        # sigma_odo: consecutive keyframes of the e1c run
        _, Tk = load_tum(os.path.join(run, "traj_kf.txt"))
        Wk = dict(zip(kf, c2w_to_w2c(Tk)))
        for i, j in zip(kf[:-1], kf[1:]):
            if i in G and j in G:
                e = se3_log((Wk[j] @ np.linalg.inv(Wk[i])) @ np.linalg.inv(G[j] @ np.linalg.inv(G[i])))
                odo_t.append(np.linalg.norm(e[:3]))
                odo_r.append(np.linalg.norm(e[3:]))
        per[seq] = dict(n_obs=len(obs), n_labels=len(set(o["label"] for o in obs)), n_pairs=len(P), n_kf=len(kf),
                        pgo=dict(sigma_n_deg=float(np.degrees(s_n)), a=a_, b=b_, bins=zb,
                                 near30=dict(sigma_n_deg=float(np.degrees(mad(P[near, :2].ravel(), True))),
                                             sigma_d_cm=100 * mad(P[near, 3], True)), gaps=gaps),
                        eff=dict(sigma_n_deg=float(np.degrees(se_n)), a=ae, b=be, bins=zbe, gaps=egaps, n=len(E)))
        pg_all.append(P)
        eff_all.append(E)
        store[seq] = (obs, anchors, Wa)
        print(seq, json.dumps({k: v for k, v in per[seq].items() if k not in ("pgo", "eff")}),
              "pgo", f"{np.degrees(s_n):.3f}deg a={a_:.4f} b={b_:.5f}", "eff", f"{np.degrees(se_n):.3f}deg a={ae:.4f} b={be:.5f}", flush=True)
    Pall = np.concatenate(pg_all)
    Eall = np.concatenate(eff_all)
    ok = [s for s in per if "pgo" in per[s]]
    sn_pgo = mad(Pall[:, :2].ravel(), True)
    a_pgo = float(np.median([per[s]["pgo"]["a"] for s in ok]))
    b_pgo = float(np.median([per[s]["pgo"]["b"] for s in ok]))
    sn_eff = mad(Eall[:, :2].ravel())
    a_eff = float(np.median([per[s]["eff"]["a"] for s in ok]))
    b_eff = float(np.median([per[s]["eff"]["b"] for s in ok]))
    th_m = float(np.degrees(np.percentile(Pall[:, 2], 99)))
    de_m = float(np.percentile(np.abs(Pall[:, 3]), 99))
    s_odo_t = float(np.sqrt(np.mean(np.square(odo_t))) / np.sqrt(3))
    s_odo_r = float(np.sqrt(np.mean(np.square(odo_r))) / np.sqrt(3))
    kappa_r1 = yaml.safe_load(open("monogs/configs/rgbd/tum/pw_calib.yaml"))["Noise"]["kappa"]
    # offline kernel check (sigma_eff vs R1 sigma)
    kc = {}
    for s, (obs, anchors, Wa) in store.items():
        kc[s] = dict(eff=kernel_check(obs, anchors, Wa, sn_eff, a_eff, b_eff), r1=kernel_check(obs, anchors, Wa, R1["sn"], R1["a"], R1["b"]))
    # sensitivity: sigma_eff with the R1 base runs (frames 0-300)
    sens = {}
    for s, run in BASE_RUN.items():
        if s in store and os.path.exists(os.path.join(run, "traj_all.txt")):
            obs = [o for o in store[s][0] if o["frame"] < 300]
            E, _, _ = eff_residuals(obs, run)
            zbe = zbins(E[:, 3], E[:, 2], False)
            sens[s] = dict(sigma_n_deg=float(np.degrees(mad(E[:, :2].ravel()))), sigma_d_cm=100 * mad(E[:, 2]), n=len(E),
                           bins=zbe)
    cal = dict(source="tools/monogs/pw2/calib2.py: e1c runs on 3 fr3 validation sequences (frames 0-500)",
               sigma_n=float(sn_pgo), sigma_d_a=a_pgo, sigma_d_b=b_pgo,
               sigma_eff_n=float(sn_eff), sigma_eff_d_a=a_eff, sigma_eff_d_b=b_eff,
               sigma_odo_t=s_odo_t, sigma_odo_r=s_odo_r, merge_deg=th_m, merge_m=de_m, kappa=kappa_r1)
    yaml.safe_dump({"Noise": cal}, open("monogs/configs/rgbd/tum/pw2_calib.yaml", "w"), sort_keys=False)
    json.dump(dict(noise=cal, per_sequence=per, kernel_check=kc, sensitivity_base=sens, n_odo=len(odo_t)),
              open(os.path.join(D, "calib2.json"), "w"), indent=1, default=float)
    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].hist(np.degrees(Pall[:, :2].ravel()), bins=120, range=(-2, 2), alpha=0.6, label="σ_pgo (GT, mọi khoảng cách)", density=True)
    ax[0].hist(np.degrees(Eall[:, :2].ravel()), bins=120, range=(-2, 2), alpha=0.6, label="σ_eff (pose ước lượng)", density=True)
    ax[0].set_xlabel("thành phần pháp tuyến (°)")
    ax[0].legend(fontsize=8)
    zz = np.linspace(0.3, 4, 50)
    for s in ok:
        ax[1].plot([r[0] for r in per[s]["pgo"]["bins"]], [100 * r[1] for r in per[s]["pgo"]["bins"]], "o-", label=f"pgo {s}")
        ax[2].plot([r[0] for r in per[s]["eff"]["bins"]], [100 * r[1] for r in per[s]["eff"]["bins"]], "o-", label=f"eff {s}")
    ax[1].plot(zz, 100 * (a_pgo + b_pgo * zz ** 2), "k--", label="trung vị fit")
    ax[2].plot(zz, 100 * (a_eff + b_eff * zz ** 2), "k--", label="trung vị fit")
    for a_ in ax[1:]:
        a_.set_xlabel("z (m)")
        a_.set_ylabel("σ_d (cm)")
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "calib2_noise.png"), dpi=110)
    # report
    L = ["# CALIB_PW2 — hai thang nhiễu (PLAN MONOGS-PW2 §4)\n",
         "Sinh bằng `tools/monogs/pw2/calib2.py`. Dữ liệu: 3 run `e1c` trên `structure_notexture_far_validation`, "
         "`structure_texture_far_validation`, `structure_notexture_near_validation` (frame 0–500). Mặt đo: extractor của SLAM ở "
         "mọi frame chia hết cho 5 và mọi keyframe; nhãn mặt thật như R1 (pose GT, 3° / 3 cm).\n",
         "## Bộ số khóa (`pw2_calib.yaml`)\n", "| Thang | σ_n | σ_d(z) = a + b·z² | Dùng ở |", "|---|---|---|---|",
         f"| σ_pgo (pose GT, mọi khoảng cách frame) | {f(np.degrees(sn_pgo))}° | a = {f(100 * a_pgo)} cm, b = {f(100 * b_pgo, 4)} cm/m² | factor mặt phẳng PGO, kiểm χ² |",
         f"| σ_eff (pose ước lượng, như anchor trong SLAM) | {f(np.degrees(sn_eff))}° | a = {f(100 * a_eff)} cm, b = {f(100 * b_eff, 4)} cm/m² | kernel, inlier, khởi tạo bền, loss |",
         f"| R1 (để so) | {f(np.degrees(R1['sn']))}° | a = {f(100 * R1['a'])} cm, b = {f(100 * R1['b'], 4)} cm/m² | |", "",
         f"σ_odo = {f(100 * s_odo_t)} cm / {f(np.degrees(s_odo_r))}° ({len(odo_t)} cặp keyframe liên tiếp của run e1c; R1 lấy từ base: 3,96 cm / 0,77°). "
         f"θ_m = {f(th_m)}°, δ_m = {f(100 * de_m)} cm (phân vị 99 %, mọi cặp). κ giữ số R1 ({f(kappa_r1, 1)}), chỉ để log.\n",
         "## Từng chuỗi\n",
         "| Chuỗi | Mặt đo | Nhãn | Cặp (pgo) | σ_pgo,n (°) | a_pgo / b_pgo | σ_pgo ≤ 30 frame: n (°) / d (cm) | Phần dư eff | σ_eff,n (°) | a_eff / b_eff |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in ok:
        v = per[s]
        L.append(f"| {s} | {v['n_obs']} | {v['n_labels']} | {v['n_pairs']} | {f(v['pgo']['sigma_n_deg'])} | {f(100 * v['pgo']['a'])} / {f(100 * v['pgo']['b'], 4)} | "
                 f"{f(v['pgo']['near30']['sigma_n_deg'])} / {f(v['pgo']['near30']['sigma_d_cm'])} | {v['eff']['n']} | {f(v['eff']['sigma_n_deg'])} | "
                 f"{f(100 * v['eff']['a'])} / {f(100 * v['eff']['b'], 4)} |")
    L += ["", "**Theo khoảng cách frame** (σ_n °, σ_d cm, số cặp):\n", "| Chuỗi | Thang | " + " | ".join(f"{lo}–{hi if hi < 10 ** 8 else '…'}" for lo, hi in GAPS) + " |",
          "|---|---|" + "---|" * len(GAPS)]
    for s in ok:
        for key, name in (("pgo", "σ_pgo"), ("eff", "σ_eff")):
            g = per[s][key]["gaps"]
            L.append(f"| {s} | {name} | " + " | ".join(f"{f(v['sigma_n_deg'])} / {f(v['sigma_d_cm'])} ({v['n']})" for v in g.values()) + " |")
    L += ["", "**σ_d theo độ sâu** (cm, số cặp):\n", "| Chuỗi | Thang | Các khoảng z (tâm, m) |", "|---|---|---|"]
    for s in ok:
        for key, name in (("pgo", "σ_pgo"), ("eff", "σ_eff")):
            L.append(f"| {s} | {name} | " + "; ".join(f"{f(r[0], 2)}: {f(100 * r[1])} ({r[2]})" for r in per[s][key]["bins"]) + " |")
    L += ["", "## Kiểm offline kernel (chỉ báo)\n",
          "| Chuỗi | σ_eff: % cặp đúng nhãn có e ≤ 3 (số cặp) | σ_eff: % anchor sai nhãn trong cổng có e ≤ 3 (số) | σ R1: đúng nhãn | σ R1: sai nhãn |",
          "|---|---|---|---|---|"]
    for s, v in kc.items():
        L.append(f"| {s} | {f(v['eff'][0], 1)} ({v['eff'][1]}) | {f(v['eff'][2], 1)} ({v['eff'][3]}) | {f(v['r1'][0], 1)} | {f(v['r1'][2], 1)} |")
    L += ["", "**Độ nhạy theo chất lượng tracking** (σ_eff đo bằng pose của run `base` hiệu chỉnh R1, frame 0–300; chỉ báo):\n",
          "| Chuỗi | σ_eff,n (°) | σ_eff,d (cm, mọi z) | Phần dư |", "|---|---|---|---|"]
    for s, v in sens.items():
        L.append(f"| {s} | {f(v['sigma_n_deg'])} | {f(v['sigma_d_cm'])} | {v['n']} |")
    L += ["", "![nhiễu](calib2_noise.png)", ""]
    open(os.path.join(D, "CALIB_PW2.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
