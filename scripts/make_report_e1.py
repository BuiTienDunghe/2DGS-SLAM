"""PLAN E1 §10: REPORT.md + figs F1–F5, every number read from the run folders.
usage: python scripts/make_report_e1.py RESULTS_ROOT SEQ [T7_JSON]
  RESULTS_ROOT = results_exp/e1 ; runs at RESULTS_ROOT/<SEQ>/<method>/seed<k>/
"""
import csv
import glob
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from evo.core import trajectory  # noqa: E402
from evo.core.trajectory import PosePath3D  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

GT_C, BASE_C, OURS_C = "#0b0b0b", "#8f8e88", "#2a78d6"     # PLAN §10: GT black, base gray, ours blue
INK2, GRID = "#52514e", "#e4e3df"
COLORS = {"base": BASE_C, "ours": OURS_C, "ours_init": "#6250d6", "ours_loss": "#1baf7a"}
# PLAN §9 reference numbers (other systems, other protocols; keyframe ATE, cm)
REFS = {"fr3_snt_far": [("StructureGS-SLAM", 7.47), ("PlanarSLAM", 2.2)],
        "fr3_snt_near": [("StructureGS-SLAM", 16.36), ("PlanarSLAM", 2.5)]}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

root, seq = sys.argv[1], sys.argv[2]
t7_json = sys.argv[3] if len(sys.argv) > 3 else None
figdir = os.path.join(root, "figs")
os.makedirs(figdir, exist_ok=True)


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def tum(p):
    a = np.loadtxt(p)
    T = np.tile(np.eye(4), (len(a), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix()
    T[:, :3, 3] = a[:, 1:4]
    return a[:, 0], T


def status(m, ape):
    if m is None:
        return "FAIL"
    if m.get("has_nan_pose") or m.get("ate_rmse_m") is None:
        return "FAIL"
    if m["ate_rmse_m"] > 0.50 or m.get("frac_frames_ape_gt_30cm", 0) > 0.20:
        return "FAIL"
    return "DEGRADED" if m["ate_rmse_m"] > 0.10 else "OK"


runs = {}
for d in sorted(glob.glob(os.path.join(root, seq, "*", "seed*"))):
    meth, sd = d.split(os.sep)[-2], d.split(os.sep)[-1]
    mp = os.path.join(d, "e1_metrics.json")
    m = json.load(open(mp)) if os.path.exists(mp) else None
    r = dict(dir=d, method=meth, seed=sd, m=m)
    if m is not None:
        r["ape"] = read_csv(os.path.join(d, "ape_per_frame.csv")) if os.path.exists(os.path.join(d, "ape_per_frame.csv")) else None
        r["frames"] = read_csv(os.path.join(d, "frame_log.csv"))
        r["est"] = tum(os.path.join(d, "traj_est_all.txt"))
        r["gt"] = tum(os.path.join(d, "traj_gt_all.txt"))
    r["status"] = status(m, r.get("ape"))
    runs[(meth, sd)] = r
methods = [k for k in ("base", "ours", "ours_init", "ours_loss") if any(mm == k for mm, _ in runs)]
seed = sorted({sd for _, sd in runs})[0]


def g(meth, key, scale=1.0, fmt="{:.2f}"):
    r = runs.get((meth, seed))
    if r is None or r["m"] is None or r["m"].get(key) is None:
        return "—"
    return fmt.format(r["m"][key] * scale).replace(".", ",")


def val(meth, key):
    r = runs.get((meth, seed))
    return None if r is None or r["m"] is None else r["m"].get(key)


# ------------------------------------------------------------------ F1 ATE bars
fig, ax = plt.subplots(figsize=(6.4, 3.6))
xs = np.arange(2)
w = 0.36
for k, meth in enumerate(m for m in ("base", "ours") if m in methods):
    v = [val(meth, "ate_rmse_m"), val(meth, "ate_kf_m")]
    st = runs[(meth, seed)]["status"]
    for j, x in enumerate(v):
        if st == "FAIL" or x is None:
            ax.bar(xs[j] + (k - 0.5) * w, 100, w * 0.94, color="white", edgecolor=COLORS[meth], hatch="//")
            ax.text(xs[j] + (k - 0.5) * w, 60, "FAIL", ha="center", color=INK2)
        else:
            ax.bar(xs[j] + (k - 0.5) * w, 100 * x, w * 0.94, color=COLORS[meth], label=meth if j == 0 else None)
            ax.text(xs[j] + (k - 0.5) * w, 100 * x * 1.08, f"{100 * x:.2f}".replace(".", ","), ha="center", fontsize=9, color=GT_C)
for name, cm in REFS.get(seq, []):
    ax.axhline(cm, ls="--", lw=1, color=INK2)
    ax.text(1.62, cm * 1.05, f"{name} {cm} cm", fontsize=8, color=INK2, ha="right")
ax.set_yscale("log")
ax.set_ylim(0.5, 130)
ax.set_xticks(xs, ["ATE-all", "ATE-kf"])
ax.set_ylabel("cm (log)")
ax.set_title(f"F1 — ATE {seq}, seed {seed[4:]} (đường nét đứt: số báo cáo, giao thức khác)", fontsize=9, loc="left")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(figdir, "F1_ate_bar.png"))
plt.close(fig)

# ------------------------------------------------------------------ F2 APE over time + plane diagnostics
ours = runs.get(("ours", seed))
fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
for meth in ("base", "ours"):
    r = runs.get((meth, seed))
    if r is None or r.get("ape") is None:
        continue
    f = np.array([int(a["frame_id"]) for a in r["ape"]])
    e = np.array([float(a["ape_m"]) for a in r["ape"]]) * 100
    axs[0].plot(f, e, lw=2 if meth == "ours" else 1.6, color=COLORS[meth], label=meth)
if ours is not None and ours["m"] is not None:
    fr = ours["frames"]
    fid = np.array([int(a["frame_id"]) for a in fr])
    plane = np.array([a["init_source"] == "plane" for a in fr])
    for x in fid[plane]:
        axs[0].axvspan(x - 1, x + 1, color=OURS_C, alpha=0.10, lw=0)
    nma = np.array([float(a["n_matched_active"] or 0) for a in fr])
    smin = np.array([float(a["sigma_min"]) if a["sigma_min"] not in ("", None) else np.nan for a in fr])
    axs[1].step(fid, nma, where="mid", color=OURS_C, lw=1.4)
    axs[1].axhline(3, ls="--", lw=1, color=INK2)
    axs[1].set_ylabel("ghép active")
    axs[2].plot(fid, smin, color=OURS_C, lw=1.4)
    axs[2].axhline(0.3, ls="--", lw=1, color=INK2)
    axs[2].set_ylabel("σ_min")
axs[0].set_ylabel("APE (cm)")
axs[0].legend(frameon=False)
axs[0].set_title(f"F2 — APE theo frame, {seq} (nền xanh: frame khởi tạo bằng plane)", fontsize=9, loc="left")
axs[2].set_xlabel("frame (chỉ số dataset)")
fig.tight_layout()
fig.savefig(os.path.join(figdir, f"F2_ape_time_{seq}.png"))
plt.close(fig)


# ------------------------------------------------------------------ F3 / F5 top view on GT principal axes
def aligned(r):
    ref = PosePath3D(poses_se3=list(r["gt"][1]))
    est = PosePath3D(poses_se3=list(r["est"][1]))
    est = trajectory.align_trajectory(est, ref, correct_scale=False)
    return np.array(est.positions_xyz), np.array(ref.positions_xyz)


anyrun = next((r for r in runs.values() if r["m"] is not None), None)
if anyrun is not None:
    gt_xyz = anyrun["gt"][1][:, :3, 3]
    mu = gt_xyz.mean(0)
    _, _, Vt = np.linalg.svd(gt_xyz - mu)
    P = Vt[:2].T
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(*((gt_xyz - mu) @ P).T, color=GT_C, lw=2, label="GT")
    for meth in ("base", "ours"):
        r = runs.get((meth, seed))
        if r is None or r["m"] is None or r["m"].get("has_nan_pose"):
            continue
        e, _ = aligned(r)
        q = (e - mu) @ P
        ax.plot(*q.T, color=COLORS[meth], lw=1.8, label=meth)
    q = (gt_xyz - mu) @ P
    ax.plot(*q[0], "o", ms=8, color=GT_C, mfc="white")
    ax.plot(*q[-1], "s", ms=8, color=GT_C)
    ax.annotate("đầu", q[0], textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK2)
    ax.annotate("cuối", q[-1], textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK2)
    ax.set_aspect("equal")
    ax.set_xlabel("trục chính 1 của GT (m)")
    ax.set_ylabel("trục chính 2 (m)")
    ax.legend(frameon=False)
    ax.set_title(f"F3 — quỹ đạo nhìn từ trên (đã căn SE(3)), {seq}", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"F3_traj_{seq}.png"))
    plt.close(fig)

    # F5 map top view, same axes, colour = height along the 3rd principal axis
    maps = [(meth, runs[(meth, seed)]) for meth in ("base", "ours") if (meth, seed) in runs and runs[(meth, seed)]["m"]]
    if maps:
        fig, axs = plt.subplots(1, len(maps), figsize=(5.2 * len(maps), 4.8), squeeze=False)
        pts_all = {}
        for meth, r in maps:
            X = np.load(os.path.join(r["dir"], "map_xyz.npy"))
            X = X[np.linspace(0, len(X) - 1, min(len(X), 120000)).astype(int)]
            pts_all[meth] = X
        allq = np.concatenate([(X - mu) @ P for X in pts_all.values()])
        lo, hi = np.percentile(allq, 1, 0), np.percentile(allq, 99, 0)
        hz = np.concatenate([(X - mu) @ Vt[2] for X in pts_all.values()])
        zlo, zhi = np.percentile(hz, 2), np.percentile(hz, 98)
        for k, (meth, X) in enumerate(pts_all.items()):
            ax = axs[0, k]
            q = (X - mu) @ P
            sc = ax.scatter(q[:, 0], q[:, 1], c=(X - mu) @ Vt[2], s=0.2, cmap="viridis", vmin=zlo, vmax=zhi, rasterized=True)
            ax.plot(*((gt_xyz - mu) @ P).T, color=GT_C, lw=1.2)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_aspect("equal")
            ax.set_title(f"{meth}: {len(X)} điểm (lấy mẫu)", fontsize=9, loc="left")
        fig.colorbar(sc, ax=axs[0, -1], label="độ cao theo trục chính 3 (m)")
        fig.suptitle(f"F5 — map Gaussian nhìn từ trên, {seq} (đường đen: quỹ đạo GT)", fontsize=9, x=0.02, ha="left")
        fig.savefig(os.path.join(figdir, f"F5_map_top_{seq}.png"), bbox_inches="tight")
        plt.close(fig)

# ------------------------------------------------------------------ F4 render grid
rb, ro = runs.get(("base", seed)), runs.get(("ours", seed))
if rb and ro and rb["m"] and ro["m"]:
    fb = sorted(glob.glob(os.path.join(rb["dir"], "renders", "*.npz")))
    names = [os.path.basename(p) for p in fb if os.path.exists(os.path.join(ro["dir"], "renders", os.path.basename(p)))]
    if names:
        fig, axs = plt.subplots(len(names), 5, figsize=(13, 2.3 * len(names)), squeeze=False)
        cols = ["RGB GT", "RGB render base", "RGB render ours", "lỗi depth base", "lỗi depth ours"]
        for i, n in enumerate(names):
            a, b = np.load(os.path.join(rb["dir"], "renders", n)), np.load(os.path.join(ro["dir"], "renders", n))
            ims = [a["rgb_gt"], a["rgb_ren"], b["rgb_ren"]]
            errs = []
            for z in (a, b):
                dg, dr = z["depth_gt"].astype(np.float32), z["depth_ren"].astype(np.float32)
                errs.append(np.where(dg > 0, np.abs(dg - dr) * 100, np.nan))
            for j in range(5):
                ax = axs[i, j]
                ax.axis("off")
                if j < 3:
                    ax.imshow(ims[j])
                else:
                    im = ax.imshow(errs[j - 3], cmap="inferno", vmin=0, vmax=20)
                if i == 0:
                    ax.set_title(cols[j], fontsize=9)
            axs[i, 0].text(-12, 240, f"frame {int(n[:5])}", rotation=90, va="center", ha="right", fontsize=9, color=INK2)
        fig.colorbar(im, ax=axs[:, 3:].ravel().tolist(), label="lỗi depth (cm), thang 0–20", shrink=0.7)
        fig.suptitle(f"F4 — ảnh render từ map lúc track, {seq}", fontsize=10, x=0.02, ha="left")
        fig.savefig(os.path.join(figdir, f"F4_render_{seq}.png"), bbox_inches="tight")
        plt.close(fig)

# ------------------------------------------------------------------ REPORT.md
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip().count("\n") + 1
envv = {}
for meth in methods:
    so = os.path.join(runs[(meth, seed)]["dir"], "stdout.log")
    if os.path.exists(so):
        envv[meth] = open(so).readline().strip()
L = []
L.append(f"# E1 — Plane-anchor tracking (2DGS-SLAM) — số liệu tự sinh\n")
L.append(f"Commit nền: `{commit}` (+ {dirty} file chưa commit) · Branch: exp/e1-plane-tracking · Máy: RTX 5060 Ti 16 GB\n")
L.append(f"Runs: 1 sequence ({seq}) × {{{', '.join(methods)}}} × 1 seed ({seed}) · Config: configs/tum/e1/\n")


def delta(key):
    b, o = val("base", key), val("ours", key)
    if b is None or o is None:
        return "—"
    return f"{(o - b) / b * 100:+.1f} %".replace(".", ",")


L.append("\n## Bảng chính — ATE và RPE\n")
L.append("| Chỉ số | " + " | ".join(methods) + " | Δ (ours so base) |")
L.append("|---|" + "---|" * len(methods) + "---|")
rows = [("ATE-all (cm)", "ate_rmse_m", 100, "{:.2f}"), ("ATE-kf (cm)", "ate_kf_m", 100, "{:.2f}"),
        ("ATE-all theo repo (cm)", "ate_all_m_repo", 100, "{:.2f}"),
        ("RPE dịch, Δ=1 frame, RMSE (cm)", "rpe_trans_rmse_m_d1", 100, "{:.3f}"),
        ("RPE quay, Δ=1 frame, RMSE (°)", "rpe_rot_rmse_deg_d1", 1, "{:.3f}"),
        ("RPE dịch, Δ=15 frame (~1 s), RMSE (cm)", "rpe_trans_rmse_m_d15", 100, "{:.2f}"),
        ("RPE quay, Δ=15 frame (~1 s), RMSE (°)", "rpe_rot_rmse_deg_d15", 1, "{:.3f}"),
        ("Tỉ lệ frame APE > 30 cm (%)", "frac_frames_ape_gt_30cm", 100, "{:.1f}"),
        ("PSNR keyframe (dB)", "mean_psnr", 1, "{:.2f}")]
for name, key, sc, fm in rows:
    L.append(f"| {name} | " + " | ".join(g(m, key, sc, fm) for m in methods) + f" | {delta(key)} |")
L.append("| Trạng thái (§8) | " + " | ".join(runs[(m, seed)]["status"] for m in methods) + " | |")
L.append("\n![F1](figs/F1_ate_bar.png)\n")
L.append(f"![F2](figs/F2_ape_time_{seq}.png)\n")
L.append(f"![F3](figs/F3_traj_{seq}.png)\n")
L.append(f"\n## Ảnh render\n\n![F4](figs/F4_render_{seq}.png)\n\n![F5](figs/F5_map_top_{seq}.png)\n")
L.append("\n## Chi phí\n")
L.append("| Method | Tracking ms/frame | CAPE-lite ms/frame | Ghép + giải ms/frame | FPS toàn hệ | Đỉnh VRAM torch frontend (GiB) | Số Gaussian cuối | Số keyframe |")
L.append("|---|---|---|---|---|---|---|---|")
for m in methods:
    mm = runs[(m, seed)]["m"] or {}
    ms_ms = None if mm.get("ms_match_mean") is None else mm["ms_match_mean"] + mm.get("ms_solve_mean", 0)
    vr = os.path.join(runs[(m, seed)]["dir"], "vram.txt")
    smi = next((ln.split()[1] for ln in open(vr) if ln.startswith("nvidia_smi")), "—") if os.path.exists(vr) else "—"
    f = lambda x, fm="{:.1f}": "—" if x is None else fm.format(x).replace(".", ",")  # noqa: E731
    L.append(f"| {m} | {f(mm.get('ms_track_mean'))} | {f(mm.get('ms_plane_extract_mean'))} | {f(ms_ms)} | "
             f"{f(mm.get('fps'), '{:.3f}')} | {f(mm.get('torch_max_allocated_frontend_gib'), '{:.2f}')} (nvidia-smi đỉnh {smi} MiB) | "
             f"{mm.get('n_gaussians', '—')} | {mm.get('n_keyframes', '—')} |")
if ours and ours["m"]:
    om = ours["m"]
    L.append("\n## Chẩn đoán plane (ours)\n")
    L.append("| % frame khởi tạo bằng plane | Plane/frame | Ghép active/frame | Anchor cuối (active) | Giải pose bị loại | Vùng dải bị loại | Anchor nhìn từ sau |")
    L.append("|---|---|---|---|---|---|---|")
    f = lambda x, fm="{:.1f}": "—" if x is None else fm.format(x).replace(".", ",")  # noqa: E731
    L.append(f"| {f(om.get('pct_init_plane'))} | {f(om.get('planes_per_frame'), '{:.2f}')} | {f(om.get('matched_active_per_frame'), '{:.2f}')} | "
             f"{om.get('n_anchors_final')} ({om.get('n_anchors_active_final')}) | {om.get('n_init_rejected')} | {om.get('n_strip_rejected')} | {om.get('n_back_facing')} |")
    src = {}
    for a in ours["frames"]:
        src[a["init_source"]] = src.get(a["init_source"], 0) + 1
    L.append(f"\ninit_source (mọi frame, kể cả frame đầu): " + ", ".join(f"{k}: {v}" for k, v in sorted(src.items())))
if t7_json and os.path.exists(t7_json):
    t = json.load(open(t7_json))
    L.append(f"\n## T7/T8 — so pose từng frame với run code gốc `{os.path.basename(t['ref'])}` (60 frame)\n")
    L.append("| Run | ATE-all (cm) | Lệch pose lớn nhất (cm) | Lệch trung vị (cm) | Lệch frame cuối (cm) | Lệch quay lớn nhất (°) | Frame đầu tiên lệch > 0,1 mm |")
    L.append("|---|---|---|---|---|---|---|")
    ra = t["ref_ate_all_m"]
    L.append(f"| {os.path.basename(t['ref'])} (tham chiếu) | {100 * ra:.2f} | 0 | 0 | 0 | 0 | — |".replace(".", ","))
    for r in t["runs"]:
        L.append((f"| {os.path.basename(r['run'])} | {100 * r['ate_all_m']:.2f} | {100 * r['max_dt_m']:.2f} | {100 * r['median_dt_m']:.2f} | "
                  f"{100 * r['final_dt_m']:.2f} | {r['max_drot_deg']:.2f} | {r['first_frame_dt_gt_1e4']} |").replace(".", ","))
L.append("\n## Bảng thô\n")
for m in methods:
    L.append(f"- `{runs[(m, seed)]['dir']}`: {envv.get(m, '')}")
L.append("\n```json")
L.append(json.dumps({m: runs[(m, seed)]["m"] for m in methods}, indent=1, ensure_ascii=False))
L.append("```")
open(os.path.join(root, "REPORT.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:40]))
