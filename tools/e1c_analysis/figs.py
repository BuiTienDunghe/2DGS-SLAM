"""PLAN E1c §10 figures + cost table for decision/final reports.
usage: python tools/e1c_analysis/figs.py OUT_DIR LABEL=RUN_DIR [LABEL=RUN_DIR ...]
  LABEL in {base, E1, E1b, E1c, E1d} (+ optional suffix like 'E1c s1'); colours fixed by PLAN §10.2.
Writes OUT_DIR/figs/{F6_sigma_nobs.png, F7_efree_eobs.png, F8_raw_error.png, F10_S5_anchor_changes.png} and
OUT_DIR/cost_table.md. Numbers from seg_metrics.json, e1_metrics.json, dec_log.csv, frame_log.csv, ape_per_frame.csv.
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "tools", "e1_analysis"))
from common import INK2, SEGMENTS, mpl_style, shade_segments  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

COL = {"base": "#8f8e88", "E1": "#eb6834", "E1b": "#6250d6", "E1c": "#2a78d6", "E1d": "#1baf7a"}
out = sys.argv[1]
runs = [a.split("=", 1) for a in sys.argv[2:]]
os.makedirs(os.path.join(out, "figs"), exist_ok=True)
mpl_style(plt)


def colour(label):
    return COL.get(label.split()[0], "#52514e")


def seg(run):
    p = os.path.join(run, "seg_metrics.json")
    if not os.path.exists(p):
        p = os.path.join(os.path.expanduser("~/2DGS-SLAM/results_exp/e1c/reused"),
                         f"{os.path.basename(os.path.dirname(run))}_{os.path.basename(run)}", "seg_metrics.json")
    return json.load(open(p))


def raw_err(run):
    def tum(p):
        a = np.loadtxt(p)
        return a[:, 1:4]
    ids = np.array([int(r["frame_id"]) for r in csv.DictReader(open(os.path.join(run, "ape_per_frame.csv")))])
    return ids, np.linalg.norm(tum(os.path.join(run, "traj_est_all.txt")) - tum(os.path.join(run, "traj_gt_all.txt")), axis=1) * 100


# F8 unaligned error, all methods; plane-init frames marked for the last plane method
fig, ax = plt.subplots(figsize=(10, 4))
shade_segments(ax)
for lab, run in runs:
    ids, e = raw_err(run)
    ax.plot(ids, e, color=colour(lab), lw=1.8 if lab.startswith("E1c") or lab.startswith("E1d") else 1.4, label=lab)
    fl = os.path.join(run, "frame_log.csv")
    if lab.split()[0] in ("E1c", "E1d"):
        pi = [int(r["frame_id"]) for r in csv.DictReader(open(fl)) if r.get("init_source") == "plane"]
        ax.plot(pi, np.full(len(pi), -2.0), "|", color=colour(lab), ms=6)
ax.set_ylabel("‖p_est − p_GT‖ (cm), không căn")
ax.set_xlabel("frame (chỉ số dataset); vạch dưới trục: frame khởi tạo bằng plane (E1c/E1d)")
ax.legend(frameon=False, loc="upper left", ncol=len(runs))
ax.set_title("F8 — sai số không căn theo frame (mỗi method 1 lần chạy)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(out, "figs", "F8_raw_error.png"))
plt.close(fig)

# F7 e_free / e_obs per segment (V_scene), one stacked bar per method
fig, ax = plt.subplots(figsize=(10, 3.8))
names = [s[0] for s in SEGMENTS]
w = 0.8 / len(runs)
for k, (lab, run) in enumerate(runs):
    m = seg(run)
    ef = [m["segments"][s].get("e_free_rms_cm") or 0 for s in names]
    eo = [m["segments"][s].get("e_obs_rms_cm") or 0 for s in names]
    x = np.arange(len(names)) + (k - (len(runs) - 1) / 2) * w
    ax.bar(x, eo, w * 0.94, color=colour(lab), alpha=0.45)
    ax.bar(x, ef, w * 0.94, bottom=eo, color=colour(lab), label=lab)
ax.set_xticks(np.arange(len(names)), [f"{s[0]}\n{s[1]}–{s[2]}" for s in SEGMENTS])
ax.set_ylabel("RMS cm/frame (nhạt: e_obs, đậm: e_free)")
ax.legend(frameon=False, ncol=len(runs))
ax.set_title("F7 — sai số chuyển động theo hướng quan sát được / tự do (V_scene)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(out, "figs", "F7_efree_eobs.png"))
plt.close(fig)

# F6 sigma_t and n_obs_t per frame for each decoupled run (dec_log.csv)
dec_runs = [(lab, run) for lab, run in runs if os.path.exists(os.path.join(run, "dec_log.csv"))]
if dec_runs:
    fig, axs = plt.subplots(len(dec_runs), 1, figsize=(10, 3.2 * len(dec_runs)), squeeze=False)
    for ax, (lab, run) in zip(axs[:, 0], dec_runs):
        shade_segments(ax)
        rows = list(csv.DictReader(open(os.path.join(run, "dec_log.csv"))))
        f = [int(r["frame_id"]) for r in rows]
        for i, c in enumerate(("#0b0b0b", "#52514e", "#a3a29c")):
            ax.plot(f, [float(r[f"sig_t{i}"]) for r in rows], color=c, lw=1.0, label=f"σ_t,{i + 1}")
        ax.axhline(0.3, ls="--", color=INK2, lw=1)
        ax2_y = [int(r["n_obs_t"]) / 3 * 1.5 for r in rows]
        ax.step(f, ax2_y, where="mid", color=colour(lab), lw=1.2, label="n_obs_t (×0,5)")
        ax.set_ylabel("σ_t")
        ax.set_title(f"F6 — {lab}: σ của khối tịnh tiến và n_obs_t theo frame (ngưỡng 0,3)", fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    axs[-1, 0].set_xlabel("frame (chỉ số dataset)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figs", "F6_sigma_nobs.png"))
    plt.close(fig)

# F10 S5: APE per frame with vertical lines where the matched anchor ids change
fig, ax = plt.subplots(figsize=(10, 3.4))
for lab, run in runs:
    fl = list(csv.DictReader(open(os.path.join(run, "frame_log.csv"))))
    if "pairs" not in fl[0] or not lab.split()[0] in ("E1c", "E1d", "E1b"):
        continue
    ape = {int(r["frame_id"]): float(r["ape_m"]) * 100 for r in csv.DictReader(open(os.path.join(run, "ape_per_frame.csv")))}
    s5 = [r for r in fl if 481 <= int(r["frame_id"]) <= 790]
    ax.plot([int(r["frame_id"]) for r in s5], [ape[int(r["frame_id"])] for r in s5], color=colour(lab), lw=1.5, label=lab)
    prev = None
    for r in s5:
        ids = frozenset(p.split(":")[1] for p in r["pairs"].split(";") if p)
        if prev is not None and ids != prev:
            ax.axvline(int(r["frame_id"]), color=colour(lab), alpha=0.15, lw=1)
        prev = ids
ax.set_xlabel("frame (S5)")
ax.set_ylabel("APE (cm)")
ax.legend(frameon=False)
ax.set_title("F10 — S5: APE và các frame đổi anchor_id (vạch dọc)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(out, "figs", "F10_S5_anchor_changes.png"))
plt.close(fig)

# cost table
L = ["| Run | Tracking ms/frame | CAPE-lite ms/frame | obs_basis ms/frame | edge ms/frame | FPS | VRAM torch frontend (GiB) | Gaussian | Keyframe |",
     "|---|---|---|---|---|---|---|---|---|"]
for lab, run in runs:
    em = json.load(open(os.path.join(run, "e1_metrics.json")))
    tr = list(csv.DictReader(open(os.path.join(run, "timing.csv"))))

    def mean(k):
        v = [float(r[k]) for r in tr if r.get(k) not in (None, "")]
        return f"{np.mean(v):.1f}".replace(".", ",") if v else "—"
    L.append(f"| {lab} | {mean('ms_track')} | {mean('ms_plane_extract')} | {mean('obs_basis_ms')} | {mean('edge_ms')} | "
             f"{em['fps']:.3f} | {em.get('torch_max_allocated_frontend_gib', float('nan')):.2f} | {em['n_gaussians']} | {em['n_keyframes']} |".replace(".", ","))
open(os.path.join(out, "cost_table.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
