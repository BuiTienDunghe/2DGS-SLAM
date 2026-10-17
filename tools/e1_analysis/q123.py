"""E1 analysis Q1–Q3 (per-segment RPE / local ATE / motion ratio, iters_done, loss balance) + F6–F8.
usage: python tools/e1_analysis/q123.py      (reads results_exp/e1, writes results_exp/e1/analysis/)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BASE_C, INK2, METHODS, OURS_C, OUT, SEGMENTS, fnum, load_run, local_ate, motion_ratio,  # noqa: E402
                    mpl_style, rmse, rpe_pairs, shade_segments)

os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
mpl_style(plt)
COL = {"base": BASE_C, "ours": OURS_C}
lam = yaml.safe_load(open("configs/tum/e1/e1_common.yaml"))["PlaneTracking"]["lambda_plane"]
runs = {m: load_run(m) for m in METHODS}
res = {"lambda_plane": lam, "segments": [s[0] for s in SEGMENTS], "q1": {}, "q2": {}, "q3": {}, "check": {}}

# cross-check: full-sequence RPE from this implementation vs e1_metrics.json (evo)
for m in METHODS:
    for d in (1, 15):
        p = rpe_pairs(runs[m], d)
        res["check"][f"{m}_d{d}"] = dict(
            mine_trans_cm=100 * rmse([x[2] for x in p]), evo_trans_cm=100 * runs[m]["metrics"][f"rpe_trans_rmse_m_d{d}"],
            mine_rot_deg=rmse([x[3] for x in p]), evo_rot_deg=runs[m]["metrics"][f"rpe_rot_rmse_deg_d{d}"])

# ---------------------------------------------------------------- Q1
pairs = {m: {d: rpe_pairs(runs[m], d) for d in (1, 15)} for m in METHODS}
slide15 = {m: rpe_pairs(runs[m], 15, sliding=True) for m in METHODS}
ratio = {m: motion_ratio(runs[m]) for m in METHODS}
ours_flog = runs["ours"]["flog"]
for name, a, b in SEGMENTS:
    row = {"frames": [a, b]}
    smin = [fnum(r["sigma_min"]) for f, r in ours_flog.items() if a <= f <= b]
    row["sigma_min_median_ours"] = float(np.nanmedian(smin)) if smin else None
    row["n_tracked"] = int(sum(1 for f in runs["ours"]["ids"] if a <= f <= b))
    row["pct_init_plane_ours"] = 100.0 * np.mean([r["init_source"] == "plane" for f, r in ours_flog.items() if a <= f <= b and f > 0])
    for m in METHODS:
        for d in (1, 15):
            sel = [x for x in pairs[m][d] if a <= x[0] <= b and a <= x[1] <= b]
            row[f"{m}_rpe_trans_cm_d{d}"] = 100 * rmse([x[2] for x in sel])
            row[f"{m}_rpe_rot_deg_d{d}"] = rmse([x[3] for x in sel])
            row[f"{m}_n_pairs_d{d}"] = len(sel)
        sel = [x for x in slide15[m] if a <= x[0] <= b and a <= x[1] <= b]
        row[f"{m}_rpe_trans_cm_d15_slide"] = 100 * rmse([x[2] for x in sel])
        row[f"{m}_rpe_rot_deg_d15_slide"] = rmse([x[3] for x in sel])
        row[f"{m}_n_pairs_d15_slide"] = len(sel)
        row[f"{m}_ate_local_cm"] = 100 * local_ate(runs[m], a, b)
        fk, de, dg = ratio[m]
        s = (fk >= a) & (fk <= b)
        # the pair (k-1, k) belongs to the segment only if frame k-1 is in it too
        prev = np.concatenate([[runs[m]["ids"][0]], runs[m]["ids"][:-1]])[1:]
        s &= (prev >= a)
        keep = s & (dg >= 0.002)
        r_k = de[keep] / dg[keep]
        row[f"{m}_r_median"] = float(np.median(r_k)) if keep.any() else None
        row[f"{m}_r_mean"] = float(np.mean(r_k)) if keep.any() else None
        row[f"{m}_path_ratio"] = float(de[s].sum() / dg[s].sum()) if s.any() else None
        row[f"{m}_n_r"] = int(keep.sum())
    res["q1"][name] = row

# ---------------------------------------------------------------- Q1 extra: unaligned position error
# The run starts at the GT pose of frame 0 (initialize() uses the GT pose), so est and GT share one world
# frame and ||p_est - p_gt|| needs no alignment. It shows where error is created, unlike the global SE(3) fit.
res["q1_raw"] = {}
raw = {}
for m in METHODS:
    e = np.linalg.norm(runs[m]["est"][:, :3, 3] - runs[m]["gt"][:, :3, 3], axis=1) * 100
    raw[m] = e
    for name, a, b in SEGMENTS:
        s = (runs[m]["ids"] >= a) & (runs[m]["ids"] <= b)
        v = e[s]
        res["q1_raw"].setdefault(name, {})[m] = dict(start=float(v[0]), end=float(v[-1]), median=float(np.median(v)),
                                                     min=float(v.min()), max=float(v.max()), change=float(v[-1] - v[0]))

# ---------------------------------------------------------------- Q2
for m in METHODS:
    fl = runs[m]["flog"]
    fk, de, dg = ratio[m]
    rk = {int(f): (e / g if g >= 0.002 else np.nan) for f, e, g in zip(fk, de, dg)}
    out = {}
    for name, a, b in SEGMENTS + [("all", 0, 10 ** 6)]:
        it = [(f, fnum(r["iters_done"])) for f, r in fl.items() if a <= f <= b and r["iters_done"] != ""]
        its = np.array([x for _, x in it])
        xs = [(x, rk.get(f, np.nan)) for f, x in it if not np.isnan(rk.get(f, np.nan))]
        rho, pv = spearmanr([x for x, _ in xs], [y for _, y in xs]) if len(xs) > 5 else (np.nan, np.nan)
        out[name] = dict(iters_median=float(np.median(its)) if len(its) else None,
                         pct_lt_120=100.0 * float(np.mean(its < 120)) if len(its) else None,
                         spearman_iters_vs_r=float(rho), spearman_p=float(pv), n=len(xs))
    res["q2"][m] = out

# ---------------------------------------------------------------- Q3
fl = runs["ours"]["flog"]
for name, a, b in SEGMENTS + [("all", 0, 10 ** 6)]:
    rows = [r for f, r in fl.items() if a <= f <= b and r["loss_render_final"] != ""]
    lp = np.array([fnum(r["loss_plane_final"]) for r in rows])
    lr = np.array([fnum(r["loss_render_final"]) for r in rows])
    ok = ~np.isnan(lp)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else None  # noqa: E731
    ratio_pr = lam * lp[ok] / lr[ok]
    base_lr = np.array([fnum(r["loss_render_final"]) for f, r in runs["base"]["flog"].items()
                        if a <= f <= b and r["loss_render_final"] != ""])
    res["q3"][name] = dict(n=len(rows), n_with_plane=int(ok.sum()),
                           plane_med=q(lp[ok], 50), plane_p90=q(lp[ok], 90),
                           render_med=q(lr, 50), render_p90=q(lr, 90),
                           ratio_med=q(ratio_pr, 50), ratio_p90=q(ratio_pr, 90),
                           base_render_med=q(base_lr, 50), base_render_p90=q(base_lr, 90))

json.dump(res, open(os.path.join(OUT, "q123.json"), "w"), indent=1)

# ---------------------------------------------------------------- F6
fig, axs = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
x = np.arange(len(SEGMENTS))
w = 0.38
for col, d in enumerate((1, 15)):
    for row, (key, unit) in enumerate((("rpe_trans_cm", "cm"), ("rpe_rot_deg", "°"))):
        ax = axs[row, col]
        for k, m in enumerate(METHODS):
            v = [res["q1"][s[0]][f"{m}_{key}_d{d}"] for s in SEGMENTS]
            ax.bar(x + (k - 0.5) * w, v, w * 0.94, color=COL[m], label=m)
        ax.set_ylabel(f"RPE {'dịch' if row == 0 else 'quay'} ({unit})")
        ax.set_title(f"Δ = {d} frame", fontsize=9, loc="left")
        ax.set_xticks(x, [f"{s[0]}\n{s[1]}–{s[2]}" for s in SEGMENTS])
axs[0, 0].legend(frameon=False)
fig.suptitle("F6 — RPE theo đoạn (RMSE, chỉ cặp nằm trọn trong đoạn), fr3_snt_far seed 0", fontsize=10, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figs", "F6_rpe_segments.png"))
plt.close(fig)

# ---------------------------------------------------------------- F7
fig, axs = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
for ax in axs:
    shade_segments(ax)
for m in METHODS:
    fk, de, dg = ratio[m]
    r = np.where(dg >= 0.002, de / np.maximum(dg, 1e-9), np.nan)
    sm = np.array([np.nanmean(r[max(0, i - 4): i + 5]) for i in range(len(r))])
    axs[0].plot(fk, sm, color=COL[m], lw=1.8 if m == "ours" else 1.5, label=m)
    it = [(f, fnum(rr["iters_done"])) for f, rr in sorted(runs[m]["flog"].items()) if rr["iters_done"] != ""]
    axs[1].plot([f for f, _ in it], [v for _, v in it], color=COL[m], lw=1.0, label=m)
axs[0].axhline(1.0, ls="--", lw=1, color=INK2)
axs[0].set_ylabel("r_k (trượt 9 frame)")
axs[0].legend(frameon=False, loc="upper right")
axs[1].set_ylabel("iters_done")
axs[1].set_xlabel("frame (chỉ số dataset)")
fig.suptitle("F7 — tỉ lệ chuyển động ước lượng/GT và số vòng tracking", fontsize=10, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figs", "F7_motion_ratio_iters.png"))
plt.close(fig)

# ---------------------------------------------------------------- F8
fig, ax = plt.subplots(figsize=(10, 3.8))
shade_segments(ax)
f = sorted(fl)
lp = [fnum(fl[i]["loss_plane_final"]) for i in f]
lr = [fnum(fl[i]["loss_render_final"]) for i in f]
ax.plot(f, [lam * v for v in lp], color=OURS_C, lw=1.4, label=f"λ_plane × loss_plane (λ = {lam})")
ax.plot(f, lr, color=INK2, lw=1.4, label="loss_render")
ax.set_yscale("log")
ax.set_ylabel("loss cuối (log)")
ax.set_xlabel("frame (chỉ số dataset)")
ax.legend(frameon=False, loc="upper left")
ax.set_title("F8 — lực của plane_loss so với loss render (ours)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figs", "F8_loss_balance.png"))
plt.close(fig)
# ---------------------------------------------------------------- F10 unaligned error
fig, ax = plt.subplots(figsize=(10, 3.8))
shade_segments(ax)
for m in METHODS:
    ax.plot(runs[m]["ids"], raw[m], color=COL[m], lw=1.8 if m == "ours" else 1.5, label=m)
ax.set_ylabel("‖p_est − p_GT‖ (cm), không căn")
ax.set_xlabel("frame (chỉ số dataset)")
ax.legend(frameon=False, loc="upper left")
ax.set_title("F10 — sai số vị trí không căn chỉnh (run bắt đầu đúng ở pose GT của frame 0)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figs", "F10_raw_error.png"))
plt.close(fig)
print(json.dumps({k: res[k] for k in ("check", "q1_raw")}, indent=1))
print(json.dumps(res["q1"], indent=1)[:3000])
