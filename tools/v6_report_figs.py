"""plan v6: figures of the core report, from the analysis JSON files.

usage: python tools/v6_report_figs.py --analysis results_m/analysis --out results_m/reports/figs
Reads A1/a1.json, A2/ate_split.json, A3*/pgo_oracle.json, A4/loader_chamfer.json, A5*/residual_radial.json and
Q14/track_stats.json; a figure whose input is missing is skipped. Static PNGs on the light chart surface.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reference palette of the dataviz method (light mode), categorical slots in their fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans",
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150,
})


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def tag(name):
    """run folder -> short label (20261004121409_R0 -> R0)."""
    return name.split("_", 1)[1] if "_" in name else name


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


def fig_iters(a1, stats, out):
    if not a1:
        return
    ref = np.concatenate([np.asarray(v["iters"]) for v in a1["iters"].values()])
    bins = np.arange(0, 125, 5) + 0.5
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    h, _ = np.histogram(ref, bins=bins)
    ax.bar(bins[:-1] + 2.5, 100 * h / ref.size, width=4.0, color=C[0], label=f"6 run tham chiếu ({ref.size} lần gọi)")
    r0 = next((v for k, v in (stats or {}).items() if tag(k) == "R0"), None)
    if r0 and r0.get("tracking"):
        it = np.asarray([p["n"] for p in r0["tracking"]["per_call"]])
        h0, _ = np.histogram(it, bins=bins)
        ax.plot(bins[:-1] + 2.5, 100 * h0 / it.size, color=C[1], marker="o", ms=4, label=f"R0 ({it.size} lần gọi)")
    ax.set_xlabel("Số vòng tracking của một lần gọi (trần 120)")
    ax.set_ylabel("% số lần gọi")
    ax.set_title("Hơn một phần ba số lần tracking chạm trần 120 vòng; phần còn lại dừng ở 75–115")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
    save(fig, out, "iters_hist.png")


def fig_ate_split(a2, out):
    if not a2:
        return
    names = list(a2)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7.2, 0.62 * len(names) + 2), 3.6))
    ax.axhspan(7.0, 9.0, color=GRID, alpha=0.6, lw=0, zorder=0, label="dải tham chiếu 7,0–9,0 cm (ATE mọi frame)")
    for i, (key, lab) in enumerate((("all", "mọi frame"), ("upto", "frame ≤ 1144"), ("after", "frame > 1144"))):
        v = [100 * (a2[n][key] or 0) for n in names]
        ax.bar(x + (i - 1) * 0.26, v, width=0.22, color=C[i], label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([tag(n) for n in names], rotation=35, ha="right")
    ax.set_ylabel("ATE RMSE (cm)")
    ax.set_title("ATE theo đoạn, cùng một phép căn SE(3) trên cả quỹ đạo")
    ax.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[1:] + h[:1], l[1:] + l[:1], ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8.5)
    save(fig, out, "ate_split.png")


def fig_oracle(a3, out):
    runs = {k: v for d in a3 for k, v in d.items() if v.get("variants")}
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5.4 * len(runs), 5.2), squeeze=False)
    for ax, (name, r) in zip(axes[0], runs.items()):
        V = r["variants"]
        labs = list(V)
        v = [100 * V[k]["all"] for k in labs]
        extra = ["(extra)" in k or k == "P-none" for k in labs]
        y = np.arange(len(labs))[::-1]
        ax.barh(y, v, height=0.42, color=[MUTED if e else C[0] for e in extra])
        p0 = 100 * V["P0"]["all"]
        ax.axvline(p0, color=INK2, lw=1)
        for yi, vi in zip(y, v):
            ax.text(vi + 0.1, yi, f"{vi:.2f}", va="center", ha="left", fontsize=8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([k.replace(" (extra)", "") for k in labs], fontsize=8)
        ax.set_xlabel("ATE mọi frame sau khi tối ưu lại đồ thị (cm)")
        ax.set_title(f"{tag(name)}: P0 = {p0:.2f} cm")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(v) * 1.12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[0]), plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    fig.legend(handles, ["biến thể trong plan", "thêm ngoài plan"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    save(fig, out, "oracle_bars.png")


def fig_time(stats, out):
    runs = {k: v for k, v in (stats or {}).items() if v.get("timing")}
    if not runs:
        return
    parts = [("t_track", "tracking"), ("t_loop", "loop detection"), ("t_kf", "keyframe"), ("t_wait", "chờ backend"),
             ("t_sync", "nhận bản đồ"), ("t_eval", "eval pose"), ("t_data", "đọc dữ liệu")]
    names = list(runs)
    fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(names) + 1.9))
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for i, (key, lab) in enumerate(parts):
        v = []
        for n in names:
            t = runs[n]["timing"]
            s = t[key]["sum_s"] + (t["t_eval_stall"]["sum_s"] if key == "t_eval" else 0.0) + (t["t_pad"]["sum_s"] if key == "t_wait" else 0.0)
            v.append(s / t["n_frames"])
        v = np.asarray(v)
        ax.barh(y, v, left=left, height=0.38, color=C[i], label=lab, edgecolor=SURF, linewidth=2)
        left += v
    for yi, tot in zip(y, left):
        ax.text(tot + 0.03, yi, f"{tot:.2f} s", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([tag(n) for n in names])
    ax.set_xlabel("Thời gian trung bình mỗi frame (giây)")
    ax.set_xlim(0, left.max() * 1.12)
    ax.set_title("Một frame mất thời gian ở đâu")
    ax.grid(axis="y", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=8.5)
    save(fig, out, "time_breakdown.png")


def fig_conv(stats, out):
    r0 = next((v for k, v in (stats or {}).items() if tag(k) == "R0" and v.get("tracking")), None)
    if r0 is None:
        r0 = next((v for v in (stats or {}).values() if v.get("tracking")), None)
    if r0 is None:
        return
    per = r0["tracking"]["per_call"]
    ks = np.arange(5, 125, 5)
    kmin = np.asarray([p["k_min"] for p in per])
    kstar = np.asarray([p["k_star"] for p in per])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    a_ = [100 * (kmin <= k).mean() for k in ks]
    s_ = [100 * (kstar <= k).mean() for k in ks]
    ax.plot(ks, a_, color=C[0], marker="o", ms=4, label="đủ vòng để tới nơi (k_min ≤ k)")
    ax.plot(ks, s_, color=C[1], marker="o", ms=4, label="pose đã yên (k* ≤ k)")
    ax.set_xlabel("Vòng k")
    ax.set_ylabel("% số lần tracking")
    ax.set_ylim(0, 104)
    ax.set_title("Tới nơi sớm, yên muộn")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5)
    T = r0["tracking"]
    rows = [("mọi lần", T["split_all"]), ("dừng sớm", T["split_early_stopped"]), ("chạm trần", T["split_capped"])]
    rows = [(l, s) for l, s in rows if s]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for i, (key, lab) in enumerate((("travel", "đi đường"), ("refine", "tinh chỉnh"), ("tail", "đuôi thừa"))):
        v = np.asarray([100 * s[key] for _, s in rows])
        bx.barh(y, v, left=left, height=0.5, color=C[i], label=lab, edgecolor=SURF, linewidth=2)
        for yi, li, vi in zip(y, left, v):
            if vi >= 12:
                bx.text(li + vi / 2, yi, f"{vi:.0f}%", va="center", ha="center", fontsize=8.5, color="#ffffff" if i != 2 else INK)
        left += v
    bx.set_yticks(y)
    bx.set_yticklabels([f"{l}\n({s['n_calls']} lần)" for l, s in rows], fontsize=8.5)
    bx.set_xlabel("% tổng số vòng")
    bx.set_xlim(0, 100)
    bx.set_title("Số vòng được tiêu vào đâu")
    bx.grid(axis="y", visible=False)
    bx.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5)
    save(fig, out, "conv_at_k.png")


def fig_loader(a4, out):
    if not a4 or "radial_shift" not in a4:
        return
    rs = a4["radial_shift"]
    xc = [np.mean(r) for r in rs["rings"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.axhline(0, color=AXIS, lw=1)
    exp = [-v["undistortion_shift_px"] for v in rs["A"]]
    ax.plot(xc, exp, color=MUTED, lw=1.2, label="− độ dịch của phép undistort")
    for i, (m, lab) in enumerate((("A", "A: RGB undistort, depth thô (loader gốc)"), ("raw", "ảnh thô (chế độ B, R)"), ("D", "D: undistort cả hai"))):
        ax.plot(xc, [v["best_shift_px"] for v in rs[m]], color=C[i], marker="o", ms=5, label=lab)
    ax.set_xlabel("Bán kính tới tâm ảnh (px)")
    ax.set_ylabel("Độ dịch hướng kính khớp nhất (px)")
    ax.set_title("Loader gốc đặt biên depth lệch ra ngoài biên màu, lệch tăng theo bán kính")
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, out, "loader_chamfer.png")


def fig_residual(a5, out):
    if not a5:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axhline(0, color=AXIS, lw=1)
    seen = set()
    for name, p in a5.items():
        ld = p["loader"]
        if ld.get("depth_undistort"):
            i, lab, lw = 1, "chế độ D", 2.4
        elif not ld["distorted"]:
            i, lab, lw = 2, "chế độ R", 2.4
        else:
            i, lab, lw = 0, "chế độ A (loader gốc)", 1.3
        e = np.asarray(p["bin_edges_px"])
        xc = 0.5 * (e[:-1] + e[1:])
        ax.plot(xc, 1e3 * np.asarray(p["mean"]), color=C[i], lw=lw, alpha=0.9 if i else 0.75, label=None if lab in seen else lab)
        seen.add(lab)
    ax.set_xlabel("Bán kính tới tâm ảnh (px), 10 bin cùng số pixel")
    ax.set_ylabel("(D render − D vào) / D vào  (×10⁻³)")
    ax.set_title("Residual depth có dấu của bản đồ theo bán kính ảnh")
    ax.legend(loc="lower left")
    save(fig, out, "residual_radial.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results_m/analysis")
    ap.add_argument("--out", default="results_m/reports/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A = a.analysis
    stats = load(os.path.join(A, "Q14", "track_stats.json"))
    fig_iters(load(os.path.join(A, "A1", "a1.json")), stats, a.out)
    fig_ate_split(load(os.path.join(A, "A2", "ate_split.json")), a.out)
    fig_oracle([load(p) for p in sorted(glob.glob(os.path.join(A, "A3*", "pgo_oracle.json")))], a.out)
    fig_time(stats, a.out)
    fig_conv(stats, a.out)
    fig_loader(load(os.path.join(A, "A4", "loader_chamfer.json")), a.out)
    a5 = {}
    for p in sorted(glob.glob(os.path.join(A, "A5*", "residual_radial.json"))):
        a5.update(load(p))
    fig_residual(a5, a.out)


if __name__ == "__main__":
    main()
