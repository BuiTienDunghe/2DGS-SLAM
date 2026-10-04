"""CPU-only figures for the baseline runs: trajectories vs GT (Umeyama-aligned), per-frame error, det(R) drift.

usage: CUDA_VISIBLE_DEVICES= python tools/plot_baseline_runs.py OUT_DIR
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.expanduser("~/2DGS-SLAM/results_exp")
RUNS = [
    ("TUM fr1/room — TF32 (lỗi)", "tum/room/20261002004114_rigidTF32_s0", "#d62728"),
    ("TUM fr1/room — fp32 (run #1)", "tum/room/20261002013837_rigid_s0", "#1f77b4"),
    ("Replica office2 — fp32 (run #2)", "replica/office2/20261002024857_rigid_s0", "#2ca02c"),
]


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    t = mu_d - R @ mu_s
    return R, t


def load(rel):
    s = torch.load(os.path.join(ROOT, rel, "final_state.pt"), map_location="cpu", weights_only=False)
    uids = sorted(u for u in s["poses"] if s["poses_gt"].get(u) is not None)
    P = np.stack([s["poses"][u].numpy() for u in uids])
    G = np.stack([s["poses_gt"][u].numpy() for u in uids])
    ev = []
    p = os.path.join(ROOT, rel, "loop_events.jsonl")
    if os.path.exists(p):
        ev = [json.loads(l) for l in open(p) if l.strip()]
    return np.array(uids), P, G, ev, s["keyframe_uids"]


def main(out):
    os.makedirs(out, exist_ok=True)
    data = {name: (load(rel), col) for name, rel, col in RUNS}
    summary = {}
    # 1. trajectories
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (name, ((uids, P, G, ev, kfs), col)) in zip(axs, data.items()):
        R, t = umeyama(P[:, :3, 3], G[:, :3, 3])
        est = P[:, :3, 3] @ R.T + t
        gt = G[:, :3, 3]
        err = np.linalg.norm(est - gt, axis=1)
        kf_mask = np.isin(uids, kfs)
        summary[name] = {"ate_all_cm": 100 * float(np.sqrt((err ** 2).mean())),
                         "ate_kf_cm": 100 * float(np.sqrt((err[kf_mask] ** 2).mean())), "n_frames": int(len(uids)),
                         "n_loop_events": len(ev), "loops": [(e["cur_uid"], e["loop_uid"]) for e in ev]}
        ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1.2, label="Ground truth")
        ax.plot(est[:, 0], est[:, 1], "-", c=col, lw=1.2, label="Ước lượng (đã căn chỉnh)")
        for e in ev:
            for u, mk in ((e["cur_uid"], "o"), (e["loop_uid"], "s")):
                i = np.searchsorted(uids, u)
                if i < len(uids) and uids[i] == u:
                    ax.plot(est[i, 0], est[i, 1], mk, c="orange", ms=7, mec="k")
        ax.set_title(f"{name}\nATE all = {summary[name]['ate_all_cm']:.2f} cm · {len(ev)} loop event", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Quỹ đạo nhìn từ trên xuống (cam = loop event: tròn = frame hiện tại, vuông = frame cũ)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "baseline_trajectories.png"), dpi=120)
    plt.close(fig)
    # 2. per-frame error + det(R)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
    for name, ((uids, P, G, ev, kfs), col) in data.items():
        R, t = umeyama(P[:, :3, 3], G[:, :3, 3])
        err = np.linalg.norm(P[:, :3, 3] @ R.T + t - G[:, :3, 3], axis=1)
        frac = uids / uids.max()
        axs[0].semilogy(frac, np.maximum(err * 1000, 1e-2), c=col, lw=1, label=name)
        det = np.linalg.det(P[:, :3, :3])
        axs[1].plot(frac, det, c=col, lw=1.2, label=name)
        summary[name]["det_min"] = float(det.min())
        summary[name]["det_max"] = float(det.max())
    axs[0].set_xlabel("vị trí trong chuỗi (0 = đầu, 1 = cuối)")
    axs[0].set_ylabel("sai số vị trí từng frame [mm] (log)")
    axs[0].set_title("Sai số quỹ đạo theo thời gian")
    axs[0].legend(fontsize=8)
    axs[1].axhline(1.0, c="k", lw=0.8, ls="--")
    axs[1].set_xlabel("vị trí trong chuỗi")
    axs[1].set_ylabel("det(R) của pose camera (đúng phải = 1)")
    axs[1].set_title("Ma trận quay bị méo dần khi chạy TF32")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "baseline_error_det.png"), dpi=120)
    plt.close(fig)
    json.dump(summary, open(os.path.join(out, "baseline_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
