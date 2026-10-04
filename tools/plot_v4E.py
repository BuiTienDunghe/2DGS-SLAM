"""plan v4 step E figures (CPU only): per-frame trajectory error of the TUM fr1/room runs after Umeyama alignment,
segment RMSE between loop events, and the e_dl durability series per event.

usage: CUDA_VISIBLE_DEVICES= python tools/plot_v4E.py --out DIR [--durability DIR/durability.jsonl] LABEL=RUN_DIR ...
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

COLORS = ["#1f77b4", "#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def load(rd):
    s = torch.load(os.path.join(rd, "final_state.pt"), map_location="cpu", weights_only=False)
    uids = sorted(u for u in s["poses"] if s["poses_gt"].get(u) is not None)
    P = np.stack([s["poses"][u].numpy() for u in uids])
    G = np.stack([s["poses_gt"][u].numpy() for u in uids])
    ev = []
    p = os.path.join(rd, "loop_events.jsonl")
    if os.path.exists(p):
        ev = [json.loads(l) for l in open(p) if l.strip()]
    return np.array(uids), P, G, ev, s["keyframe_uids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--durability", default=None)
    ap.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [r.split("=", 1) for r in a.runs]
    data = {}
    for i, (lab, rd) in enumerate(runs):
        if not os.path.exists(os.path.join(rd, "final_state.pt")):
            print("skip (no final_state):", lab, rd)
            continue
        data[lab] = (load(rd), COLORS[i % len(COLORS)])
    summary = {}
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.8))
    bounds = None
    for lab, ((uids, P, G, ev, kfs), col) in data.items():
        R, t = umeyama(P[:, :3, 3], G[:, :3, 3])
        err = np.linalg.norm(P[:, :3, 3] @ R.T + t - G[:, :3, 3], axis=1)
        kf = np.isin(uids, kfs)
        ev_u = sorted(e["cur_uid"] for e in ev)
        if bounds is None and ev_u:
            bounds = ev_u
        segs = [0] + ev_u + [int(uids.max()) + 1]
        seg = {}
        for s0, s1 in zip(segs[:-1], segs[1:]):
            m = (uids >= s0) & (uids < s1)
            if m.any():
                seg[f"{s0}-{s1 - 1}"] = 100 * float(np.sqrt((err[m] ** 2).mean()))
        summary[lab] = {"ate_all_cm": 100 * float(np.sqrt((err ** 2).mean())), "ate_kf_cm": 100 * float(np.sqrt((err[kf] ** 2).mean())),
                        "n_frames": int(len(uids)), "loops": [(e["cur_uid"], e["loop_uid"]) for e in ev], "segment_rmse_cm": seg}
        ax.plot(uids, err * 100, c=col, lw=1, label=f"{lab} (ATE {summary[lab]['ate_all_cm']:.2f} cm)")
    for u in (bounds or []):
        ax.axvline(u, c="k", lw=0.8, ls="--")
    ax.set_xlabel("frame")
    ax.set_ylabel("sai số vị trí sau căn chỉnh Umeyama [cm]")
    ax.set_title("TUM fr1/room: sai số từng frame (đường đứt = loop event có PGO)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "E_traj_error_v4.png"), dpi=120)
    plt.close(fig)
    if a.durability and os.path.exists(a.durability):
        rows = [json.loads(l) for l in open(a.durability) if l.strip()]
        by_run = {}
        for r in rows:
            by_run.setdefault(r["run"], []).append(r)
        states = ["rigid@k", "post@k", "pre@k+1", "final"]
        fig, axs = plt.subplots(1, max(1, len(by_run)), figsize=(4.6 * max(1, len(by_run)), 4.2), squeeze=False)
        for ax, (run, rs) in zip(axs[0], by_run.items()):
            for j, r in enumerate(sorted(rs, key=lambda x: x["event_id"])):
                ys = [None if not isinstance(r.get(s), dict) else r[s].get("median_mm") for s in states]
                xs = [i for i, y in enumerate(ys) if y is not None]
                ax.plot(xs, [ys[i] for i in xs], "o-", c=COLORS[j % len(COLORS)], label=f"event {r['event_id']} ({r['cur']}↔{r['loop']})")
            ax.set_xticks(range(len(states)))
            ax.set_xticklabels(states, fontsize=8)
            ax.set_ylabel("trung vị |gap| hai lớp [mm]")
            ax.set_title(run, fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle("Độ bền phần sửa: lỗi hai lớp trên Π*(k) tại các thời điểm sau event k (lớp theo thời điểm sinh)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "E_durability_v4.png"), dpi=120)
        plt.close(fig)
    json.dump(summary, open(os.path.join(a.out, "E_traj_summary_v4.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
