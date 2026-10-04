"""P2: reliability S_i and node sets for every dump of the tuning runs + ablation (§6 P2).

usage: python tools/build_nodes.py --out results_exp/reports/P2 [--no-ablation] RUN_DIR [RUN_DIR ...]
Metrics: coverage (same-scan node within 1.5 r), scan separation (mixed voxels with >= 2 nodes),
node noise (distance of a reliable node to the plane of its members), counts, time.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform.dump import EventState, list_dumps, load_dump  # noqa: E402
from deform.nodes import build_edges, build_nodes, systematic_nodes  # noqa: E402
from deform.pipeline import delta_T_dict, delta_t_frames, inp_from_dump  # noqa: E402
from deform.reliability import compute_reliability  # noqa: E402
from deform.render_utils import DEV, make_cam  # noqa: E402
from deform.rigid import apply_rigid  # noqa: E402


@torch.no_grad()
def coverage(x, t0, alpha, g, tn, r, dt, chunk=8192):
    sel = torch.nonzero(alpha > 0.5).reshape(-1)
    hit = torch.zeros(len(sel), dtype=torch.bool, device=DEV)
    for s in range(0, len(sel), chunk):
        i = sel[s:s + chunk]
        d = torch.cdist(x[i], g)
        ok = (d <= 1.5 * r) & ((t0[i].float()[:, None] - tn[None].float()).abs() <= dt)
        hit[s:s + chunk] = ok.any(1)
    return float(hit.float().mean()) if len(sel) else None


@torch.no_grad()
def separation(nodes, active, x, r, alpha):
    cand = nodes["cand"]
    vk = torch.floor(x[cand] / r).long()
    vk = vk - vk.min(0).values
    span = vk.max(0).values + 1
    key = (vk[:, 0] * span[1] + vk[:, 1]) * span[2] + vk[:, 2]
    act = active[cand]
    uk, inv = torch.unique(key, return_inverse=True)
    has_old = torch.zeros(len(uk), dtype=torch.bool, device=DEV).index_fill_(0, inv[~act], True)
    has_new = torch.zeros(len(uk), dtype=torch.bool, device=DEV).index_fill_(0, inv[act], True)
    mixed = has_old & has_new
    vox_node = torch.unique(torch.stack([inv, nodes["node_of_cand"]], 1), dim=0)
    nodes_per_vox = torch.bincount(vox_node[:, 0], minlength=len(uk))
    if int(mixed.sum()) == 0:
        return None, 0
    return float((nodes_per_vox[mixed] >= 2).float().mean()), int(mixed.sum())


@torch.no_grad()
def node_noise(nodes, x):
    rel = ~nodes["filler"]
    noc, cand = nodes["node_of_cand"], nodes["cand"]
    K = nodes["g"].shape[0]
    p = x[cand].double()
    cnt = torch.zeros(K, device=DEV, dtype=torch.float64).index_add_(0, noc, torch.ones_like(p[:, 0]))
    mu = torch.zeros((K, 3), device=DEV, dtype=torch.float64).index_add_(0, noc, p) / cnt[:, None].clamp_min(1)
    q = p - mu[noc]
    C = torch.zeros((K, 3, 3), device=DEV, dtype=torch.float64).index_add_(0, noc, q[:, :, None] * q[:, None, :])
    ok = rel & (cnt >= 3)
    if int(ok.sum()) == 0:
        return None
    _, V = torch.linalg.eigh(C[ok])
    n = V[:, :, 0]
    d = ((nodes["g"][ok].double() - mu[ok]) * n).sum(-1).abs()
    return float(d.median())


@torch.no_grad()
def s_factor_diag(rel, alpha, cfg):
    """Which factor of S_i (§4.3) is low, and what S / R would be under candidate scales (repair round §8.2-4)."""
    rc = cfg["reliability"]
    n, rD, en = rel["n"], rel["rD"], rel["en"]
    ok = torch.isfinite(rD) & torch.isfinite(en) & (n >= rc["n_min"])
    a = alpha.reshape(-1)

    def q(x, ps=(0.25, 0.5, 0.75, 0.9)):
        x = x[ok].float()
        if x.numel() == 0:
            return None
        x = x[torch.randperm(x.numel(), device=x.device)[:1000000]]
        return {f"p{int(100 * p)}": float(torch.quantile(x, p)) for p in ps}

    f_n = 1 - torch.exp(-n / rc["n0"])
    f_D = torch.exp(-(rD / rc["sigma_D"]) ** 2)
    f_e = torch.exp(-(en / rc["sigma_n"]) ** 2)
    out = {"frac_scored": float(ok.float().mean()), "rD_mm": {k: 1e3 * v for k, v in (q(rD) or {}).items()},
           "en": q(en), "n": q(n), "f_n": q(f_n), "f_alpha": q(a), "f_D": q(f_D), "f_e": q(f_e), "candidates": {}}
    smax = rel.get("smax")
    for sD in (0.01, 0.02, 0.03, 0.05):
        for sn in (0.2, 0.3, 0.5):
            S = f_n * a * torch.exp(-(rD / sD) ** 2) * torch.exp(-(en / sn) ** 2)
            S = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
            R = rel["H"] & (S >= rc["tau_S"])
            out["candidates"][f"sD{int(sD * 1e3)}_sn{sn}"] = {"frac_R": float(R.float().mean()),
                                                              "S_median_scored": float(S[ok].median()) if ok.any() else None}
    return out


def stats(nodes, edges, x, t0, alpha, active, r, dt):
    cov = coverage(x, t0, alpha, nodes["g"], nodes["t"], r, dt)
    sep, n_mixed = separation(nodes, active, x, r, alpha)
    return {"n_nodes": int(nodes["g"].shape[0]), "filler_frac": float(nodes["filler"].float().mean()),
            "n_edges": int(edges.shape[0]), "coverage": cov, "separation": sep, "n_mixed_voxels": n_mixed,
            "node_noise_mm": None if node_noise(nodes, x) is None else 1e3 * node_noise(nodes, x),
            "time_s": nodes["time_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "fig"), exist_ok=True)
    rows, Svals, topdown = [], {}, None
    for rd in a.runs:
        for p in list_dumps(rd):
            d = load_dump(p)
            st = EventState(d)
            cfg = dcfg.resolve({}, st.config)
            inp = inp_from_dump(d, st.frame)
            kf = sorted(inp["kf_uids"])
            dt = delta_t_frames(kf, inp["W"])
            r = cfg["nodes"]["r_node"]
            cams = {u: make_cam(u, inp["poses_pre"][u], inp["intr"]) for u in kf}
            rel = compute_reliability(inp["g"], cams, inp["frame_fn"], inp["t0"], dt, cfg, inp["tr"], kf)
            alpha = inp["g"]["opacity"].reshape(-1)
            dT = delta_T_dict(kf, inp["all_cam_ids"], inp["poses_pre"], inp["poses_pgo"])
            xyz_r, _ = apply_rigid(inp["g"]["xyz"], inp["g"]["rot"], inp["tc"], dT)
            row = {"scene": st.meta["scene"], "event_id": st.meta["event_id"], "delta_t": dt,
                   "reliability": {"time_s": rel["time_s"], "kfs_used": rel["n_kfs_used"], "halved": rel["halved"],
                                   "frac_R": float(rel["R"].float().mean()), "frac_H": float(rel["H"].float().mean()),
                                   "S_median": float(rel["S"].median())},
                   "S_factors": s_factor_diag(rel, alpha, cfg)}
            Svals.setdefault(st.meta["scene"], {"old": [], "new": []})
            Svals[st.meta["scene"]]["old"].append(rel["S"][~inp["active"]].cpu().numpy())
            Svals[st.meta["scene"]]["new"].append(rel["S"][inp["active"]].cpu().numpy())
            for var, x in (("A", xyz_r), ("B", inp["g"]["xyz"])):
                nd = build_nodes(x, inp["t0"], inp["tc"], rel["S"], rel["R"], alpha, r, dt)
                E = build_edges(nd, r, dt)
                row[f"main_{var}"] = stats(nd, E, x, inp["t0"], alpha, inp["active"], r, dt)
                row[f"main_{var}"]["time_total_s"] = rel["time_s"] + nd["time_s"]
                if var == "B" and topdown is None:
                    topdown = (st.meta["scene"], st.meta["event_id"], nd["g"].cpu().numpy(), nd["t"].cpu().numpy(),
                               nd["filler"].cpu().numpy())
            if not a.no_ablation:
                x = inp["g"]["xyz"]
                abl = {}
                for name, kw in (("ii_voxel", dict(use_time=False, use_S=False)),
                                 ("iii_voxel_time", dict(use_time=True, use_S=False)),
                                 ("iv_main", dict(use_time=True, use_S=True))):
                    nd = build_nodes(x, inp["t0"], inp["tc"], rel["S"], rel["R"], alpha, r, dt, **kw)
                    abl[name] = stats(nd, build_edges(nd, r, dt), x, inp["t0"], alpha, inp["active"], r, dt)
                n_target = abl["iv_main"]["n_nodes"]
                pick = systematic_nodes(x, alpha, n_target, seed=0)
                nd = {"g": x[pick], "t": inp["t0"][pick].float(), "filler": torch.zeros(len(pick), dtype=torch.bool, device=DEV)}
                abl["i_systematic"] = {"n_nodes": len(pick),
                                       "coverage": coverage(x, inp["t0"], alpha, nd["g"], nd["t"], r, dt)}
                row["ablation"] = abl
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("scene", "event_id")} | {"B": row["main_B"], "rel": row["reliability"]}),
                  flush=True)
            del inp, cams, rel
            torch.cuda.empty_cache()
    with open(os.path.join(a.out, "P2_metrics.json"), "w") as f:
        json.dump({"events": rows}, f, indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if topdown is not None:
        sc, ev, gN, tn, fl = topdown
        fig, ax = plt.subplots(figsize=(6, 6))
        s = ax.scatter(gN[~fl, 0], gN[~fl, 1], c=tn[~fl], s=6, cmap="viridis", label="reliable")
        ax.scatter(gN[fl, 0], gN[fl, 1], c=tn[fl], s=10, marker="x", cmap="viridis", label="filler")
        fig.colorbar(s, ax=ax, label="t_k (frame index)")
        ax.set_aspect("equal")
        ax.set_title(f"{sc} event {ev}: nodes (top-down)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig", "P2_nodes_topdown.png"), dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for sc, v in Svals.items():
        for lay in ("old", "new"):
            x = np.concatenate(v[lay]) if v[lay] else np.zeros(0)
            if x.size:
                ax.hist(x, bins=50, histtype="step", density=True, label=f"{sc} {lay}")
    ax.set_xlabel("S_i")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig", "P2_hist_S.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
