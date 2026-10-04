"""P5: end-of-run metrics (§5.2) for every run + paired per-event metrics (§5.1) for method runs.

usage: python tools/aggregate_runs.py --out results_exp/reports/P5 [--no-events] RUN_DIR [RUN_DIR ...]
For a method run each dump gives a pair: rigid offline (replayed from "pre") vs the actual result ("post").
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform.dump import EventState, gauss_to_dev, list_dumps, load_dump  # noqa: E402
from deform.metrics import evaluate_event, mesh_scene  # noqa: E402
from deform.render_utils import DEV  # noqa: E402
from deform.rigid import replay  # noqa: E402


def read_csv(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return {k: (float(v) if v not in ("", None) and k != "run_name" else v) for k, v in rows[0].items()} if rows else None


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def sign_test_p(k, n):
    """One-sided P(X >= k), X ~ Bin(n, 1/2)."""
    if n == 0:
        return None
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


@torch.no_grad()
def map_quality(state_path, mesh_path, seed=0):
    """Acc / Comp vs GT mesh and whole-map e_step for a final_state*.pt."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    if not os.path.exists(state_path):
        return None
    s = torch.load(state_path, map_location="cpu", weights_only=False)
    gs = s["gaussians"]
    xyz = gs["xyz"].numpy().astype(np.float64)
    alpha = gs["opacity"].reshape(-1).numpy()
    out = {"n_gauss": int(xyz.shape[0])}
    sel = alpha > 0.5
    if mesh_path:
        sc = mesh_scene(mesh_path)
        d = sc.compute_distance(o3d.core.Tensor(xyz[sel].astype(np.float32))).numpy()
        out["acc_median_mm"] = 1e3 * float(np.median(d))
        out["acc_frac_gt_1cm"] = float((d > 0.01).mean())
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        pts = np.asarray(mesh.sample_points_uniformly(200000, seed=seed).points) if len(mesh.triangles) else None
        if pts is not None:
            dd, _ = cKDTree(xyz[sel]).query(pts, k=1, workers=4)
            out["comp_mean_mm"] = 1e3 * float(dd.mean())
            out["comp_ratio_2cm"] = float((dd < 0.02).mean())
    # whole-map e_step (B / N-bar on the map itself, 16 nn within 5 cm)
    tc = gs["tc"].numpy()
    idx = np.nonzero(sel)[0]
    rng = np.random.default_rng(seed)
    q = rng.choice(idx, size=min(200000, len(idx)), replace=False)
    tree = cKDTree(xyz[idx])
    dist, j = tree.query(xyz[q], k=17, distance_upper_bound=0.05, workers=4)
    nb = np.where(np.isfinite(dist), idx[np.minimum(j, len(idx) - 1)], -1)
    nb[nb == q[:, None]] = -1
    order = np.argsort(nb < 0, axis=1, kind="stable")
    nb = np.take_along_axis(nb, order, axis=1)[:, :16]
    valid = nb >= 0
    nd = (valid & (tc[np.maximum(nb, 0)] != tc[q][:, None])).sum(1)
    nv = valid.sum(1)
    B = np.nonzero((nd >= 3) & (nv >= 3))[0]
    Np = np.nonzero((nd == 0) & (nv >= 3))[0]
    Nn = rng.choice(Np, size=min(len(B), len(Np)), replace=False) if len(Np) else Np

    def dev(rows):
        if len(rows) == 0:
            return np.zeros(0)
        P = xyz[np.maximum(nb[rows], 0)]
        m = valid[rows][..., None].astype(np.float64)
        c = m.sum(1)
        mu = (P * m).sum(1) / np.maximum(c, 1)
        Q = (P - mu[:, None]) * m
        C = np.einsum("nki,nkj->nij", Q, Q) / np.maximum(c, 1)[..., None]
        _, V = np.linalg.eigh(C)
        n = V[:, :, 0]
        return np.abs(((xyz[q[rows]] - mu) * n).sum(-1))
    eB, eN = dev(B), dev(Nn)
    if len(eB) and len(eN):
        out["estep_mm"] = 1e3 * float(np.median(eB) - np.median(eN))
        out["estep_p90B_mm"] = 1e3 * float(np.quantile(eB, 0.9))
    return out


def run_info(rd, root):
    info = {"run": rd, "tag": os.path.basename(rd).split("_", 1)[1] if "_" in os.path.basename(rd) else rd}
    info["metrics"] = read_csv(os.path.join(rd, "metrics.csv"))
    info["metrics_prerefine"] = read_csv(os.path.join(rd, "metrics_prerefine.csv"))
    rp = os.path.join(rd, "resources.json")
    info["resources"] = json.load(open(rp)) if os.path.exists(rp) else None
    ev = read_jsonl(os.path.join(rd, "loop_events.jsonl"))
    info["n_events"] = len(ev)
    info["n_accepted"] = sum(1 for e in ev if e.get("accepted") is True)
    info["n_fallback"] = sum(1 for e in ev if e.get("accepted") is False)
    info["fallback_reasons"] = {}
    for e in ev:
        if e.get("accepted") is False:
            r = str(e.get("fallback_reason")).split(":")[0]
            info["fallback_reasons"][r] = info["fallback_reasons"].get(r, 0) + 1
    info["t_event_s"] = [e.get("t_event_s", (e.get("t_stage_s") or {}).get("total")) for e in ev]
    info["mode"] = ev[0]["mode"] if ev else ("deform" if "deform" in info["tag"] else "rigid")
    st = os.path.join(rd, "final_state.pt")
    cfg = torch.load(st, map_location="cpu", weights_only=False)["config"] if os.path.exists(st) else None
    mesh = None
    if cfg is not None and cfg["Dataset"]["type"] == "replica":
        p = os.path.join(root, "datasets", "replica", f'{cfg["Dataset"]["sequence_name"]}_mesh.ply')
        mesh = p if os.path.exists(p) else None
    info["scene"] = f'{cfg["Dataset"]["type"]}/{cfg["Dataset"]["sequence_name"]}' if cfg else None
    info["seed"] = cfg["Results"].get("seed") if cfg else None
    info["map_pre"] = map_quality(st, mesh)
    info["map_refined"] = map_quality(os.path.join(rd, "final_state_refined.pt"), mesh)
    return info


def paired_events(rd, root):
    rows = []
    for p in list_dumps(rd):
        d = load_dump(p)
        st = EventState(d)
        cfg = dcfg.resolve({}, st.config)
        with torch.no_grad():
            xr, qr, _ = replay(d, use_online=False)  # clean fp32 rigid reference (no TF32 noise)
            methods = {"pre": (gauss_to_dev(st.gpre), st.poses_pre),
                       "rigid": (gauss_to_dev(st.gpre, xyz=xr, rot=qr), st.poses_pgo),
                       "method": (gauss_to_dev(st.gpre, xyz=d["gauss_post"]["xyz"], rot=d["gauss_post"]["rot"]),
                                  d["poses_final"])}
            mesh = None
            if st.dtype_name == "replica":
                mp = os.path.join(root, "datasets", "replica", f'{st.config["Dataset"]["sequence_name"]}_mesh.ply')
                mesh = mp if os.path.exists(mp) else None
            ev = evaluate_event(st, methods, "rigid", cfg, mesh_path=mesh, seed=int(st.meta.get("seed", 0)))
        rr = {"run": rd, "scene": st.meta["scene"], "event_id": st.meta["event_id"], "J_eval": ev["J_eval"],
              "accepted": d.get("deform_log", {}).get("accepted"),
              "fallback_reason": d.get("deform_log", {}).get("fallback_reason"), "methods": {}}
        for k, m in ev["methods"].items():
            rr["methods"][k] = {"edl": (m["e_dl"] or {}).get("median"), "estep": (m["e_step"] or {}).get("e_step"),
                                "psnr": (m["render"] or {}).get("psnr"), "ate": m["ate_kf"],
                                "acc": (m["acc"] or {}).get("acc_median")}
        rows.append(rr)
        print(json.dumps({k: rr[k] for k in ("scene", "event_id", "accepted")} |
                         {"edl_rigid": rr["methods"]["rigid"]["edl"], "edl_method": rr["methods"]["method"]["edl"]}),
              flush=True)
        torch.cuda.empty_cache()
    return rows


def paired_summary(rows, metric):
    out = {}
    for sc in sorted({r["scene"] for r in rows}):
        rr = [r for r in rows if r["scene"] == sc and r["J_eval"] if metric == "edl"] if metric == "edl" else \
             [r for r in rows if r["scene"] == sc]
        pairs = [(r["methods"]["rigid"][metric], r["methods"]["method"][metric]) for r in rr
                 if r["methods"]["rigid"][metric] is not None and r["methods"]["method"][metric] is not None]
        if not pairs:
            out[sc] = {"n": 0}
            continue
        a = np.array(pairs)
        relc = [(m - r) / r for r, m in pairs if r != 0]
        k = int(sum(m < r for r, m in pairs))
        out[sc] = {"n": len(pairs), "rigid_median_mm": 1e3 * float(np.median(a[:, 0])),
                   "method_median_mm": 1e3 * float(np.median(a[:, 1])),
                   "median_rel_change": float(np.median(relc)) if relc else None,
                   "n_improved": k, "p_sign": sign_test_p(k, len(pairs))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(a.out, "fig"), exist_ok=True)
    t0 = time.time()
    infos = [run_info(rd, root) for rd in a.runs]
    for i in infos:
        print(json.dumps({k: i[k] for k in ("run", "scene", "mode", "seed", "n_events", "n_accepted", "n_fallback")}))
    events = []
    if not a.no_events:
        for i in infos:
            if i["mode"] == "deform":
                events += paired_events(i["run"], root)
    summ = {"edl": paired_summary(events, "edl"), "estep": paired_summary(events, "estep")} if events else {}
    with open(os.path.join(a.out, "P5_metrics.json"), "w") as f:
        json.dump({"runs": infos, "events": events, "paired": summ, "time_s": time.time() - t0}, f, indent=1, default=str)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
