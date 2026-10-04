"""Plan v2 diagnostics D1 (oracle), D2 (pair-filter ablation), D3 (registration pairs) on loop dumps.

usage: python tools/run_diag_v2.py --exp D1|D2|D3 [--only NAME[,NAME]] --out results_exp/reports/v2/diag RUN_DIR...
Every config is V-A from the frozen A-1 (configs/deform/selected.yaml) with the plan v2 switches.
Per (event, config) it appends one row to OUT/rows.jsonl and stores the |gap| / validity / pair-coverage
of every pixel of the unified evaluation set Pi* in OUT/maps/<dump>.pt. Pi* (cells B of J_eval, base
gating) is built from the rigid result with the BASE config, so it is identical for every config and
every invocation; the intersection over configs is done once by tools/aggregate_diag_v2.py.
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False  # plan v2 Q1 (the deform pipeline has no cuDNN ops)
torch.backends.cudnn.allow_tf32 = False

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.dump import EventState, gauss_to_dev, list_dumps, load_dump  # noqa: E402
from deform.influence import influence  # noqa: E402
from deform.pipeline import correct_map, inp_from_dump, prep_key, with_pos  # noqa: E402
from deform.render_utils import DEV  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED = os.path.join(ROOT, "configs", "deform", "selected.yaml")
BIG = 10 ** 9
# the oracle always uses the converged solver (tol_mode energy): with the legacy stopping rule LBFGS stops
# after one iteration on ~1e5-1e6 pairs and would not be an upper bound
ORACLE = {"corr": {"mode": "oracle", "stride": 1, "eps_d": "eval", "s_gate": {"mode": "off"},
                   "max_per_node": BIG, "max_total": BIG}, "accept": {"enforce": False},
          "solver": {"tol_mode": "energy"}}
F1W = {"s_gate": {"mode": "weight"}}
ALL = {"s_gate": {"mode": "weight"}, "eps_d": 0.10, "split": "checkerboard", "stride": "half"}
CONFIGS = {
    "A-1-ref": {},
    "O1": ORACLE,
    "O1-loose": {**ORACLE, "energy": {"w_p": 0.1, "w_reg": 0.1}},
    "F1w": {"corr": F1W},
    "F1t": {"corr": {"s_gate": {"tau": 0.1}}},
    "F1o": {"corr": {"s_gate": {"layers": "old"}}},
    "F2": {"corr": {"eps_d": 0.10}},
    "F3": {"corr": {"split": "checkerboard"}},
    "F4": {"corr": {"stride": "half"}},
    "ALL-1": {"corr": ALL},
    "ALL-10": {"corr": ALL, "energy": {"w_p": 10.0}},
    "ALL-noF3": {"corr": {k: v for k, v in ALL.items() if k != "split"}},  # report only (F3 leakage control)
    "D3-p2p": {"corr": {"mode": "registration", "residual": "p2p"}},
    "D3-p2l": {"corr": {"mode": "registration", "residual": "p2l"}},
    "D3-p2p-10": {"corr": {"mode": "registration", "residual": "p2p"}, "energy": {"w_p": 10.0}},
    "D3-all": {"corr": {"mode": "registration", "residual": "p2p", "reg_use_eval": True}},
    # plan v3: per-node pair cap 64 -> 256 (alone and with the S-gate relaxations)
    "F6": {"corr": {"max_per_node": 256}},
    "F1t+F6": {"corr": {"s_gate": {"tau": 0.1}, "max_per_node": 256}},
    "F1o+F6": {"corr": {"s_gate": {"layers": "old"}, "max_per_node": 256}},
    # plan v4 K3: F1o with the hash cap + deterministic renders / torch ops + gauge anchor (= candidate v5),
    # 4 more salts (sampling spread of the hash draw) and 4 more seeds of the random cap (spread of v3's draw)
    "F1o-hash": {"corr": {"s_gate": {"layers": "old"}, "select": "hash"}, "det": {"render": True, "torch": True},
                 "pose_sync": {"anchor": "prior"}},
    **{f"F1o-hash-s{k}": {"corr": {"s_gate": {"layers": "old"}, "select": "hash",
                                   "hash_salt": ((k + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF},
                          "det": {"render": True, "torch": True}, "pose_sync": {"anchor": "prior"}} for k in (1, 2, 3, 4)},
    **{f"F1o-r{k}": {"corr": {"s_gate": {"layers": "old"}, "seed_offset": k}} for k in (1, 2, 3, 4)},
}
EXPS = {"D1": ["A-1-ref", "O1", "O1-loose"],
        "D2": ["A-1-ref", "F1w", "F1t", "F1o", "F2", "F3", "F4", "ALL-1", "ALL-10", "ALL-noF3"],
        "D3": ["A-1-ref", "D3-p2p", "D3-p2l", "D3-p2p-10", "D3-all"],
        "V3": ["A-1-ref", "O1", "F1t", "F1o", "F1w", "F6", "F1t+F6", "F1o+F6"],
        "V4": ["A-1-ref", "F1o", "F1o-hash", "F1o-hash-s1", "F1o-hash-s2", "F1o-hash-s3", "F1o-hash-s4",
               "F1o-r1", "F1o-r2", "F1o-r3", "F1o-r4"],
        "V4rep": ["A-1-ref", "F1o", "F1o-hash"]}
# with --solver-mode grad the oracle is capped (it has 1e5-1e6 pairs); its convergence is reported, not gated
ORACLE_MAX_ITER = 1000


def mesh_path_for(st):
    if st.dtype_name != "replica":
        return None
    p = os.path.join(ROOT, "datasets", "replica", f'{st.config["Dataset"]["sequence_name"]}_mesh.ply')
    return p if os.path.exists(p) else None


def make_cfg(base, name):
    """NAME or NAME+conv (same config with the converged solver, tol_mode energy; report only)."""
    if name.endswith("+conv"):
        cfg = make_cfg(base, name[:-len("+conv")])
        cfg["solver"]["tol_mode"] = "energy"
        return cfg
    over = copy.deepcopy(CONFIGS[name])
    c = over.get("corr", {})
    if c.get("eps_d") == "eval":
        c["eps_d"] = base["corr"]["eps_eval"]
    if c.get("stride") == "half":
        c["stride"] = max(1, int(base["corr"]["stride"]) // 2)
    if c.get("mode") == "oracle" and base.get("dataset_type") == "replica":
        # Replica 1200x680: stride 1 would give ~3e6 oracle pairs; stride 2 still contains every evaluation
        # pixel (evaluation stride 8) and keeps the problem at the TUM-oracle scale (~6e5 pairs)
        c["stride"] = 2
    cfg = dcfg._merge(base, over)
    cfg["corr"]["cos_theta_n"] = base["corr"]["cos_theta_n"]
    return cfg


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def dominant_nodes(st, inp, ctx, Pi, poses_rig, G_rig, m_old, m_new):
    """Node with the largest weight at the new-layer 3D point of every Pi* pixel (V-A: x_pre = rigid)."""
    from scipy.spatial import cKDTree

    nodes, nc = ctx["nodes"], None
    out = {}
    xyz = G_rig["xyz"]
    xyz_np = xyz.detach().cpu().numpy().astype(np.float64)
    for uid, (m, _) in Pi.items():
        cam = st.cam(uid, poses=poses_rig)
        L = M.render_layers(cam, G_rig, m_old, m_new)
        r = M.ray_dirs(cam)
        c2w = torch.linalg.inv(cam.T.double()).float()
        y = (r * L["Dn"][None])[:, m].T @ c2w[:3, :3].T + c2w[:3, 3]
        cand = torch.nonzero((L["Cn"] > 0.5) & m_new).reshape(-1)
        if y.shape[0] == 0 or cand.numel() == 0:
            out[uid] = torch.full((y.shape[0],), -1, dtype=torch.long, device=DEV)
            continue
        _, j = cKDTree(xyz_np[cand.cpu().numpy()]).query(y.cpu().numpy().astype(np.float64), k=1, workers=4)
        tau = inp["t0"][cand[torch.as_tensor(j, device=DEV, dtype=torch.long)]].float()
        nc = ctx["_nc"]
        idx, w = influence(y, tau, nodes["g"], nodes["t"], nc["r_node"], ctx["dt"], nc["K"], nc["K_cand"], nc["beta"])
        out[uid] = idx[torch.arange(idx.shape[0], device=DEV), w.argmax(1)]
    return out


def pair_split_overlap(ctx, Pi_regions):
    """T3: pairs whose source pixel lies in a cell B of a J_eval keyframe (must be 0 except for oracle)."""
    p = ctx.get("pairs") or {}
    if not p or p.get("P", 0) == 0 or "src_uid" not in p:
        return 0
    n = 0
    for uid, reg in Pi_regions.items():
        sel = p["src_uid"] == int(uid)
        if bool(sel.any()):
            n += int(reg.reshape(-1)[p["src_pix"][sel]].sum())
    return n


def run_event(path, names, out_dir, exp, solver_mode=None):
    d = load_dump(path)
    st = EventState(d)
    inp = inp_from_dump(d, st.frame)
    user = load_deform_config(SELECTED)
    base = dcfg.resolve(user, st.config)
    base["seed"] = int(st.meta.get("seed", 0))
    if solver_mode:
        base["solver"]["tol_mode"] = solver_mode

    def cfg_for(name):
        c = make_cfg(base, name)
        if solver_mode:  # every config (oracle included) uses the same solver in a v3 run
            c["solver"]["tol_mode"] = solver_mode
            if name.startswith("O1"):
                c["solver"]["max_iter_conv"] = ORACLE_MAX_ITER
        return c
    cache = {}
    m_old, m_new = M.layer_masks(st.active)
    H, W = int(st.intr["H"]), int(st.intr["W"])
    dump = os.path.basename(path)
    # reference: A-1 run gives the rigid state and J_eval; Pi* from the base config
    ref = correct_map(inp, base, "A", cache)
    G_rig, P_rig = ref["G_rig"], ref["poses_rig"]
    J_eval = ref["J_eval"]
    Pi, dD = M.pi_star(st, G_rig, P_rig, J_eval, m_old, m_new, base)
    regionsB = {u: M.checker(H, W, base["corr"]["checker"], 1) for u in J_eval}
    rig_maps = M.gap_maps(st, G_rig, P_rig, Pi, m_old, m_new)
    ctx_ref = cache[prep_key(base, "A")]
    ctx_ref["_nc"] = base["nodes"]
    dom = dominant_nodes(st, inp, ctx_ref, Pi, P_rig, G_rig, m_old, m_new)
    G_pre = gauss_to_dev(st.gpre)
    JL = sorted(set(ref["J_opt"]) | set(J_eval))
    sets = M.step_sets(st, G_pre, st.poses_pre, JL, base["seed"]) if JL else None
    mp = os.path.join(out_dir, "maps", dump + ".pt")
    store = torch.load(mp, weights_only=False) if os.path.exists(mp) else {"configs": {}}
    pix = {u: torch.nonzero(m.reshape(-1)).reshape(-1).cpu() for u, (m, _) in Pi.items()}
    if "pix" in store:
        same = set(store["pix"]) == set(pix) and all(torch.equal(store["pix"][u], pix[u]) for u in pix)
        if not same:
            raise RuntimeError(f"Pi* changed between invocations for {dump}")
    store["pix"] = pix
    store["dD"] = {u: dD[u].reshape(-1)[pix[u].to(DEV)].cpu() for u in pix}
    store["rigid"] = {u: (rig_maps[u]["gap"].reshape(-1)[pix[u].to(DEV)].cpu(),
                          rig_maps[u]["valid"].reshape(-1)[pix[u].to(DEV)].cpu()) for u in pix}

    mesh = mesh_path_for(st)
    acc_idx = None
    if mesh is not None and J_eval:  # M5 visible set fixed from the rigid reference at J_eval (as P3)
        vis = M.visible_mask(st, G_rig, P_rig, sorted(J_eval)) & (G_rig["opacity"].reshape(-1) > 0.5)
        acc_idx = torch.nonzero(vis).reshape(-1).cpu().numpy()

    def metrics_of(Gx, Px):
        rq = M.render_quality(st, Gx, Px, sorted(J_eval)) if J_eval else {}
        es = M.e_step(Gx["xyz"], sets) if sets is not None else {}
        acc = M.mesh_acc(Gx["xyz"], acc_idx, mesh) if acc_idx is not None else None
        return {"psnr": rq.get("psnr"), "depth_l1": rq.get("depth_l1"), "ate": M.ate_kf(st, Px),
                "estep": es.get("e_step"), "acc": None if acc is None else acc["acc_median"],
                "acc_gt1cm": None if acc is None else acc["frac_gt_1cm"]}

    rows = []
    meta = {"dump": dump, "scene": st.meta["scene"], "event_id": st.meta["event_id"], "J_eval": J_eval,
            "J_opt": ref["J_opt"], "n_pi_pixels": int(sum(v.numel() for v in pix.values()))}
    rows.append({**meta, "config": "rigid", "exp": exp, **metrics_of(G_rig, P_rig)})
    for name in names:
        t0 = time.time()
        cfg = cfg_for(name)
        try:
            res = ref if name == "A-1-ref" else correct_map(inp, cfg, "A", cache)
        except Exception as ex:  # one failing config (e.g. out of memory) must not stop the others
            err = f"{type(ex).__name__}: {str(ex)[:300]}"
            rows.append({**meta, "config": name, "exp": exp, "error": err})
            print(json.dumps({"ev": meta["event_id"], "cfg": name, "error": err}), flush=True)
            cache.pop(prep_key(cfg, "A"), None)
            torch.cuda.empty_cache()
            continue
        ctx = cache[prep_key(cfg, "A")]
        Gx = with_pos(inp["g"], res["xyz"], res["rot"])
        maps = M.gap_maps(st, Gx, res["poses"], Pi, m_old, m_new)
        cov = ctx.get("covered_nodes")
        cfg_store = {}
        for u in pix:
            ii = pix[u].to(DEV)
            dn = dom[u]
            c = torch.zeros_like(dn, dtype=torch.bool) if cov is None else torch.where(dn >= 0, cov[dn.clamp_min(0)], torch.zeros_like(dn, dtype=torch.bool))
            cfg_store[u] = (maps[u]["gap"].reshape(-1)[ii].cpu(), maps[u]["valid"].reshape(-1)[ii].cpu(), c.cpu())
        store["configs"][name] = cfg_store
        lg = res["log"]
        row = {**meta, "config": name, "exp": exp, **metrics_of(Gx, res["poses"]),
               "accepted": res["accepted"], "reason": res["reason"],
               "t3_overlap_B": None if cfg["corr"].get("mode") == "oracle" else pair_split_overlap(ctx, regionsB),
               "eval_wall_s": time.time() - t0,
               **{k: lg.get(k) for k in ("n_nodes", "n_filler", "n_edges", "n_loop_kfs", "n_opt_kfs", "n_eval_kfs",
                                         "n_src_kfs", "corr_raw", "corr_gated", "corr_capped", "funnel", "gate_frac",
                                         "n_nodes_with_pairs", "frac_res_gt30mm", "accept_checks", "E_init",
                                         "E_final", "lbfgs_iters", "edl_opt_init_mm", "edl_opt_final_mm",
                                         "max_node_disp_m", "max_node_rot_deg", "t_stage_s", "reg_kfs",
                                         "pose_sync", "stretch", "reliability", "solver_grad", "anchor",
                                         "det_warnings")}}
        rows.append(row)
        print(json.dumps({"ev": meta["event_id"], "cfg": name, "acc": res["accepted"], "reason": res["reason"],
                          "pairs": lg.get("corr_capped"), "t_s": round(time.time() - t0, 1),
                          "t3": row["t3_overlap_B"]}), flush=True)
        del res, Gx, maps
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    torch.save(store, mp)
    with open(os.path.join(out_dir, "rows.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    del cache, inp
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXPS))
    ap.add_argument("--only", default=None)
    ap.add_argument("--conv", action="store_true", help="also run NAME+conv (converged solver) for each config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--solver-mode", default=None, help="override solver.tol_mode for every config (plan v3: grad)")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    names = list(EXPS[a.exp]) if not a.only else [n for n in a.only.split(",")]
    if a.conv:  # converged-solver twin of every non-oracle config (report only)
        names += [n + "+conv" for n in names if not n.startswith("O1")]
    if "A-1-ref" in names:
        names.remove("A-1-ref")
    names = ["A-1-ref"] + names
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for rd in a.runs:
        for p in list_dumps(rd):
            run_event(p, names, a.out, a.exp, a.solver_mode)
    print(f"[diag_v2] {a.exp} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
