"""correct_map(): the full loop-event correction (plan §4.2), shared by offline tools and the backend.

Input `inp` (dict, everything on GPU unless noted):
  g        : {xyz, rot, scale, opacity, f_dc}  map BEFORE any correction
  t0, tc   : [N] long ; active [N] bool (snapshot before reactivation)
  kf_uids, all_cam_ids : lists ; poses_pre / poses_pgo : {uid: 4x4 c2w float64 (cpu)}
  intr     : {fx, fy, cx, cy, W, H} ; frame_fn(uid) -> (rgb, depth scaled) ; tr : Training config
  W        : old_than_N_keyframe ; seed : int
Returns dict: xyz, rot (final map), poses {uid: c2w}, accepted, reason, log, and (for metrics) the
rigid state, J_opt, J_eval.

Stages [1]-[5] do not depend on the energy weights; with `cache` (a dict per event) they are computed
once per variant and reused across the P3 grid. Timing logs always report the from-scratch cost.
"""
import copy
import json
import time

import torch

from deform import metrics as M
from deform.apply import apply_to_gaussians, stretch_diag, sync_poses
from deform.correspondences import HASH_SALT, build_pairs, build_pairs_registration, cap_pairs
from deform.field import blend_quat, phi, quat_of, quat_to_rotmat
from deform.influence import influence
from deform.nodes import build_edges, build_nodes
from deform.reliability import compute_reliability
from deform.render_utils import DEV, det_scope, make_cam, torch_det_scope
from deform.rigid import apply_rigid, quat_mul
from deform.solver import Problem, solve, wmedian


def rel_key(cfg):
    """Cache key of stage [1] (reliability): the render build decides S (plan v4 A3), nothing else does."""
    return "rel_det" if (cfg.get("det") or {}).get("render", False) else "rel"


def prep_key(cfg, variant):
    """Cache key of stages [1]-[5]: everything that changes the pairs (plan v2 switches included).
    Constant for every config of the P3 grid, so P3 caching is unchanged."""
    k = {"corr": cfg["corr"], "reg": cfg.get("reg"), "tau_S": cfg["reliability"]["tau_S"],
         "enforce": cfg["accept"].get("enforce", True), "min_corr": cfg["accept"]["min_corr"],
         "det": cfg.get("det")}
    return f"prep_{variant}_" + json.dumps(k, sort_keys=True, default=str)


def pair_sources(J_opt, J_eval, H, W, cfg):
    """(uid, region) allowed to produce pairs. Default: J_opt (A-1). F3 / D3-all: + cells A (parity 0)
    of every J_eval keyframe. Oracle (D1): cells B (parity 1) of J_eval = the evaluation cells."""
    c = cfg["corr"]
    tile = c["checker"]
    if c.get("mode", "pixel") == "oracle":
        return [(u, M.checker(H, W, tile, 1)) for u, _ in J_eval]
    src = list(J_opt)
    extra = c.get("split", "keyframe") == "checkerboard" or (c.get("mode") == "registration" and c.get("reg_use_eval"))
    if extra:
        have = {u for u, _ in src}
        src += [(u, M.checker(H, W, tile, 0)) for u, _ in J_eval if u not in have]
    return src


def delta_T_dict(kf_uids, all_cam_ids, poses_pre, poses_pgo):
    """{uid: float32 4x4} exactly like the backend (duplicates in all_cam_ids -> identity)."""
    kf, done, out = set(kf_uids), set(), {}
    for fid in all_cam_ids:
        if fid not in kf:
            continue
        if fid in done:
            out[fid] = torch.eye(4, device=DEV)
            continue
        opt_w2c = torch.as_tensor(poses_pgo[fid]).to(DEV).to(torch.float32).inverse()
        old_w2c = torch.linalg.inv(torch.as_tensor(poses_pre[fid]).double()).float().to(DEV)
        out[fid] = opt_w2c.inverse() @ old_w2c
        done.add(fid)
    return out


def delta_t_frames(kf_uids, W):
    u = torch.tensor(sorted(kf_uids), dtype=torch.float64)
    return float(W) if u.numel() < 2 else float(W * torch.median(u[1:] - u[:-1]).item())


def node_init(nodes, dT, variant):
    """V-A: identity. V-B: R0 = dR_kappa, t0 = dT_kappa g - g (missing kappa -> identity), §4.6."""
    gN = nodes["g"]
    Kn = gN.shape[0]
    R0 = torch.eye(3, dtype=torch.float64, device=DEV).repeat(Kn, 1, 1)
    t_init = torch.zeros((Kn, 3), dtype=torch.float64, device=DEV)
    if variant == "B":
        for k_uid, T in dT.items():
            sel = nodes["kappa"] == int(k_uid)
            if bool(sel.any()):
                Td = T.double()
                R0[sel] = Td[:3, :3]
                gk = gN[sel].double()
                t_init[sel] = gk @ Td[:3, :3].T + Td[:3, 3] - gk
    return R0, t_init


def anchor_gauge(inp, src_poses, xyz_new, rot_new, poses_new):
    """plan v4 A4: the deformation + Kabsch sync is defined up to a global rigid motion; the pose graph pins the
    first frame with a fixed prior (sigma 1e-9). Move Gaussians and poses by M0 = P_src(a) P_new(a)^-1 so that
    the anchored frame a keeps exactly its source (PGO) pose; e_dl and aligned ATE are invariant."""
    anchors = [int(u) for u in (inp.get("anchor_uids") or []) if int(u) in poses_new]
    fallback = not anchors
    a = min(anchors) if anchors else min(int(u) for u in poses_new)
    P_ref = torch.as_tensor(src_poses[a], dtype=torch.float64)
    M0 = P_ref @ torch.linalg.inv(poses_new[a].double())
    poses_out = {u: (M0 @ P.double()) for u, P in poses_new.items()}
    R0, t0v = M0[:3, :3].to(DEV), M0[:3, 3].to(DEV)
    xyz_out = (xyz_new.double() @ R0.T + t0v).float()
    q0 = quat_of(R0[None]).float().expand(rot_new.shape[0], 4)
    rot_out = torch.nn.functional.normalize(quat_mul(q0, rot_new.float()), dim=-1)
    ang = float(torch.rad2deg(torch.arccos(((torch.trace(M0[:3, :3]) - 1) / 2).clamp(-1, 1))))
    resid = float((poses_out[a][:3, 3] - P_ref[:3, 3]).norm())
    return xyz_out, rot_out, poses_out, {"uid": a, "fallback_min_uid": fallback, "t_mm": 1e3 * float(M0[:3, 3].norm()),
                                         "rot_deg": ang, "resid_m": resid, "n_anchors": len(anchors)}


def rigid_result(inp, dT):
    xyz, rot = apply_rigid(inp["g"]["xyz"], inp["g"]["rot"], inp["tc"], dT)
    poses = {u: torch.as_tensor(inp["poses_pgo"][u]).double().cpu() for u in inp["poses_pgo"]}
    return xyz, rot, poses


def with_pos(G, xyz, rot):
    out = dict(G)
    out["xyz"], out["rot"] = xyz, rot
    return out


@torch.no_grad()
def _prepare(inp, cfg, variant, cache=None):
    """Stages [1]-[5] (independent of the energy weights) -> ctx."""
    key = prep_key(cfg, variant)
    if cache is not None and key in cache:
        return cache[key]
    t_all = time.perf_counter()
    ts, log = {}, {"variant": variant}
    seed = int(inp.get("seed", 0))
    g = inp["g"]
    kf_uids = sorted(inp["kf_uids"])
    dt = delta_t_frames(kf_uids, inp["W"])
    log["delta_t"] = dt
    nc = cfg["nodes"]
    dT = delta_T_dict(kf_uids, inp["all_cam_ids"], inp["poses_pre"], inp["poses_pgo"])
    xyz_rig, rot_rig, poses_rig = rigid_result(inp, dT)
    G_rig = with_pos(g, xyz_rig, rot_rig)
    m_old, m_new = M.layer_masks(inp["active"])
    alpha = g["opacity"].reshape(-1)
    ctx = {"ts": ts, "log": log, "dt": dt, "dT": dT, "xyz_rig": xyz_rig, "rot_rig": rot_rig,
           "poses_rig": poses_rig, "G_rig": G_rig, "kf_uids": kf_uids, "fallback": None}

    def done(reason=None):
        if inp.get("drop_pair") is not None:  # a probe context must never be reused for another input
            ctx["_no_cache"] = True
        ctx["fallback"] = reason
        ctx["t_prepare"] = time.perf_counter() - t_all + (rel_time if rel_cached else 0.0)
        if cache is not None and not ctx.get("_no_cache"):
            cache[key] = ctx
        return ctx

    # [1] reliability on the pre state at pre-PGO poses (shared by both variants)
    rk = rel_key(cfg)
    rel_cached = cache is not None and rk in cache
    if rel_cached:
        rel = cache[rk]
    else:
        cams_pre = {u: make_cam(u, inp["poses_pre"][u], inp["intr"]) for u in kf_uids}
        rel = compute_reliability(g, cams_pre, inp["frame_fn"], inp["t0"], dt, cfg, inp["tr"], kf_uids)
        del cams_pre
        if cache is not None:
            cache[rk] = rel
    rel_time = rel["time_s"]
    S, Rm = rel["S"], rel["R"]
    ctx["S"] = S
    ts["reliability"] = rel_time
    log["reliability"] = {"n_reliable": int(Rm.sum()), "frac_reliable": float(Rm.float().mean()),
                          "S_median": float(S.median()), "kfs_used": rel["n_kfs_used"], "halved": rel["halved"]}

    # [2] pre coordinates per variant
    if variant == "A":
        x_pre, q_pre = xyz_rig, rot_rig
    else:
        x_pre, q_pre = g["xyz"].float(), g["rot"].float()
    ctx["x_pre"], ctx["q_pre"] = x_pre, q_pre

    # [3] nodes, edges, influence of every Gaussian (tau = t0)
    t0_ = time.perf_counter()
    nodes = build_nodes(x_pre, inp["t0"], inp["tc"], S, Rm, alpha, nc["r_node"], dt, nc.get("min_alpha", 0.1))
    Kn = nodes["g"].shape[0]
    edges = build_edges(nodes, nc["r_node"], dt)
    ts["nodes"] = time.perf_counter() - t0_
    log.update({"n_nodes": Kn, "n_filler": int(nodes["filler"].sum()), "n_edges": int(edges.shape[0])})
    ctx["nodes"], ctx["edges"] = nodes, edges
    if Kn < 2:
        return done("few_nodes")
    t0_ = time.perf_counter()
    gN = nodes["g"]
    idx_g, w_g, dmax_g = influence(x_pre, inp["t0"].float(), gN, nodes["t"], nc["r_node"], dt,
                                   nc["K"], nc["K_cand"], nc["beta"], return_dmax=True)
    ts["influence"] = time.perf_counter() - t0_
    ctx.update({"idx_g": idx_g, "w_g": w_g, "dmax_g": dmax_g})

    # [4] initial node transforms and the phi_0 state
    R0, t_init = node_init(nodes, dT, variant)
    pos0 = phi(x_pre, idx_g, w_g, gN.double(), R0, t_init).float()
    qb0 = blend_quat(idx_g, w_g, quat_of(R0)).float()
    Rbar0 = quat_to_rotmat(qb0.double()).float()
    rot0 = torch.nn.functional.normalize(quat_mul(qb0, q_pre), dim=-1)
    G0 = with_pos(g, pos0, rot0)
    ctx.update({"R0": R0, "t_init": t_init})

    # [5] loop keyframes on the rigid state (shared with the metrics), split, correspondences on J_opt
    t0_ = time.perf_counter()
    H, W = int(inp["intr"]["H"]), int(inp["intr"]["W"])
    st_like = _StateLike(inp, kf_uids)
    JL, _ = M.select_loop_kfs(st_like, G_rig, poses_rig, m_old, m_new, cfg)
    log["n_loop_kfs"] = len(JL)
    if not JL:
        return done("no_overlap")
    J_opt, J_eval = M.split_opt_eval(JL, H, W, cfg["corr"]["checker"])
    ctx["J_opt"], ctx["J_eval"] = [u for u, _ in J_opt], [u for u, _ in J_eval]
    log["n_opt_kfs"], log["n_eval_kfs"] = len(J_opt), len(J_eval)
    J_src = pair_sources(J_opt, J_eval, H, W, cfg)
    log["n_src_kfs"] = len(J_src)
    cams_pgo = {u: make_cam(u, inp["poses_pgo"][u], inp["intr"]) for u, _ in J_src}
    if cfg["corr"].get("mode", "pixel") == "registration":
        pairs = build_pairs_registration(cams_pgo, G0, m_old, m_new, J_src, inp["t0"], x_pre, pos0, Rbar0, cfg, seed)
        log["reg_kfs"] = pairs.pop("reg_kfs")
    else:
        pairs = build_pairs(cams_pgo, G0, m_old, m_new, S, J_src, inp["t0"], x_pre, pos0, Rbar0, cfg, seed)
    gc = pairs["gate_counts"]
    nb = max(1, pairs["n_both_opaque"])
    log["corr_raw"] = pairs["n_both_opaque"]
    log["gate_frac"] = {"alpha": gc["alpha"] / max(1, pairs["n_raw"]), "dist": gc["dist"] / nb,
                        "normal": gc["normal"] / nb, "S": gc["S"] / nb, "edge": gc["edge"] / nb}
    log["corr_gated"] = pairs["P"]
    log["funnel"] = pairs.pop("funnel", None)
    ctx["dD_vals"] = pairs.pop("dD_vals", None)
    inf_old = inf_new = None
    covered = torch.zeros(Kn, dtype=torch.bool, device=DEV)
    drop = inp.get("drop_pair")  # plan v4 K2 probe: remove one gated pair (uid, flat pixel) before the cap
    if pairs["P"] > 0 and drop is not None:
        keep = ~((pairs["src_uid"] == int(drop[0])) & (pairs["src_pix"] == int(drop[1])))
        log["dropped_pair"] = [int(drop[0]), int(drop[1]), int((~keep).sum())]
        P0 = pairs["P"]
        pairs = {k: (v[keep] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == P0 else v) for k, v in pairs.items()}
        pairs["P"] = int(pairs["n"].shape[0])
    if pairs["P"] > 0:
        inf_new = influence(pairs["x_new"].float(), pairs["tau_new"], gN, nodes["t"], nc["r_node"], dt,
                            nc["K"], nc["K_cand"], nc["beta"])
        node_of_pair = inf_new[0][torch.arange(pairs["P"], device=DEV), inf_new[1].argmax(1)]
        pairs["_in"], pairs["_wn"], pairs["_node"] = inf_new[0], inf_new[1], node_of_pair
        # gated (pre-cap) identities and node assignment, kept for the K2a selection probe
        ctx["gated_ids"] = torch.stack([pairs["src_uid"].long(), pairs["src_pix"].long()], 1).cpu()
        ctx["gated_node"] = node_of_pair.cpu()
        ctx["gated_omega"] = None if pairs.get("omega") is None else pairs["omega"].cpu()
        pairs = cap_pairs(pairs, node_of_pair, cfg["corr"]["max_per_node"], cfg["corr"]["max_total"],
                          seed + int(cfg["corr"].get("seed_offset", 0)),
                          omega=pairs.get("omega"), select=cfg["corr"].get("select", "random"), img_w=W,
                          salt=int(cfg["corr"].get("hash_salt", HASH_SALT)))
        pairs["P"] = int(pairs["n"].shape[0])
        inf_new = (pairs.pop("_in"), pairs.pop("_wn"))
        covered[pairs.pop("_node")] = True
        inf_old = influence(pairs["x_old"].float(), pairs["tau_old"], gN, nodes["t"], nc["r_node"], dt,
                            nc["K"], nc["K_cand"], nc["beta"])
    log["corr_capped"] = pairs["P"]
    log["n_nodes_with_pairs"] = int(covered.sum())
    ctx["covered_nodes"] = covered
    ts["corr"] = time.perf_counter() - t0_
    ctx.update({"pairs": pairs, "inf_old": inf_old, "inf_new": inf_new})
    if pairs["P"] == 0:
        return done("few_corr")
    if pairs["P"] < cfg["accept"]["min_corr"] and cfg["accept"].get("enforce", True):  # A1
        return done("few_corr")
    return done(None)


@torch.no_grad()
def correct_map(inp, cfg, variant, cache=None):
    """plan v4 A3: the whole event runs inside det_scope / torch_det_scope when cfg["det"] asks for it."""
    det = cfg.get("det") or {}
    warn_sink = []
    with det_scope(det.get("render", False)), torch_det_scope(det.get("torch", False), warn_sink):
        res = _correct_map(inp, cfg, variant, cache)
    if det.get("torch", False):
        res["log"]["det_warnings"] = list(warn_sink)
    return res


def _correct_map(inp, cfg, variant, cache=None):
    t_all = time.perf_counter()
    ctx = _prepare(inp, cfg, variant, cache)
    ts = dict(ctx["ts"])
    log = copy.deepcopy(ctx["log"])
    t_prep = ctx["t_prepare"]
    seed = int(inp.get("seed", 0))
    g = inp["g"]
    nc = cfg["nodes"]
    dt = ctx["dt"]
    xyz_rig, rot_rig, poses_rig, G_rig = ctx["xyz_rig"], ctx["rot_rig"], ctx["poses_rig"], ctx["G_rig"]
    extra = {"J_opt": ctx.get("J_opt", []), "J_eval": ctx.get("J_eval", [])}
    cached = cache is not None

    def elapsed():  # event time as if computed from scratch
        return (t_prep + (time.perf_counter() - t_all)) if cached else (time.perf_counter() - t_all)

    def fallback(reason):
        log["accepted"] = False
        log["fallback_reason"] = reason
        ts["total"] = elapsed()
        log["t_stage_s"] = ts
        return {"xyz": xyz_rig, "rot": rot_rig, "poses": poses_rig, "accepted": False, "reason": reason,
                "log": log, "G_rig": G_rig, "poses_rig": poses_rig, **extra}

    if ctx["fallback"] is not None:
        return fallback(ctx["fallback"])
    nodes, edges, S = ctx["nodes"], ctx["edges"], ctx["S"]
    gN = nodes["g"]
    Kn = gN.shape[0]
    R0, t_init = ctx["R0"], ctx["t_init"]
    x_pre, q_pre = ctx["x_pre"], ctx["q_pre"]
    idx_g, w_g, dmax_g = ctx["idx_g"], ctx["w_g"], ctx["dmax_g"]
    kf_uids = ctx["kf_uids"]

    # [6] solve
    prob = Problem(gN, R0, t_init, edges, ctx["pairs"], ctx["inf_old"], ctx["inf_new"], cfg)
    theta0 = torch.cat([torch.zeros((Kn, 3), dtype=torch.float64, device=DEV), t_init], 1)
    with torch.enable_grad():
        theta, slog = solve(prob, theta0, cfg)
    ts["solve"] = slog["solve_s"]
    log.update({"E_init": slog["E_init"], "E_final": slog["E_final"], "lbfgs_iters": slog["lbfgs_iters"]})
    if "grad" in slog:
        log["solver_grad"] = slog["grad"]
    Rn, tn, om = prob.unpack(theta)
    extra["node_t"], extra["node_rot"] = tn.detach().cpu(), om.detach().cpu()  # plan v4 K1/K2 probes
    r0 = prob.residual_con(*prob.unpack(theta0)[:2]).abs()
    r1 = prob.residual_con(Rn, tn).abs()
    log["edl_opt_init_mm"] = 1e3 * float(r0.median())
    log["edl_opt_final_mm"] = 1e3 * float(r1.median())
    log["frac_res_gt30mm"] = float((r1 > 0.03).float().mean())  # plan v2 proxy for wrong pairs
    disp = (tn - t_init).norm(dim=1)
    rotd = torch.rad2deg(om.norm(dim=1))
    log["max_node_disp_m"] = float(disp.max())
    log["median_node_disp_m"] = float(disp.median())
    log["max_node_rot_deg"] = float(rotd.max())
    log["median_node_rot_deg"] = float(rotd.median())
    # [7] acceptance A2-A4 (A2 on the weighted median when pairs carry omega, plan v2 F1w)
    ac = cfg["accept"]
    enforce = ac.get("enforce", True)
    w_pair = ctx["pairs"].get("omega")
    m0, m1 = float(wmedian(r0, w_pair)), float(wmedian(r1, w_pair))
    checks = {"A1_pairs": int(ctx["pairs"]["P"]), "A1": int(ctx["pairs"]["P"]) >= ac["min_corr"],
              "A2_ratio": m1 / m0 if m0 > 0 else None, "A2": m1 <= (1.0 - ac["min_gain"]) * m0,
              "A3": float(disp.max()) <= ac["max_disp"] and float(rotd.max()) <= ac["max_rot_deg"]}
    log["accept_checks"] = checks
    if not slog["finite"]:
        return fallback("nonfinite")
    if not checks["A2"] and enforce:
        return fallback("no_gain")
    if not checks["A3"] and enforce:
        return fallback("too_large")
    checks["A4_pre_apply"] = elapsed() <= ac["time_budget_s"]
    if not checks["A4_pre_apply"] and enforce:
        return fallback("timeout")

    # [8] apply
    t0_ = time.perf_counter()
    xyz_new, rot_new, _ = apply_to_gaussians(x_pre, q_pre, idx_g, w_g, gN, Rn, tn)
    ts["apply"] = time.perf_counter() - t0_
    if not bool(torch.isfinite(xyz_new).all()):
        return fallback("nonfinite")
    # [9] pose sync (source state: V-A rigid map at P*, V-B pre map at P)
    t0_ = time.perf_counter()
    src_poses = inp["poses_pgo"] if variant == "A" else inp["poses_pre"]
    G_src = with_pos(g, x_pre, q_pre)
    field = {"g": gN, "R": Rn, "t": tn, "t_nodes": nodes["t"], "r_node": nc["r_node"], "delta_t": dt,
             "K": nc["K"], "K_cand": nc["K_cand"], "beta": nc["beta"]}
    all_uids = sorted(inp["poses_pgo"].keys())
    poses_new, _, plog = sync_poses(src_poses, G_src, xyz_new, S, kf_uids, all_uids, inp["intr"], field, cfg)
    if cfg["pose_sync"].get("anchor", "none") == "prior":
        xyz_new, rot_new, poses_new, alog = anchor_gauge(inp, src_poses, xyz_new, rot_new, poses_new)
        log["anchor"] = alog
    ts["pose_sync"] = time.perf_counter() - t0_
    log["pose_sync"] = plog
    if cfg["diagnostics"].get("stretch_samples", 0):
        log["stretch"] = stretch_diag(x_pre, idx_g, w_g, dmax_g, gN, Rn, tn, cfg["diagnostics"]["stretch_samples"],
                                      cfg["diagnostics"].get("stretch_h", 0.01), seed)
    ts["total"] = elapsed()
    log["t_stage_s"] = ts
    checks["A4"] = ts["total"] <= ac["time_budget_s"]
    if not checks["A4"] and enforce:
        return fallback("timeout")
    log["accepted"] = True
    log["fallback_reason"] = None
    return {"xyz": xyz_new, "rot": rot_new, "poses": poses_new, "accepted": True, "reason": None, "log": log,
            "G_rig": G_rig, "poses_rig": poses_rig, **extra}


class _StateLike:
    """Minimal adaptor so deform.metrics helpers can render cameras from `inp`."""

    def __init__(self, inp, kf_uids):
        self.intr = inp["intr"]
        self.kf_uids = kf_uids

    def cam(self, uid, which="pgo", poses=None):
        return make_cam(uid, poses[uid], self.intr)


def gauss_from_dump(g):
    """Activated Gaussian tensors on DEV. With the raw parameters (plan v4 dumps) the activations are applied
    here in float32 exactly as GaussianModel does, so offline == online input bit for bit."""
    if "scale_raw" in g:
        return {"xyz": g["xyz"].float().to(DEV), "rot": torch.nn.functional.normalize(g["rot_raw"].float().to(DEV)),
                "scale": torch.exp(g["scale_raw"].float().to(DEV)), "opacity": torch.sigmoid(g["opacity_raw"].float().to(DEV)),
                "f_dc": g["f_dc"].float().to(DEV)}
    return {"xyz": g["xyz"].float().to(DEV), "rot": g["rot"].float().to(DEV), "scale": g["scale"].float().to(DEV),
            "opacity": g["opacity"].float().to(DEV), "f_dc": g["f_dc"].float().to(DEV)}


def inp_from_dump(dump, frame_fn):
    g = dump["gauss_pre"]
    cfg = dump["config"]
    return {
        "g": gauss_from_dump(g),
        "t0": g["t0"].long().to(DEV), "tc": g["tc"].long().to(DEV), "active": g["active"].bool().to(DEV),
        "kf_uids": list(dump["keyframe_uids"]), "all_cam_ids": list(dump["all_cam_ids"]),
        "poses_pre": dump["poses_pre"], "poses_pgo": dump["poses_pgo"], "intr": dump["intrinsics"],
        "frame_fn": frame_fn, "tr": cfg["Training"],
        "W": int(dump["meta"].get("old_than_N_keyframe", cfg["Training"]["old_than_N_keyframe"])),
        "seed": int(dump["meta"].get("seed", 0)),
        # plan v4 A4: frames pinned by a unary prior in the pose graph (older dumps: the first frame)
        "anchor_uids": [int(u) for u in dump.get("anchor_uids", [min(int(x) for x in dump["all_cam_ids"])])],
    }
