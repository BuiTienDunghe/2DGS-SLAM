"""Offline experiment P vs D on loop dumps: can the deformation graph close a global loop without PGO?

usage: python tools/pd_experiment.py --out DIR [--d-active] [--only-event CUR] [--append] RUN_DIR [RUN_DIR ...]
Nothing in deform/ or the SLAM pipeline is changed: the pinned energy lives here and is handed to the pipeline only
for the D+fine branches of this tool. Every loop event starts from the dumped state BEFORE PGO (gauss_pre, poses_pre):
  P0      PGO + rigid fix (upstream): poses_pgo of the dump, map moved rigidly per keyframe
  P1      P0 + deformation with configs/deform/selected_v6.yaml (correct_map, unchanged)
  D       no PGO, coarse deformation graph
            H = P_L m P_cur^-1 (active-region coordinates -> old-region frame); m = loop measurement of the event
                (stored loop_transform, else recovered by tools/pgo_replay.py), P = poses_pre
            E_coarse  5000 hash-selected new-layer Gaussians (t0 >= s_k) visible in the current view, target H x
                      (point-to-point through the existing "con" term; the target hangs on a pinned node)
            E_pin     nodes with t <= L hard-pinned; E_reg as in the pipeline (temporal chain included); w_p = 0
            init      nodes after L: Exp(a Log H), a = clip((t - L) / (cur - L), 0, 1)
            solver    existing LBFGS, tol_mode grad (V1)
            poses     keyframes after L: Kabsch on the nodes of their own scan (|t_node - uid| <= Delta_t) that
                      dominate the Gaussians they see (g -> g + t); other frames follow the nearest keyframe
            check     two layers rendered at the current view: >= 500 pixel pairs and median gap < 50 mm,
                      else "D reject" (the event then counts with P0 in the statistics)
  D+fine  D, then the pixel-pair deformation of the pipeline on D's result (E_con + E_reg + E_pin, w_p = 0), with
          the loop keyframes of P1 so that Pi* stays held out; the pipeline's acceptance A1-A4 decides whether the
          fine result is kept (else D+fine = D); the unaccepted result is still measured as "D+fine/raw"
  D-active (optional) as D but only the nodes of the active region are free (ElasticFusion style)
Diagnostics, not part of the pass rule: D@N = D's iterate after N LBFGS iterations of the same solve; D/poseG = D
with poses from a Kabsch on the keyframe's own Gaussians; D+fineP = the fine stage with the prior of selected_v6.
Hash selection, deterministic renders and torch deterministic mode as selected_v6.
Per event and branch: keyframe ATE right after the event (M.ate_kf), two-layer gap on Pi*(k) (birth-time layers,
Pi* from the P0 state as in v4/v5, pixels valid in every branch), pose deviation from P0, stretch (largest E_reg
edge residual / relative rotation of adjacent nodes, largest node displacement), solve and total time.
Writes DIR/pd.jsonl (one row per event).
"""
import argparse
import contextlib
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import gtsam  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform import pipeline as PL  # noqa: E402
from deform.apply import apply_to_gaussians, kabsch  # noqa: E402
from deform.correspondences import splitmix64  # noqa: E402
from deform.dump import EventState, list_dumps, load_dump  # noqa: E402
from deform.field import blend_quat, phi, quat_of, quat_to_rotmat  # noqa: E402
from deform.influence import influence  # noqa: E402
from deform.nodes import build_edges, build_nodes  # noqa: E402
from deform.pipeline import correct_map, delta_T_dict, delta_t_frames, inp_from_dump, rigid_result, with_pos  # noqa: E402
from deform.render_utils import DEV, det_scope, make_cam, render_subset, torch_det_scope  # noqa: E402
from deform.rigid import quat_mul  # noqa: E402
from deform.solver import Problem, solve  # noqa: E402
from pgo_replay import Replay, load_attempts, mat  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_COARSE = 5000
CHECK_MIN_PAIRS, CHECK_MAX_GAP = 500, 0.050
SNAP_AT = (500, 1500)  # LBFGS iterations at which D's iterate is also evaluated (diagnostic)


class PinnedProblem(Problem):
    """Energy with hard-pinned nodes: the rows of theta that belong to pinned nodes are replaced by their initial
    values before every evaluation, so their gradient is zero and LBFGS never moves them."""

    def __init__(self, *a, pin=None):
        super().__init__(*a)
        self.pin = pin.reshape(-1, 1).to(self.g.device)
        self.theta_ref = torch.cat([torch.zeros_like(self.t_init), self.t_init], 1)

    def terms(self, theta):
        return super().terms(torch.where(self.pin, self.theta_ref, theta))


@contextlib.contextmanager
def pipeline_pinned(pin, JL):
    """D+fine: the pipeline builds its Problem with pinned nodes and uses the loop keyframes JL (those of P1)."""
    orig_p, orig_s = PL.Problem, M.select_loop_kfs
    if pin is not None:
        PL.Problem = lambda *a: PinnedProblem(*a, pin=pin)
    if JL is not None:
        M.select_loop_kfs = lambda st, G, poses, mo, mn, cfg: (list(JL), [])
    try:
        yield
    finally:
        PL.Problem, M.select_loop_kfs = orig_p, orig_s


@contextlib.contextmanager
def lbfgs_snapshots(at, store):
    """Copy the LBFGS iterate after the given numbers of steps (the solve itself is untouched)."""
    orig = torch.optim.LBFGS.step
    cnt = {"n": 0, "t0": time.perf_counter()}

    def step(self, closure):
        out = orig(self, closure)
        cnt["n"] += 1
        if cnt["n"] in at:
            store[cnt["n"]] = (self._params[0].detach().clone(), time.perf_counter() - cnt["t0"])
        return out

    torch.optim.LBFGS.step = step
    try:
        yield
    finally:
        torch.optim.LBFGS.step = orig


def so3(Rm):
    U, _, Vt = np.linalg.svd(Rm)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def se3_interp(H, a):
    """Exp(a_i Log H) for every a_i -> [n,4,4] float64 (gtsam Pose3 chart)."""
    xi = gtsam.Pose3.Logmap(gtsam.Pose3(H))
    vals, inv = np.unique(a, return_inverse=True)
    tab = np.stack([gtsam.Pose3.Expmap(float(v) * xi).matrix() for v in vals])
    return tab[inv]


def rot_angle_deg(R):
    """Rotation angle from ||R - I||_F = 2 sqrt(2) sin(theta / 2): exact for small angles too."""
    return float(np.degrees(2.0 * np.arcsin(min(1.0, np.linalg.norm(R - np.eye(3)) / (2.0 * np.sqrt(2.0))))))


def angles_deg(R):
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    return torch.rad2deg(2.0 * torch.arcsin(((R - eye).flatten(-2).norm(dim=-1) / (2.0 * 2.0 ** 0.5)).clamp(max=1.0)))


def pose_dev(Pa, Pb, uids):
    dt, dr = [], []
    for u in uids:
        A, B = mat(Pa[u]), mat(Pb[u])
        dt.append(float(np.linalg.norm(A[:3, 3] - B[:3, 3])))
        dr.append(rot_angle_deg(A[:3, :3].T @ B[:3, :3]))
    if not dt:
        return None
    return {"t_med_mm": 1e3 * float(np.median(dt)), "t_max_mm": 1e3 * float(np.max(dt)),
            "r_med_deg": float(np.median(dr)), "r_max_deg": float(np.max(dr)), "n": len(dt)}


def layer_pairs_view(cam, G, m_old, m_new, cfg):
    """Pixel pairs of the two layers in one view (stride grid, both alpha > 0.95, normals agree, no depth edge)."""
    c = cfg["corr"]
    Hh, Ww = int(cam.image_height), int(cam.image_width)
    L = M.render_layers(cam, G, m_old, m_new)
    m = M.stride_grid(Hh, Ww, c["stride"]) & (L["Oo"] > M.ALPHA_THR) & (L["On"] > M.ALPHA_THR)
    m &= (L["No"] * L["Nn"]).sum(0) > c["cos_theta_n"]
    m &= ~(M.edge_mask(L["Do"], c["g_max"]) | M.edge_mask(L["Dn"], c["g_max"]))
    gap = M.layer_gap(cam, L).abs()[m]
    return int(m.sum()), (None if gap.numel() == 0 else float(gap.median()))


def build_graph(inp, rel, cfg):
    g = inp["g"]
    x_pre = g["xyz"].float()
    alpha = g["opacity"].reshape(-1)
    kf_uids = sorted(inp["kf_uids"])
    dt = delta_t_frames(kf_uids, inp["W"])
    nc = cfg["nodes"]
    t0_ = time.perf_counter()
    nodes = build_nodes(x_pre, inp["t0"], inp["tc"], rel["S"], rel["R"], alpha, nc["r_node"], dt, nc.get("min_alpha", 0.1))
    edges = build_edges(nodes, nc["r_node"], dt)
    idx_g, w_g = influence(x_pre, inp["t0"].float(), nodes["g"], nodes["t"], nc["r_node"], dt, nc["K"], nc["K_cand"], nc["beta"])
    Kn = nodes["g"].shape[0]
    act = inp["active"].reshape(-1)[nodes["cand"]].float()
    n_act = torch.zeros(Kn, device=DEV).scatter_add_(0, nodes["node_of_cand"], act)
    # visibility of every keyframe on the pre state, once (pose extraction)
    alpha_ok = alpha > 0.5
    return {"nodes": nodes, "edges": edges, "idx_g": idx_g, "w_g": w_g, "dt": dt, "kf_uids": kf_uids, "alpha_ok": alpha_ok,
            "top": idx_g[torch.arange(idx_g.shape[0], device=DEV), w_g.argmax(1)],
            "node_active": n_act / nodes["n_members"].clamp_min(1) > 0.5, "t_graph_s": time.perf_counter() - t0_}


def coarse(inp, H_np, rel, cfg, graph, m_new, pin_mode, snap_at=()):
    """Branch D (pin_mode "time": nodes with t <= L pinned) or D-active ("active": only active-region nodes free).
    Returns {"": result, "@N": result of the iterate after N LBFGS steps, ...}; result = dict(xyz, rot, poses,
    poses_gauss, log, reject)."""
    t_all = time.perf_counter()
    g = inp["g"]
    x_pre, q_pre = g["xyz"].float(), g["rot"].float()
    nodes, edges, idx_g, w_g, dt, kf_uids = (graph[k] for k in ("nodes", "edges", "idx_g", "w_g", "dt", "kf_uids"))
    nc = cfg["nodes"]
    L, C = int(inp["loop_uid"]), int(inp["cur_uid"])
    gN = nodes["g"]
    gd = gN.double()
    Kn = gN.shape[0]
    t_nodes = nodes["t"].double()
    pin = (nodes["t"] <= float(L)) if pin_mode == "time" else ~graph["node_active"]
    log0 = {"n_nodes": Kn, "n_edges": int(edges.shape[0]), "n_pinned": int(pin.sum()), "delta_t": dt, "t_graph_s": graph["t_graph_s"]}
    if int(pin.sum()) == 0 or int((~pin).sum()) == 0 or edges.shape[0] == 0 or C <= L:
        return {"": {"reject": "degenerate graph (no pinned / no free node / no edge)", "log": log0}}
    # ---- init: nodes after L get Exp(a Log H), a = clip((t - L) / (C - L), 0, 1); pinned nodes identity
    a = ((t_nodes - L) / float(C - L)).clamp(0, 1).cpu().numpy()
    Ts = torch.from_numpy(se3_interp(H_np, a)).to(DEV)
    R0 = Ts[:, :3, :3].contiguous()
    t_init = (R0 @ gd[..., None]).squeeze(-1) + Ts[:, :3, 3] - gd
    # torch.where, not masked assignment: index_put_ with a broadcast value fails in torch deterministic mode
    R0 = torch.where(pin[:, None, None], torch.eye(3, dtype=torch.float64, device=DEV)[None], R0)
    t_init = torch.where(pin[:, None], torch.zeros_like(t_init), t_init)
    # ---- E_coarse: new-layer Gaussians visible in the current view, hash-selected
    cam = make_cam(C, inp["poses_pre"][C], inp["intr"])
    Cn = render_subset(cam, g, m_new)["contrib_full"]
    vis = torch.nonzero((Cn > 0.5) & m_new & graph["alpha_ok"]).reshape(-1)
    log0["n_visible_new"] = int(vis.numel())
    if vis.numel() < 100:
        return {"": {"reject": f"only {int(vis.numel())} new-layer Gaussians in the current view", "log": log0}}
    h = splitmix64(vis.cpu().numpy().astype(np.uint64))
    sel = vis[torch.from_numpy(np.argsort(h, kind="stable")[:N_COARSE].copy()).to(DEV)]
    Hm = torch.from_numpy(H_np).to(DEV)
    xs = x_pre[sel].double()
    ys = xs @ Hm[:3, :3].T + Hm[:3, 3]
    Mn, K = int(sel.numel()), idx_g.shape[1]
    io = torch.full((Mn, K), int(torch.nonzero(pin).reshape(-1)[0]), dtype=torch.long, device=DEV)
    wo = torch.zeros((Mn, K), dtype=torch.float32, device=DEV)
    wo[:, 0] = 1.0  # phi(target) = target: the node is pinned at identity
    nrm = torch.zeros((Mn, 3), dtype=torch.float64, device=DEV)
    nrm[:, 2] = 1.0
    cfgD = copy.deepcopy(cfg)
    cfgD["corr"]["residual"] = "p2p"
    cfgD["energy"]["w_p"] = 0.0
    theta0 = torch.cat([torch.zeros((Kn, 3), dtype=torch.float64, device=DEV), t_init], 1)
    prob = PinnedProblem(gN, R0, t_init, edges, {"x_old": ys, "x_new": xs, "n": nrm, "P": Mn}, (io, wo), (idx_g[sel], w_g[sel]), cfgD, pin=pin)
    snaps = {}
    with torch.enable_grad(), lbfgs_snapshots(set(snap_at), snaps):
        theta_fin, slog = solve(prob, theta0, cfgD)
    r0 = prob.residual_con(*prob.unpack(theta0)[:2])
    log0.update({"n_coarse": Mn, "E_init": slog["E_init"], "coarse_res_init_mm": 1e3 * float(r0.median())})
    t_setup = time.perf_counter() - t_all - slog["solve_s"]
    # keyframe visibility on the pre state (shared by every iterate that is evaluated)
    t0_ = time.perf_counter()
    S = rel["S"].reshape(-1).double()
    t0f = inp["t0"].reshape(-1).double()
    vis_kf = {}
    for uid in kf_uids:
        if uid > L:
            Cj = render_subset(make_cam(uid, inp["poses_pre"][uid], inp["intr"]), g)["contrib_full"]
            V = (Cj > 0.5) & graph["alpha_ok"]
            nd, cnt = torch.unique(graph["top"][V], return_counts=True)
            own = (t_nodes[nd] - float(uid)).abs() <= dt
            Vg = torch.nonzero(V & ((t0f - float(uid)).abs() <= dt)).reshape(-1)
            vis_kf[uid] = (nd[own], cnt[own].double(), Vg, Cj[Vg].double() * S[Vg].clamp_min(0.05))
    t_vis = time.perf_counter() - t0_
    eye = torch.eye(4, dtype=torch.float64, device=DEV)
    kf_t = torch.tensor(kf_uids, dtype=torch.float64)
    k, l = edges[:, 0], edges[:, 1]

    def finish(theta, solve_s, iters, ginfo):
        t_f = time.perf_counter()
        log = dict(log0)
        Rn, tn, om = prob.unpack(theta)
        with torch.no_grad():
            E1 = {kk: float(v) for kk, v in prob.terms(theta).items()}
        r1 = prob.residual_con(Rn, tn)
        log.update({"solve_s": solve_s, "lbfgs_iters": iters, "solver_grad": ginfo, "E_final": E1,
                    "coarse_res_final_mm": 1e3 * float(r1.median()), "coarse_res_final_p90_mm": 1e3 * float(torch.quantile(r1, 0.9)),
                    "coarse_res_final_max_mm": 1e3 * float(r1.max())})
        if not bool(torch.isfinite(theta).all()):
            return {"reject": "nonfinite solution", "log": log}
        # stretch: E_reg residual per edge, relative rotation of adjacent nodes, node motion (from init and absolute)
        r_kl = (Rn[k] @ (gd[l] - gd[k])[..., None]).squeeze(-1) + gd[k] + tn[k] - gd[l] - tn[l]
        r_lk = (Rn[l] @ (gd[k] - gd[l])[..., None]).squeeze(-1) + gd[l] + tn[l] - gd[k] - tn[k]
        edge_res = torch.maximum(r_kl.norm(dim=1), r_lk.norm(dim=1))
        rel_ang = angles_deg(Rn[k].transpose(1, 2) @ Rn[l])
        i_max = int(edge_res.argmax())
        log.update({"edge_res_max_m": float(edge_res.max()), "edge_res_p99_m": float(torch.quantile(edge_res, 0.99)), "edge_res_med_m": float(edge_res.median()),
                    "edge_res_max_edge_len_m": float((gd[l] - gd[k]).norm(dim=1)[i_max]), "edge_res_max_dt_frames": float((t_nodes[k[i_max]] - t_nodes[l[i_max]]).abs()),
                    "edge_rot_max_deg": float(rel_ang.max()), "edge_rot_p99_deg": float(torch.quantile(rel_ang, 0.99)),
                    "n_edges_gt_0p10m": int((edge_res > 0.10).sum()), "n_edges_gt_5deg": int((rel_ang > 5.0).sum()),
                    "node_disp_abs_max_m": float(tn.norm(dim=1).max()), "node_rot_abs_max_deg": float(angles_deg(Rn).max()),
                    "node_disp_from_init_max_m": float((tn - t_init).norm(dim=1).max()), "node_rot_from_init_max_deg": float(torch.rad2deg(om.norm(dim=1)).max()),
                    "dtheta_to_final_max_m": float((theta[:, 3:] - theta_fin[:, 3:]).norm(dim=1).max())})
        xyz_new, rot_new, _ = apply_to_gaussians(x_pre, q_pre, idx_g, w_g, gN, Rn, tn)
        # poses of the keyframes after L: Kabsch on the nodes around each keyframe (diagnostic: on its own Gaussians)
        a_all, b_all = gd, gd + tn
        Ms, Mg, n_fb, n_fb_g, n_used = {}, {}, 0, 0, []
        for uid in kf_uids:
            if uid <= L:
                Ms[uid] = Mg[uid] = eye
                continue
            nd, cnt, Vg, wg = vis_kf[uid]
            n_used.append(int(nd.numel()))
            if nd.numel() >= 4:
                Ms[uid] = kabsch(a_all[nd], b_all[nd], cnt)
            else:  # field at the camera centre (as apply.sync_poses does)
                n_fb += 1
                o = torch.as_tensor(inp["poses_pre"][uid], dtype=torch.float64)[:3, 3].to(DEV)[None]
                ii, ww = influence(o.float(), torch.tensor([float(uid)], device=DEV), gN, nodes["t"], nc["r_node"], dt, nc["K"], nc["K_cand"], nc["beta"])
                po = phi(o, ii, ww, gd, Rn, tn)[0]
                Rb = quat_to_rotmat(blend_quat(ii, ww, quat_of(Rn)).double())[0]
                Mj = eye.clone()
                Mj[:3, :3] = Rb
                Mj[:3, 3] = po - Rb @ o[0]
                Ms[uid] = Mj
            if int(Vg.numel()) >= cfg["pose_sync"]["min_gauss"]:
                Mg[uid] = kabsch(x_pre[Vg].double(), xyz_new[Vg].double(), wg)
            else:
                n_fb_g += 1
                Mg[uid] = Ms[uid]
        poses, poses_g = {}, {}
        for uid in sorted(inp["poses_pre"].keys()):
            j = uid
            if uid not in Ms:
                dist = (kf_t - float(uid)).abs()
                j = kf_uids[int(torch.nonzero(dist == dist.min())[0])]
            P = torch.as_tensor(inp["poses_pre"][uid], dtype=torch.float64).to(DEV)
            poses[uid], poses_g[uid] = (Ms[j] @ P).cpu(), (Mg[j] @ P).cpu()
        log.update({"pose_fallbacks": n_fb, "pose_gauss_fallbacks": n_fb_g,
                    "pose_nodes_min": min(n_used) if n_used else None, "pose_nodes_med": float(np.median(n_used)) if n_used else None})
        # time as if this iterate had been the end of the solve; the reliability stage is added by the caller
        log["t_stages_s"] = graph["t_graph_s"] + t_setup + solve_s + t_vis + (time.perf_counter() - t_f)
        return {"xyz": xyz_new, "rot": rot_new, "poses": poses, "poses_gauss": poses_g, "log": log, "reject": None}

    out = {"": finish(theta_fin, slog["solve_s"], slog["lbfgs_iters"], slog.get("grad"))}
    for n in sorted(snaps):
        th, ts = snaps[n]
        out[f"@{n}"] = finish(th, ts, n, {"converged": False, "note": "iterate of the same solve"})
    return out


def fine_stage(inp, state_G, poses, cfg, over, rel, L, JL, variant, sink, cacheF=None):
    """Pixel-pair deformation of the pipeline on a D result, nodes with t <= L pinned. Runs unenforced and applies the
    acceptance A1-A4 here, so that a refused result can still be measured. Returns (branch log, raw result | None, cacheF)."""
    inpD = dict(inp)
    inpD["g"] = state_G
    inpD["poses_pre"] = inpD["poses_pgo"] = poses
    cfgF = dcfg._merge(cfg, _deep({"accept": {"enforce": False}}, over))
    if cacheF is None:
        cacheF = {PL.rel_key(cfgF): rel}
    t0_ = time.perf_counter()
    with pipeline_pinned(None, JL):
        with torch.no_grad(), det_scope(True), torch_det_scope(True, sink):
            ctxF = PL._prepare(inpD, cfgF, variant, cacheF)
    pinF = (ctxF["nodes"]["t"] <= float(L)) if "nodes" in ctxF else None
    if pinF is None or int(pinF.sum()) == 0 or ctxF.get("fallback"):
        return {"accepted": False, "reason": ctxF.get("fallback") or "no pinned node"}, None, cacheF
    with pipeline_pinned(pinF, JL):
        res = correct_map(inpD, cfgF, variant, cache=cacheF)
    lf = res["log"]
    ck = lf.get("accept_checks") or {}
    reason = res["reason"]
    if reason is None:
        for key, why in (("A1", "few_corr"), ("A2", "no_gain"), ("A3", "too_large"), ("A4_pre_apply", "timeout"), ("A4", "timeout")):
            if ck.get(key) is False:
                reason = why
                break
    b = {"accepted": reason is None, "reason": reason, "t_fine_s": lf["t_stage_s"].get("total"), "solve_s": lf["t_stage_s"].get("solve"),
         "iters": lf.get("lbfgs_iters"), "pairs": lf.get("corr_capped"), "n_pinned": int(pinF.sum()), "accept_checks": ck,
         "edl_opt_init_mm": lf.get("edl_opt_init_mm"), "edl_opt_final_mm": lf.get("edl_opt_final_mm"),
         "max_node_disp_m": lf.get("max_node_disp_m"), "max_node_rot_deg": lf.get("max_node_rot_deg"), "median_node_disp_m": lf.get("median_node_disp_m"),
         "converged": (lf.get("solver_grad") or {}).get("converged"), "grad_ratio": (lf.get("solver_grad") or {}).get("grad_ratio"),
         "wall_s": time.perf_counter() - t0_}
    if res.get("node_t") is not None:  # where the largest node motion sits
        disp = res["node_t"].norm(dim=1)
        cov = ctxF["covered_nodes"].cpu()
        i = int(disp.argmax())
        tn_ = ctxF["nodes"]["t"].cpu()
        b.update({"disp_max_node_time": float(tn_[i]), "disp_max_node_has_pairs": bool(cov[i]),
                  "disp_max_with_pairs_m": float(disp[cov].max()) if bool(cov.any()) else None,
                  "disp_max_without_pairs_m": float(disp[~cov].max()) if bool((~cov).any()) else None,
                  "n_nodes_gt_0p10m": int((disp > 0.10).sum()), "n_nodes_with_pairs": int(cov.sum())})
    raw = res if res["accepted"] else None
    return b, raw, cacheF


def _deep(a, b):
    return dcfg._merge(a, b)


def fmt(x, nd=1, scale=1.0):
    return "—" if x is None else f"{scale * x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "deform", "selected_v6.yaml"))
    ap.add_argument("--only-event", type=int, default=None, help="cur_uid of the single event to run (debug)")
    ap.add_argument("--d-active", action="store_true", help="also run the optional D-active branch")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--debug-max-iter", type=int, default=None, help="cap of the LBFGS iterations (code-path test only)")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    user = load_deform_config(a.config)
    out_path = os.path.join(a.out, "pd.jsonl")
    if not a.append:
        open(out_path, "w").close()
    n_rows = 0
    for rd in a.runs:
        dumps = list_dumps(rd)
        D = [load_dump(p) for p in dumps]
        loops, checks = Replay(D, load_attempts(rd), D[0]["config"]["Training"]).recover_all()
        for d in D:
            st = EventState(d)
            C, L = int(d["meta"]["cur_uid"]), int(d["meta"]["loop_uid"])
            if a.only_event is not None and C != a.only_event:
                continue
            t_ev = time.time()
            torch.manual_seed(0)
            row = {"run": os.path.basename(rd), "event": f"{C}<->{L}", "cur": C, "loop": L, "event_id": d["meta"]["event_id"],
                   "raw_params": "scale_raw" in d["gauss_pre"], "n_gauss": int(d["gauss_pre"]["xyz"].shape[0])}

            def emit():
                with open(out_path, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")

            if (L, C) not in loops:
                row["note"] = "loop factor not kept in the graph (graph error < 50): no loop measurement, event skipped"
                emit()
                n_rows += 1
                print(f"== {row['run']} {row['event']}: {row['note']}", flush=True)
                continue
            chk = checks[(L, C)]
            row["loop_meas"] = {"source": chk["source"], "lm_drift_m": chk["lm_drift_from_xstar_m"], "err_at_xstar": chk["err_at_xstar"],
                                "logged_pgo_err_after": chk["logged_pgo_err_after"]}
            cfg = dcfg.resolve(user, st.config)
            cfg["seed"] = int(d["meta"].get("seed", 0))
            if a.debug_max_iter:
                cfg["solver"]["max_iter_conv"] = a.debug_max_iter
            variant = cfg["variant"]
            inp = inp_from_dump(d, st.frame)
            P_L, P_C = mat(inp["poses_pre"][L]), mat(inp["poses_pre"][C])
            H_np = P_L @ loops[(L, C)] @ np.linalg.inv(P_C)
            H_np[:3, :3] = so3(H_np[:3, :3])
            dPGO = mat(inp["poses_pgo"][C]) @ np.linalg.inv(P_C)  # what PGO did to the current frame
            row["H"] = {"t_mm": 1e3 * float(np.linalg.norm(H_np[:3, 3])), "rot_deg": rot_angle_deg(H_np[:3, :3]),
                        "cur_shift_mm": 1e3 * float(np.linalg.norm((H_np @ P_C)[:3, 3] - P_C[:3, 3])),
                        "pgo_cur_shift_mm": 1e3 * float(np.linalg.norm((dPGO @ P_C)[:3, 3] - P_C[:3, 3])),
                        "H_vs_pgo_cur_mm": 1e3 * float(np.linalg.norm((H_np @ P_C)[:3, 3] - (dPGO @ P_C)[:3, 3])),
                        "H_vs_pgo_rot_deg": rot_angle_deg(H_np[:3, :3].T @ so3(dPGO[:3, :3]))}
            m_old, m_new = M.layer_masks_birth(inp["t0"], M.birth_split(C, L))
            G_pre = inp["g"]
            kf_uids = sorted(inp["kf_uids"])
            after_L = [u for u in kf_uids if u > L]
            row["n_kf"], row["n_kf_after_L"] = len(kf_uids), len(after_L)
            br = {}
            # ------------------------------------------------ P0: PGO + rigid
            dT = delta_T_dict(kf_uids, inp["all_cam_ids"], inp["poses_pre"], inp["poses_pgo"])
            xyz_rig, rot_rig, poses_rig = rigid_result(inp, dT)
            G_rig = with_pos(G_pre, xyz_rig, rot_rig)
            states = {"pre": (G_pre, {u: torch.as_tensor(v).double().cpu() for u, v in inp["poses_pre"].items()}), "P0": (G_rig, poses_rig)}
            # ------------------------------------------------ P1: P0 + deformation (selected_v6)
            cache, sink = {}, []
            t0_ = time.perf_counter()
            with torch.no_grad(), det_scope(True), torch_det_scope(True, sink):
                ctx1 = PL._prepare(inp, cfg, variant, cache)  # so that the timing below counts every stage exactly once
            res1 = correct_map(inp, cfg, variant, cache=cache)
            lg = res1["log"]
            br["P1"] = {"accepted": res1["accepted"], "reason": res1["reason"], "t_total_s": lg["t_stage_s"].get("total"),
                        "solve_s": lg["t_stage_s"].get("solve"), "iters": lg.get("lbfgs_iters"), "pairs": lg.get("corr_capped"),
                        "max_node_disp_m": lg.get("max_node_disp_m"), "max_node_rot_deg": lg.get("max_node_rot_deg"),
                        "wall_s": time.perf_counter() - t0_, "converged": (lg.get("solver_grad") or {}).get("converged")}
            states["P1"] = (with_pos(G_pre, res1["xyz"], res1["rot"]), res1["poses"])
            JL1 = sorted(set(res1.get("J_opt", [])) | set(res1.get("J_eval", [])))
            row["n_loop_kfs"] = len(JL1)
            rel = cache[PL.rel_key(cfg)]
            t_rel = float(rel["time_s"])
            row["t_reliability_s"] = t_rel
            # ------------------------------------------------ D (and D-active): coarse graph, no PGO
            with torch.no_grad(), det_scope(True), torch_det_scope(True, sink):
                # step 1 check: H applied to the new layer must reduce its gap to the old layer in the current view
                n_b, gap_b = layer_pairs_view(make_cam(C, inp["poses_pre"][C], inp["intr"]), G_pre, m_old, m_new, cfg)
                Hm = torch.from_numpy(H_np).to(DEV)
                xyz_H = torch.where(m_new[:, None], (G_pre["xyz"].double() @ Hm[:3, :3].T + Hm[:3, 3]).float(), G_pre["xyz"])
                qH = quat_of(Hm[:3, :3][None]).float().expand(G_pre["rot"].shape[0], 4)
                rot_H = torch.where(m_new[:, None], torch.nn.functional.normalize(quat_mul(qH, G_pre["rot"].float()), dim=-1), G_pre["rot"])
                n_a, gap_a = layer_pairs_view(make_cam(C, torch.from_numpy(H_np @ P_C), inp["intr"]), with_pos(G_pre, xyz_H, rot_H), m_old, m_new, cfg)
                row["H_check"] = {"pairs_before": n_b, "gap_before_mm": None if gap_b is None else 1e3 * gap_b,
                                  "pairs_after": n_a, "gap_after_mm": None if gap_a is None else 1e3 * gap_a,
                                  "ok": (gap_a is not None) and (gap_b is None or gap_a < gap_b)}
                del xyz_H, rot_H
                graph = build_graph(inp, rel, cfg)
                modes = [("D", "time", SNAP_AT)] + ([("D-active", "active", ())] if a.d_active else [])
                dres = {}
                for base, pm, snap in modes:
                    for suf, r in coarse(inp, H_np, rel, cfg, graph, m_new, pm, snap).items():
                        name = base + suf
                        b = dict(r["log"])
                        if r.get("reject") is None:
                            t0_ = time.perf_counter()
                            n6, gap6 = layer_pairs_view(make_cam(C, r["poses"][C], inp["intr"]), with_pos(G_pre, r["xyz"], r["rot"]), m_old, m_new, cfg)
                            b["check_pairs"], b["check_gap_mm"] = n6, None if gap6 is None else 1e3 * gap6
                            b["t_stages_s"] += time.perf_counter() - t0_
                            if n6 < CHECK_MIN_PAIRS or gap6 is None or gap6 >= CHECK_MAX_GAP:
                                r["reject"] = f"check 6: {n6} pixel pairs, median gap {fmt(gap6, 1, 1e3)} mm"
                        b["reject"] = r.get("reject")
                        if "t_stages_s" in b:
                            b["t_total_s"] = b["t_stages_s"] + t_rel  # reliability counted once, as in P1
                        br[name], dres[name] = b, r
                        if r.get("xyz") is not None:  # a rejected D is still measured (reported, not used in the statistics)
                            states[name] = (with_pos(G_pre, r["xyz"], r["rot"]), r["poses"])
                            if suf == "":
                                states[name + "/poseG"] = (None, r["poses_gauss"])
            # ------------------------------------------------ D+fine: pixel-pair deformation on D's result, pinned
            rD = dres["D"]
            if rD.get("reject") is None:
                bF, rawF, cacheF = fine_stage(inp, states["D"][0], rD["poses"], cfg, {"energy": {"w_p": 0.0}}, rel, L, JL1 or None, variant, sink)
                if bF.get("t_fine_s") is not None:
                    bF["t_total_s"] = br["D"]["t_stages_s"] + bF["t_fine_s"]
                br["D+fine"] = bF
                if rawF is not None:
                    raw_state = (with_pos(G_pre, rawF["xyz"], rawF["rot"]), rawF["poses"])
                    if bF["accepted"]:
                        states["D+fine"] = raw_state
                    else:
                        states["D+fine"] = states["D"]
                        states["D+fine/raw"] = raw_state
                else:
                    states["D+fine"] = states["D"]
                # diagnostic: the same fine stage with the prior of selected_v6 (shares the pairs of the stage above)
                bP, rawP, _ = fine_stage(inp, states["D"][0], rD["poses"], cfg, {}, rel, L, JL1 or None, variant, sink, cacheF)
                if bP.get("t_fine_s") is not None:
                    bP["t_total_s"] = br["D"]["t_stages_s"] + bP["t_fine_s"]
                br["D+fineP"] = bP
                if rawP is not None:
                    states["D+fineP" if bP["accepted"] else "D+fineP/raw"] = (with_pos(G_pre, rawP["xyz"], rawP["rot"]), rawP["poses"])
                del rawF, rawP, cacheF
            # ------------------------------------------------ metrics
            with torch.no_grad(), det_scope(True):
                Pi = None
                if res1.get("J_eval"):
                    Pi, _ = M.pi_star(st, G_rig, poses_rig, res1["J_eval"], m_old, m_new, cfg)
                    row["n_pi"] = int(sum(int(m.sum()) for m, _ in Pi.values()))
                maps = {}
                for name, (G, P) in states.items():
                    b = br.setdefault(name, {})
                    b["ate_kf_m"] = M.ate_kf(st, P)
                    if name not in ("P0", "pre"):
                        b["dev_vs_P0_afterL"] = pose_dev(P, poses_rig, after_L)
                        b["dev_vs_P0_all"] = pose_dev(P, poses_rig, kf_uids)
                    if Pi is not None and G is not None:
                        maps[name] = M.gap_maps(st, G, P, Pi, m_old, m_new)
                if Pi is not None:
                    # common pixel set: valid in the branches of the pass rule (a rejected D does not shrink it)
                    common = [n for n in ("P0", "P1", "D", "D+fine") if n in maps and not br[n].get("reject")]
                    row["gap_common_branches"] = common
                    n_common = 0
                    for name, mp in maps.items():
                        own, com = [], []
                        for u, (m, _) in Pi.items():
                            v = m & mp[u]["valid"]
                            own.append(mp[u]["gap"][v])
                            if name != "pre":
                                for o in common:
                                    v = v & maps[o][u]["valid"]
                                com.append(mp[u]["gap"][v])
                        own = torch.cat(own)
                        b = br[name]
                        b["gap_own_mm"] = None if own.numel() == 0 else 1e3 * float(own.median())
                        b["gap_valid_frac"] = own.numel() / max(1, row["n_pi"])
                        if com:
                            com = torch.cat(com)
                            if name in common:
                                n_common = int(com.numel())
                            b["gap_mm"] = None if com.numel() == 0 else 1e3 * float(com.median())
                            b["gap_p90_mm"] = None if com.numel() == 0 else 1e3 * float(torch.quantile(com.float(), 0.9))
                            b["gap_n"] = int(com.numel())
                    row["n_pi_common"] = n_common
            row["det_warnings"] = list(sink)
            row["branches"] = br
            row["event_wall_s"] = time.time() - t_ev
            emit()
            n_rows += 1
            hc = row["H_check"]
            print(f"== {row['run']} {row['event']}  H {row['H']['t_mm']:.0f} mm / {row['H']['rot_deg']:.2f} deg (cur shift {row['H']['cur_shift_mm']:.0f} mm, PGO {row['H']['pgo_cur_shift_mm']:.0f} mm, "
                  f"diff {row['H']['H_vs_pgo_cur_mm']:.0f} mm)  H check {fmt(hc['gap_before_mm'])} -> {fmt(hc['gap_after_mm'])} mm ({hc['pairs_before']} -> {hc['pairs_after']} px) ok={hc['ok']}  "
                  f"loop kfs {row['n_loop_kfs']}  Pi* {row.get('n_pi')} / common {row.get('n_pi_common')}  [{row['event_wall_s']:.0f} s]", flush=True)
            for name, b in br.items():
                dv = b.get("dev_vs_P0_afterL") or {}
                extra = ""
                if "edge_res_max_m" in b:
                    extra = (f"  coarse res {fmt(b.get('coarse_res_init_mm'))}->{fmt(b.get('coarse_res_final_mm'), 2)} mm  edge max {fmt(b['edge_res_max_m'], 1, 1e3)} mm / {fmt(b['edge_rot_max_deg'], 2)} deg"
                             f"  node abs {fmt(b['node_disp_abs_max_m'], 0, 1e3)} mm  to final {fmt(b.get('dtheta_to_final_max_m'), 1, 1e3)} mm  conv {(b.get('solver_grad') or {}).get('converged')}"
                             f"  check {b.get('check_pairs')} px {fmt(b.get('check_gap_mm'))} mm  pose nodes min/med {b.get('pose_nodes_min')}/{b.get('pose_nodes_med')} fb {b.get('pose_fallbacks')}")
                if "accept_checks" in b:
                    extra += f"  pairs {b.get('pairs')}  node max {fmt(b.get('max_node_disp_m'), 0, 1e3)} mm / {fmt(b.get('max_node_rot_deg'), 2)} deg  conv {b.get('converged')}"
                if b.get("reject") or b.get("reason"):
                    extra += f"  [{b.get('reject') or b.get('reason')}]"
                print(f"   {name:13s} ATE {fmt(b.get('ate_kf_m'), 2, 100)} cm  gap {fmt(b.get('gap_mm'))} mm (own {fmt(b.get('gap_own_mm'))}, valid {fmt(b.get('gap_valid_frac'), 2)})  "
                      f"dev vs P0 t {fmt(dv.get('t_med_mm'))}/{fmt(dv.get('t_max_mm'))} mm r {fmt(dv.get('r_med_deg'), 2)}/{fmt(dv.get('r_max_deg'), 2)} deg  "
                      f"time {fmt(b.get('t_total_s'))} s (solve {fmt(b.get('solve_s'))} s, {b.get('lbfgs_iters', b.get('iters'))} it){extra}", flush=True)
            del states, res1, dres, maps, graph, cache, ctx1, rel
            torch.cuda.empty_cache()
    print(f"[pd_experiment] {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
