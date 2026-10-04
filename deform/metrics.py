"""Per-event metrics M1-M5 (plan §5.1) + layer split, loop keyframes, opt/eval split.

A "method" is (G, poses): G = Gaussian dict on GPU (xyz/rot/scale/opacity/f_dc), poses = {uid: c2w}.
All sets (J_L, J_opt/J_eval, Pi_eval, B / N-bar, M5 visible set) are fixed once from the reference
(rigid) result and reused for every method so comparisons stay paired.
"""
import math

import numpy as np
import torch

from deform.render_utils import DEV, render_subset

ALPHA_THR = 0.95


# ---------------------------------------------------------------- pixel helpers
def stride_grid(H, W, s):
    g = torch.zeros((H, W), dtype=torch.bool, device=DEV)
    g[::s, ::s] = True
    return g


def checker(H, W, tile, parity):
    yy = torch.arange(H, device=DEV)[:, None] // tile
    xx = torch.arange(W, device=DEV)[None, :] // tile
    return ((yy + xx) % 2) == parity


def edge_mask(D, g_max):
    """True where a 4-neighbour differs by more than g_max relative depth."""
    D = D.reshape(D.shape[-2], D.shape[-1])
    Dp = torch.nn.functional.pad(D[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    c = Dp[1:-1, 1:-1]
    rel = torch.zeros_like(c, dtype=torch.bool)
    for nb in (Dp[:-2, 1:-1], Dp[2:, 1:-1], Dp[1:-1, :-2], Dp[1:-1, 2:]):
        rel |= (c - nb).abs() / c.clamp_min(1e-6) > g_max
    return rel


def ray_dirs(cam):
    H, W = int(cam.image_height), int(cam.image_width)
    v, u = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32),
                          torch.arange(W, device=DEV, dtype=torch.float32), indexing="ij")
    return torch.stack(((u - float(cam.cx)) / float(cam.fx), (v - float(cam.cy)) / float(cam.fy),
                        torch.ones_like(u)), 0)  # [3,H,W], z = 1


def unit(n):
    return n / n.norm(dim=0, keepdim=True).clamp_min(1e-8)


# ---------------------------------------------------------------- layers
def layer_masks(active):
    a = active.to(DEV).bool()
    return ~a, a  # old = inactive before reactivation, new = active


def birth_split(cur_uid, loop_uid):
    """plan v5 A1: s_k = (loop_uid + cur_uid) / 2."""
    return 0.5 * (float(cur_uid) + float(loop_uid))


def layer_masks_birth(t0, split):
    t = t0.to(DEV).reshape(-1).float()
    return t < split, t >= split


def layers_for(active, t0, cur_uid, loop_uid, cfg):
    """(m_old, m_new, info) for the configured layer mode (layers.mode active | birth)."""
    mode = (cfg.get("layers") or {}).get("mode", "active")
    if mode == "birth":
        s = birth_split(cur_uid, loop_uid)
        mo, mn = layer_masks_birth(t0, s)
        return mo, mn, {"mode": "birth", "split_t0": s}
    if mode != "active":
        raise ValueError(f"layers.mode must be active or birth, got {mode!r}")
    mo, mn = layer_masks(active)
    return mo, mn, {"mode": "active"}


def layer_split_log(active, t0, cur_uid, loop_uid):
    """Both splits side by side: counts per layer and the overlap of the 'new' layers (plan v5 A1 log)."""
    a_old, a_new = layer_masks(active)
    s = birth_split(cur_uid, loop_uid)
    b_old, b_new = layer_masks_birth(t0, s)
    inter_new = int((a_new & b_new).sum())
    union_new = int((a_new | b_new).sum())
    return {"split_t0": s,
            "active": {"old": int(a_old.sum()), "new": int(a_new.sum())},
            "birth": {"old": int(b_old.sum()), "new": int(b_new.sum())},
            "new_overlap_jaccard": inter_new / max(1, union_new),
            "active_new_born_before_split": int((a_new & b_old).sum()),
            "birth_new_inactive": int((b_new & a_old).sum())}


def render_layers(cam, G, m_old, m_new):
    po = render_subset(cam, G, m_old)
    pn = render_subset(cam, G, m_new)
    return {
        "Do": po["rend_depth_median"][0], "Dn": pn["rend_depth_median"][0],
        "Oo": po["rend_alpha"][0], "On": pn["rend_alpha"][0],
        "No": unit(po["rend_normal"]), "Nn": unit(pn["rend_normal"]),
        "Co": po["contrib_full"], "Cn": pn["contrib_full"],
    }


def layer_gap(cam, L):
    """Signed distance between the two layers along the old normal, per pixel (metres).

    y_new - y_old = R (Dn - Do) ray, n = R N_old  ->  n.(y_new - y_old) = (Dn - Do) * N_old.ray
    """
    r = ray_dirs(cam)
    return (L["Dn"] - L["Do"]) * (L["No"] * r).sum(0)


# ---------------------------------------------------------------- loop keyframes / split
def select_loop_kfs(st, G, poses, m_old, m_new, cfg):
    c = cfg["corr"]
    rows = []
    for uid in st.kf_uids:
        L = render_layers(st.cam(uid, poses=poses), G, m_old, m_new)
        fo = (L["Oo"] > ALPHA_THR).float().mean().item()
        fn = (L["On"] > ALPHA_THR).float().mean().item()
        fb = ((L["Oo"] > ALPHA_THR) & (L["On"] > ALPHA_THR)).float().mean().item()
        rows.append({"uid": int(uid), "f_old": fo, "f_new": fn, "f_both": fb})
    sel = [r for r in rows if r["f_old"] >= c["cov_layer"] and r["f_new"] >= c["cov_layer"]
           and r["f_both"] >= c["cov_both"]]
    sel = sorted(sel, key=lambda r: -r["f_both"])[: c["max_loop_kfs"]]
    return sorted(r["uid"] for r in sel), rows


def split_opt_eval(JL, H, W, tile=32):
    """Even positions -> opt, odd -> eval. One keyframe: 32x32 checkerboard. Returns [(uid, region|None)]."""
    JL = sorted(JL)
    if len(JL) == 1:
        u = JL[0]
        return [(u, checker(H, W, tile, 0))], [(u, checker(H, W, tile, 1))]
    return [(u, None) for u in JL[0::2]], [(u, None) for u in JL[1::2]]


# ---------------------------------------------------------------- M1
def eval_pixel_set(st, G_ref, poses_ref, J_eval, m_old, m_new, cfg):
    """Pi_eval from the reference (rigid) result: loose gating, no S gating."""
    c = cfg["corr"]
    H, W = int(st.intr["H"]), int(st.intr["W"])
    grid = stride_grid(H, W, c["stride"])
    out = {}
    for uid, region in J_eval:
        L = render_layers(st.cam(uid, poses=poses_ref), G_ref, m_old, m_new)
        m = grid & (L["Oo"] > ALPHA_THR) & (L["On"] > ALPHA_THR)
        m &= (L["Dn"] - L["Do"]).abs() < c["eps_eval"]
        m &= (L["No"] * L["Nn"]).sum(0) > c["cos_theta_n"]
        m &= ~(edge_mask(L["Do"], c["g_max"]) | edge_mask(L["Dn"], c["g_max"]))
        if region is not None:
            m &= region
        out[uid] = (m, region)
    return out


def pi_star(st, G_ref, poses_ref, J_eval_uids, m_old, m_new, cfg_base):
    """Plan v2 §2: unified evaluation set = cells B (parity 1) of the 32x32 checkerboard on every J_eval
    keyframe, evaluation gating of eval_pixel_set, built from the reference (rigid) result and the BASE
    (A-1) config so that no switch under test can change it. Also returns |Dn - Do| of the reference."""
    H, W = int(st.intr["H"]), int(st.intr["W"])
    tile = cfg_base["corr"]["checker"]
    J = [(u, checker(H, W, tile, 1)) for u in sorted(J_eval_uids)]
    Pi = eval_pixel_set(st, G_ref, poses_ref, J, m_old, m_new, cfg_base)
    dD = {}
    for uid in Pi:
        L = render_layers(st.cam(uid, poses=poses_ref), G_ref, m_old, m_new)
        dD[uid] = (L["Dn"] - L["Do"]).abs()
    return Pi, dD


def gap_maps(st, G, poses, Pi, m_old, m_new):
    """Per eval keyframe: |gap| and validity (both layers still rendered with alpha > 0.95)."""
    res = {}
    for uid, (m, _) in Pi.items():
        cam = st.cam(uid, poses=poses)
        L = render_layers(cam, G, m_old, m_new)
        res[uid] = {"gap": layer_gap(cam, L).abs(), "valid": (L["Oo"] > ALPHA_THR) & (L["On"] > ALPHA_THR)}
    return res


def edl_paired(Pi, maps_by_method):
    """Median / p90 / inlier(<1 cm) of |gap| on Pi ∩ (valid for every method). Values in metres."""
    vals = {k: [] for k in maps_by_method}
    n_pix = 0
    for uid, (m, _) in Pi.items():
        keep = m.clone()
        for mm in maps_by_method.values():
            keep &= mm[uid]["valid"]
        n_pix += int(keep.sum())
        for k, mm in maps_by_method.items():
            vals[k].append(mm[uid]["gap"][keep])
    out = {}
    for k, v in vals.items():
        x = torch.cat(v) if v else torch.zeros(0, device=DEV)
        if x.numel() == 0:
            out[k] = {"median": None, "p90": None, "inlier_1cm": None, "n": 0}
        else:
            out[k] = {"median": float(x.median()), "p90": float(torch.quantile(x.float(), 0.9)),
                      "inlier_1cm": float((x < 0.01).float().mean()), "n": int(x.numel())}
    return out, n_pix


def dD_hist_values(st, G, poses, JL, m_old, m_new, stride):
    """|Dn - Do| on loop keyframes where both layers are opaque (before any gating)."""
    H, W = int(st.intr["H"]), int(st.intr["W"])
    grid = stride_grid(H, W, stride)
    vals = []
    for uid in JL:
        L = render_layers(st.cam(uid, poses=poses), G, m_old, m_new)
        m = grid & (L["Oo"] > ALPHA_THR) & (L["On"] > ALPHA_THR)
        vals.append((L["Dn"] - L["Do"]).abs()[m])
    return torch.cat(vals).cpu().numpy() if vals else np.zeros(0)


# ---------------------------------------------------------------- M2
def visible_mask(st, G, poses, uids, thr=0.5):
    vis = torch.zeros(G["xyz"].shape[0], dtype=torch.bool, device=DEV)
    for uid in uids:
        p = render_subset(st.cam(uid, poses=poses), G)
        vis |= p["contrib_full"] > thr
    return vis


def step_sets(st, G_pre, poses_pre, JL, seed, radius=0.05, k=16):
    """Boundary set B and control set N-bar, fixed on the pre state (indices into all Gaussians)."""
    from scipy.spatial import cKDTree

    alpha = G_pre["opacity"].reshape(-1) > 0.5
    vis = visible_mask(st, G_pre, poses_pre, JL) & alpha
    pool = torch.nonzero(alpha).reshape(-1).cpu().numpy()
    cand = torch.nonzero(vis).reshape(-1).cpu().numpy()
    if len(cand) == 0:
        return None
    xyz = G_pre["xyz"].detach().cpu().numpy().astype(np.float64)
    tree = cKDTree(xyz[pool])
    d, j = tree.query(xyz[cand], k=k + 1, distance_upper_bound=radius, workers=4)
    nbr = np.where(np.isfinite(d), pool[np.minimum(j, len(pool) - 1)], -1)
    nbr[nbr == cand[:, None]] = -1  # drop self
    # keep the k closest non-self neighbours
    order = np.argsort(nbr < 0, axis=1, kind="stable")
    nbr = np.take_along_axis(nbr, order, axis=1)[:, :k]
    tc = st.gpre["tc"].numpy()
    valid = nbr >= 0
    n_valid = valid.sum(1)
    diff = valid & (tc[np.maximum(nbr, 0)] != tc[cand][:, None])
    n_diff = diff.sum(1)
    is_B = (n_diff >= 3) & (n_valid >= 3)
    is_N = (n_diff == 0) & (n_valid >= 3)
    B = np.nonzero(is_B)[0]
    Np = np.nonzero(is_N)[0]
    rng = np.random.default_rng(seed)
    Nsel = rng.choice(Np, size=min(len(B), len(Np)), replace=False) if len(Np) else Np
    return {"cand": cand, "nbr": nbr, "B": B, "N": Nsel, "n_vis": int(len(cand))}


def plane_dev(xyz, sets, rows):
    """|n_i.(mu_i - mean_nbrs)| with PCA plane over neighbours (excluding i). xyz: [N,3] GPU."""
    if len(rows) == 0:
        return torch.zeros(0, device=DEV)
    idx = torch.as_tensor(sets["cand"][rows], device=DEV, dtype=torch.long)
    nb = torch.as_tensor(sets["nbr"][rows], device=DEV, dtype=torch.long)
    m = (nb >= 0).float()[..., None]
    P = xyz[nb.clamp_min(0)].double()
    cnt = m.sum(1).clamp_min(1)
    mu = (P * m).sum(1) / cnt
    Q = (P - mu[:, None]) * m
    C = Q.transpose(1, 2) @ Q / cnt[..., None]
    _, V = torch.linalg.eigh(C)
    n = V[:, :, 0]
    return ((xyz[idx].double() - mu) * n).sum(-1).abs().float()


def e_step(xyz, sets):
    eB = plane_dev(xyz, sets, sets["B"])
    eN = plane_dev(xyz, sets, sets["N"])
    if eB.numel() == 0 or eN.numel() == 0:
        return {"e_step": None, "med_B": None, "med_N": None, "p90_B": None, "n_B": int(eB.numel())}
    return {"e_step": float(eB.median() - eN.median()), "med_B": float(eB.median()),
            "med_N": float(eN.median()), "p90_B": float(torch.quantile(eB, 0.9)), "n_B": int(eB.numel())}


# ---------------------------------------------------------------- M3, M4, M5
def render_quality(st, G, poses, uids):
    from gaussian_splatting.utils.image_utils import psnr

    tr = st.tr
    ps, dl = [], []
    for uid in uids:
        cam = st.cam(uid, poses=poses)
        p = render_subset(cam, G)
        img, depth = st.frame(uid)
        im = p["render"].clamp(0, 1)
        m = (img > 0) & (im > 0)
        if m.any():
            ps.append(float(psnr(im[m].unsqueeze(0), img[m].unsqueeze(0))))
        dkey = "rend_depth_expected" if tr.get("depth_type") == "expected" else "rend_depth_median"
        D = p[dkey][0]
        dm = (depth > tr["depth_min_threshold"]) & (depth < tr["depth_max_threshold"]) & (p["rend_alpha"][0] > ALPHA_THR)
        if dm.any():
            dl.append(float((D - depth).abs()[dm].mean()))
    return {"psnr": float(np.mean(ps)) if ps else None, "depth_l1": float(np.mean(dl)) if dl else None}


def ate_kf(st, poses, uids=None):
    from utils.eval_utils import evaluate_evo

    uids = sorted(uids if uids is not None else st.kf_uids)
    uids = [u for u in uids if st.poses_gt.get(u) is not None]
    if len(uids) < 3:
        return None
    est = [np.asarray(poses[u], dtype=np.float64) for u in uids]
    gt = [np.asarray(st.poses_gt[u], dtype=np.float64) for u in uids]
    return float(evaluate_evo(gt, est, "event", quiet=True))


_MESH = {}


def mesh_scene(path):
    import open3d as o3d

    if path not in _MESH:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) == 0:
            import trimesh
            tm = trimesh.load(path, process=False)
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
                                             o3d.utility.Vector3iVector(np.asarray(tm.faces)))
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        _MESH[path] = sc
    return _MESH[path]


def mesh_acc(xyz, idx, mesh_path):
    import open3d as o3d

    if mesh_path is None or len(idx) == 0:
        return None
    sc = mesh_scene(mesh_path)
    pts = xyz[torch.as_tensor(idx, device=xyz.device)].detach().cpu().numpy().astype(np.float32)
    d = sc.compute_distance(o3d.core.Tensor(pts)).numpy()
    return {"acc_median": float(np.median(d)), "frac_gt_1cm": float((d > 0.01).mean()), "n": int(len(d))}


# ---------------------------------------------------------------- all metrics for one event
def evaluate_event(st, methods, ref, cfg, mesh_path=None, seed=0, extras=False):
    """methods: {name: (G, poses)}; ref: name of the reference (rigid) method that fixes the sets."""
    m_old, m_new = layer_masks(st.active)
    G_ref, P_ref = methods[ref]
    H, W = int(st.intr["H"]), int(st.intr["W"])
    JL, cov_rows = select_loop_kfs(st, G_ref, P_ref, m_old, m_new, cfg)
    out = {"n_loop_kfs": len(JL), "J_L": JL, "coverage_rows": cov_rows, "methods": {}}
    if not JL:
        out["status"] = "no_overlap"
    J_opt, J_eval = split_opt_eval(JL, H, W, cfg["corr"]["checker"]) if JL else ([], [])
    out["J_opt"] = [u for u, _ in J_opt]
    out["J_eval"] = [u for u, _ in J_eval]
    eval_uids = sorted({u for u, _ in J_eval})

    # M1
    if J_eval:
        Pi = eval_pixel_set(st, G_ref, P_ref, J_eval, m_old, m_new, cfg)
        maps = {k: gap_maps(st, G, P, Pi, m_old, m_new) for k, (G, P) in methods.items()}
        edl, n_pix = edl_paired(Pi, maps)
        out["n_eval_pix"] = n_pix
        if extras:
            out["_maps"] = maps
            out["_Pi"] = Pi
    else:
        edl = {k: None for k in methods}
        out["n_eval_pix"] = 0
    # M2 (sets from the pre state if present, else reference)
    pre_name = "pre" if "pre" in methods else ref
    sets = step_sets(st, methods[pre_name][0], methods[pre_name][1], JL or st.kf_uids[-1:], seed) if JL else None
    out["n_B"] = None if sets is None else int(len(sets["B"]))
    # M5 visible set fixed from the reference at J_eval
    acc_idx = None
    if mesh_path is not None and eval_uids:
        vis = visible_mask(st, G_ref, P_ref, eval_uids) & (G_ref["opacity"].reshape(-1) > 0.5)
        acc_idx = torch.nonzero(vis).reshape(-1).cpu().numpy()
    for k, (G, P) in methods.items():
        r = {"e_dl": edl.get(k) if isinstance(edl, dict) else None}
        r["e_step"] = e_step(G["xyz"], sets) if sets is not None else None
        r["render"] = render_quality(st, G, P, eval_uids) if eval_uids else None
        r["ate_kf"] = ate_kf(st, P)
        r["acc"] = mesh_acc(G["xyz"], acc_idx, mesh_path) if acc_idx is not None else None
        out["methods"][k] = r
    if extras:
        out["_JL"] = JL
        out["_sets"] = sets
    return out
