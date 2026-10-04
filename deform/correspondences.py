"""Two-layer correspondences on the loop keyframes (plan §4.7, steps 2 and 5-8; plan v2 switches).

Keyframe selection (step 3) and the opt/eval split (step 4) come from deform.metrics so that the
pipeline and the metrics share exactly the same J_opt / J_eval. Which (keyframe, region) pairs may
produce correspondences is decided by the caller (pipeline.pair_sources); with the default config this
is J_opt only, exactly as A-1.

Plan v2 additions (all off by default):
  corr.s_gate.mode  hard (A-1) | weight (F1w: keep every pair, omega = max(min(S_old, S_new), floor)) | off
  corr.s_gate.tau / layers   F1t (tau 0.1) / F1o (old layer only)
  registration pairs (D3): build_pairs_registration()
Every pair also carries src_uid / src_pix (keyframe and flat pixel index) for the split checks (T3).
"""
import numpy as np
import torch

from deform.metrics import ALPHA_THR, edge_mask, ray_dirs, render_layers, stride_grid
from deform.render_utils import DEV, render_attribute


def _s_gate(S_o, S_n, cfg):
    """-> (s_ok [H,W] bool, omega [H,W] float or None)."""
    sg = cfg["corr"].get("s_gate", {}) or {}
    mode = sg.get("mode", "hard")
    if mode == "off":
        return torch.ones_like(S_o, dtype=torch.bool), None
    if mode == "weight":
        # rendered S is an alpha-blended average, so S == 1 comes out as 1 +- 1e-7; omega is rounded to 1e-4
        # (far finer than any meaningful S difference) so that equal scores give exactly equal weights (T2)
        omega = torch.round(torch.minimum(S_o, S_n).clamp(0.0, 1.0) * 1e4) / 1e4
        omega = omega.clamp_min(float(sg.get("floor", 0.05)))
        return torch.ones_like(S_o, dtype=torch.bool), omega
    tau = sg.get("tau")
    tau = cfg["reliability"]["tau_S"] if tau is None else tau
    if sg.get("layers", "both") == "old":
        return S_o >= tau, None
    return torch.minimum(S_o, S_n) >= tau, None


def _anchor_local(y, cand, pos0_np, pos0, x_pre, Rbar0, t0):
    """Nearest contributing Gaussian of the layer -> (tau, local inverse of phi_0 at the anchor)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(pos0_np[cand.cpu().numpy()])
    _, j = tree.query(y.cpu().numpy().astype(np.float64), k=1, workers=4)
    a = cand[torch.as_tensor(j, device=DEV, dtype=torch.long)]
    # local inverse of phi_0 around the anchor: x = mu_pre_a + Rbar0_a^T (y - phi0(mu_pre_a))
    xl = x_pre[a].double() + (Rbar0[a].double().transpose(1, 2) @ (y.double() - pos0[a].double())[..., None]).squeeze(-1)
    return t0[a].float(), xl


@torch.no_grad()
def build_pairs(cams, G0, m_old, m_new, S, J_opt, t0, x_pre, pos0, Rbar0, cfg, seed):
    """cams: {uid: Camera at P*_j}; G0: phi_0 state (positions pos0, rotations Rbar0 R_pre).
    J_opt: [(uid, region|None)] keyframes (and pixel regions) allowed to produce pairs.

    Returns dict: x_old, x_new [P,3] (pre coords), tau_old, tau_new [P], n [P,3] (world), src_uid,
    src_pix [P], omega [P] (weight mode only) and logs.
    """
    c = cfg["corr"]
    H, W = int(next(iter(cams.values())).image_height), int(next(iter(cams.values())).image_width)
    grid = stride_grid(H, W, c["stride"])
    xs = {"old": [], "new": []}
    taus = {"old": [], "new": []}
    ns, src_u, src_p, oms = [], [], [], []
    gate = {"alpha": 0, "dist": 0, "normal": 0, "S": 0, "edge": 0}
    funnel = {"raw": 0, "alpha": 0, "dist": 0, "normal": 0, "S": 0, "edge": 0}
    n_both = 0
    n_raw = 0
    dD_vals = []
    pos0_np = pos0.detach().cpu().numpy().astype(np.float64)
    for uid, region in J_opt:
        cam = cams[uid]
        L = render_layers(cam, G0, m_old, m_new)
        S_o, _ = render_attribute(cam, G0, S, m_old)
        S_n, _ = render_attribute(cam, G0, S, m_new)
        base = grid.clone()
        if region is not None:
            base &= region
        n_raw += int(base.sum())
        a_ok = (L["Oo"] > ALPHA_THR) & (L["On"] > ALPHA_THR)
        d_ok = (L["Dn"] - L["Do"]).abs() < c["eps_d"]
        n_ok = (L["No"] * L["Nn"]).sum(0) > c["cos_theta_n"]
        s_ok, omega = _s_gate(S_o[0], S_n[0], cfg)
        e_ok = ~(edge_mask(L["Do"], c["g_max"]) | edge_mask(L["Dn"], c["g_max"]))
        both = base & a_ok
        n_both += int(both.sum())
        gate["alpha"] += int((base & ~a_ok).sum())
        gate["dist"] += int((both & ~d_ok).sum())
        gate["normal"] += int((both & ~n_ok).sum())
        gate["S"] += int((both & ~s_ok).sum())
        gate["edge"] += int((both & ~e_ok).sum())
        # sequential funnel (plan v2 §3.2): raw -> alpha -> eps_d -> normal -> S -> edge
        f = base
        funnel["raw"] += int(f.sum())
        for k, ok in (("alpha", a_ok), ("dist", d_ok), ("normal", n_ok), ("S", s_ok), ("edge", e_ok)):
            f = f & ok
            funnel[k] += int(f.sum())
        dD_vals.append((L["Dn"] - L["Do"]).abs()[both])
        m = both & d_ok & n_ok & s_ok & e_ok
        if not bool(m.any()):
            continue
        r = ray_dirs(cam)
        c2w = torch.linalg.inv(cam.T.double()).float()
        Rw, tw = c2w[:3, :3], c2w[:3, 3]
        for lay, D, Cfull, msk in (("old", L["Do"], L["Co"], m_old), ("new", L["Dn"], L["Cn"], m_new)):
            pc = (r * D[None])[:, m].T  # camera-frame points [P,3]
            y = pc @ Rw.T + tw
            cand = torch.nonzero((Cfull > 0.5) & msk).reshape(-1)
            if cand.numel() == 0:
                xs[lay].append(None)
                continue
            tau, xl = _anchor_local(y, cand, pos0_np, pos0, x_pre, Rbar0, t0)
            taus[lay].append(tau)
            xs[lay].append(xl)
        if xs["old"][-1] is None or xs["new"][-1] is None:
            xs["old"].pop(); xs["new"].pop()
            continue
        No = L["No"][:, m].T
        ns.append(No.double() @ Rw.T.double())
        pix = torch.nonzero(m.reshape(-1)).reshape(-1)
        src_p.append(pix)
        src_u.append(torch.full_like(pix, int(uid)))
        if omega is not None:
            oms.append(omega[m].float())
    out = {"gate_counts": gate, "funnel": funnel, "n_raw": n_raw, "n_both_opaque": n_both,
           "dD_vals": torch.cat(dD_vals).cpu().numpy() if dD_vals else np.zeros(0)}
    if not ns:
        out.update({"P": 0})
        return out
    out.update({
        "x_old": torch.cat(xs["old"]), "x_new": torch.cat(xs["new"]),
        "tau_old": torch.cat(taus["old"]), "tau_new": torch.cat(taus["new"]),
        "n": torch.nn.functional.normalize(torch.cat(ns), dim=-1),
        "src_uid": torch.cat(src_u), "src_pix": torch.cat(src_p),
    })
    if oms:
        out["omega"] = torch.cat(oms)
    out["P"] = int(out["n"].shape[0])
    return out


@torch.no_grad()
def build_pairs_registration(cams, G0, m_old, m_new, sources, t0, x_pre, pos0, Rbar0, cfg, seed):
    """D3 (plan v2 §3.3): per keyframe, ICP of the new layer onto the old layer; keyframes passing
    Q1-Q4 give pairs on their final inliers (stride corr.stride): x_new = y_new (tau of the nearest
    new-layer Gaussian), x_old = H_j y_new (tau of the nearest old-layer Gaussian of the associated
    old point), n = old normal at the associated pixel. No S gate."""
    from scipy.spatial import cKDTree

    from deform.registration import register

    c = cfg["corr"]
    H, W = int(next(iter(cams.values())).image_height), int(next(iter(cams.values())).image_width)
    grid = stride_grid(H, W, c["stride"]).reshape(-1)
    xs = {"old": [], "new": []}
    taus = {"old": [], "new": []}
    ns, src_u, src_p, kf_rows = [], [], [], []
    funnel = {"kfs": 0, "kfs_passed": 0, "overlap": 0, "inliers": 0, "sampled": 0}
    pos0_np = pos0.detach().cpu().numpy().astype(np.float64)
    sigma_c = float(cfg["energy"]["sigma_c"])
    for uid, region in sources:
        cam = cams[uid]
        L = render_layers(cam, G0, m_old, m_new)
        reg = register(L, cam, cfg["reg"], sigma_c, allowed=region)
        st = reg["stats"]
        kf_rows.append({"uid": int(uid), "region": None if region is None else "A", "passed": reg["passed"],
                        "fail": reg["fail"], **{k: (bool(v)) for k, v in reg["Q"].items()}, **st})
        funnel["kfs"] += 1
        funnel["overlap"] += st["n_overlap"]
        if not reg["passed"]:
            continue
        funnel["kfs_passed"] += 1
        funnel["inliers"] += st["n_inlier"]
        keep = grid[reg["src_pix"]]
        sp, tp = reg["src_pix"][keep], reg["tgt_pix"][keep]
        if sp.numel() == 0:
            continue
        funnel["sampled"] += int(sp.numel())
        r = ray_dirs(cam).double().reshape(3, -1)
        c2w = torch.linalg.inv(cam.T.double())
        Rw, tw = c2w[:3, :3], c2w[:3, 3]
        Hc = reg["H_c"]
        p_new = (r[:, sp] * L["Dn"].double().reshape(-1)[sp][None]).T  # camera frame
        p_tgt = p_new @ Hc[:3, :3].T + Hc[:3, 3]
        q_old = (r[:, tp] * L["Do"].double().reshape(-1)[tp][None]).T
        y_new = (p_new @ Rw.T + tw).float()
        y_tgt = (p_tgt @ Rw.T + tw).float()
        y_old = (q_old @ Rw.T + tw).float()
        cand_n = torch.nonzero((L["Cn"] > 0.5) & m_new).reshape(-1)
        cand_o = torch.nonzero((L["Co"] > 0.5) & m_old).reshape(-1)
        if cand_n.numel() == 0 or cand_o.numel() == 0:
            continue
        tau_n, xl_n = _anchor_local(y_new, cand_n, pos0_np, pos0, x_pre, Rbar0, t0)
        tree = cKDTree(pos0_np[cand_o.cpu().numpy()])
        _, j = tree.query(y_old.cpu().numpy().astype(np.float64), k=1, workers=4)
        a = cand_o[torch.as_tensor(j, device=DEV, dtype=torch.long)]
        xl_o = x_pre[a].double() + (Rbar0[a].double().transpose(1, 2) @ (y_tgt.double() - pos0[a].double())[..., None]).squeeze(-1)
        xs["new"].append(xl_n); taus["new"].append(tau_n)
        xs["old"].append(xl_o); taus["old"].append(t0[a].float())
        nq = L["No"].double().reshape(3, -1)[:, tp].T
        ns.append(nq @ Rw.T)
        src_p.append(sp)
        src_u.append(torch.full_like(sp, int(uid)))
    out = {"funnel": funnel, "reg_kfs": kf_rows, "n_raw": funnel["overlap"], "n_both_opaque": funnel["overlap"],
           "gate_counts": {"alpha": 0, "dist": 0, "normal": 0, "S": 0, "edge": 0}, "dD_vals": np.zeros(0)}
    if not ns:
        out.update({"P": 0})
        return out
    out.update({
        "x_old": torch.cat(xs["old"]), "x_new": torch.cat(xs["new"]),
        "tau_old": torch.cat(taus["old"]), "tau_new": torch.cat(taus["new"]),
        "n": torch.nn.functional.normalize(torch.cat(ns), dim=-1),
        "src_uid": torch.cat(src_u), "src_pix": torch.cat(src_p),
    })
    out["P"] = int(out["n"].shape[0])
    return out


# plan v4 A1: fixed salt of the pair hash (the splitmix64 increment; any constant would do, it is part of the
# frozen config so that a different salt is a different, reproducible draw)
HASH_SALT = 0x9E3779B97F4A7C15
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x, salt=HASH_SALT):
    """Finaliser of splitmix64 on uint64 numpy arrays (wrap-around arithmetic): h = mix(x + salt).
    With salt = the splitmix64 increment, h(0), h(salt), h(2 salt) are the first outputs of the seed-0 stream."""
    with np.errstate(over="ignore"):
        z = (np.asarray(x, dtype=np.uint64) + np.uint64(salt & 0xFFFFFFFFFFFFFFFF)) & _M64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
        return z ^ (z >> np.uint64(31))


def pair_keys(src_uid, src_pix, img_w):
    """Integer key of a pair from its source pixel: (uid << 40) | (v << 20) | u, int64 [P] (CPU)."""
    uid = src_uid.detach().cpu().long()
    pix = src_pix.detach().cpu().long()
    v, u = pix // int(img_w), pix % int(img_w)
    return (uid << 40) | (v << 20) | u


def pair_hash_order(keys, salt=HASH_SALT):
    """int64 [P] whose signed order equals the unsigned order of splitmix64(key): computed on the CPU in
    numpy uint64 (well-defined wrap-around), so the result is the same whatever device the pairs live on."""
    h = splitmix64(keys.numpy().astype(np.uint64), salt)
    return torch.from_numpy((h ^ np.uint64(1 << 63)).view(np.int64).copy())


def _lex_order(*cols):
    """Row order of a lexicographic sort by cols[0], then cols[1], ... (stable argsorts, last key first)."""
    o = torch.arange(cols[0].shape[0], device=cols[0].device)
    for c in reversed(cols):
        o = o[torch.argsort(c[o], stable=True)]
    return o


def cap_pairs(pairs, node_of_pair, max_per_node, max_total, seed, omega=None, select="random", img_w=None,
              salt=HASH_SALT):
    """Keep at most max_per_node pairs per node, then at most max_total overall.

    select="random" (frozen A-1 .. v4): one seeded global permutation decides the order inside every node.
    select="hash" (plan v4 A1): the order inside a node is by splitmix64(key(uid, u, v)) with a fixed salt,
    ties by key then by index; max_total keeps the smallest hashes overall. Adding or removing one pair
    then changes the kept set of at most one node, and the choice is the same in every process.
    With omega (F1w), pairs with larger omega are kept first inside each node (stable), then the above."""
    P = node_of_pair.shape[0]
    if select == "hash":
        keys = pair_keys(pairs["src_uid"], pairs["src_pix"], img_w)
        h = pair_hash_order(keys, salt).to(node_of_pair.device)
        keys = keys.to(node_of_pair.device)
        cols = [node_of_pair] + ([-omega] if omega is not None else []) + [h, keys]
        o = _lex_order(*cols)
        nd_s = node_of_pair[o]
        first = torch.ones_like(nd_s, dtype=torch.bool)
        first[1:] = nd_s[1:] != nd_s[:-1]
        start_idx = torch.cummax(torch.where(first, torch.arange(P, device=nd_s.device), torch.zeros_like(nd_s)), 0).values
        rank = torch.arange(P, device=nd_s.device) - start_idx
        keep = o[rank < max_per_node]
        if keep.numel() > max_total:
            keep = keep[_lex_order(h[keep], keys[keep])[:max_total]]
        keep = torch.sort(keep).values
        return {k: (v[keep] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == P else v) for k, v in pairs.items()}
    if select != "random":
        raise ValueError(f"corr.select must be random or hash, got {select!r}")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(P, generator=gen).to(node_of_pair.device)
    if omega is not None:
        perm = perm[torch.argsort(-omega[perm], stable=True)]
    nd = node_of_pair[perm]
    o = torch.argsort(nd, stable=True)
    nd_s = nd[o]
    first = torch.ones_like(nd_s, dtype=torch.bool)
    first[1:] = nd_s[1:] != nd_s[:-1]
    start_idx = torch.cummax(torch.where(first, torch.arange(P, device=nd.device), torch.zeros_like(nd_s)), 0).values
    rank = torch.arange(P, device=nd.device) - start_idx
    keep = perm[o[rank < max_per_node]]
    if keep.numel() > max_total:
        sub = torch.randperm(keep.numel(), generator=gen)[:max_total].to(keep.device)
        keep = keep[sub]
    keep = torch.sort(keep).values
    return {k: (v[keep] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == P else v) for k, v in pairs.items()}
