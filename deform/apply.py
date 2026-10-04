"""Apply the deformation to Gaussians and re-sync camera poses (plan §4.9)."""
import torch

from deform.field import blend_quat, phi, quat_of
from deform.influence import influence, weights_at
from deform.render_utils import DEV, make_cam, render_subset
from deform.rigid import quat_mul


def apply_to_gaussians(x_pre, q_pre, idx, w, g, R, t):
    """mu'' = phi(x_pre); q'' = qbar (x) q_pre (scale untouched)."""
    xyz = phi(x_pre, idx, w, g.double(), R.double(), t.double()).float()
    qn = quat_of(R)
    qb = blend_quat(idx, w, qn).float()
    q = torch.nn.functional.normalize(quat_mul(qb, q_pre.float()), dim=-1)
    return xyz, q, qb


def kabsch(a, b, w):
    """argmin_M sum w ||M a - b||^2 over SE(3). a,b [n,3] float64, w [n]. Returns 4x4."""
    ws = w.sum().clamp_min(1e-12)
    ca = (w[:, None] * a).sum(0) / ws
    cb = (w[:, None] * b).sum(0) / ws
    Hm = ((a - ca) * w[:, None]).T @ (b - cb)
    U, _, Vt = torch.linalg.svd(Hm)
    V = Vt.T
    d = torch.sign(torch.det(V @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, float(d) if d != 0 else 1.0], dtype=a.dtype, device=a.device))
    Rm = V @ D @ U.T
    M = torch.eye(4, dtype=a.dtype, device=a.device)
    M[:3, :3] = Rm
    M[:3, 3] = cb - Rm @ ca
    return M


@torch.no_grad()
def sync_poses(src_poses, G_src, xyz_new, S, kf_uids, all_uids, intr, field, cfg):
    """Kabsch per keyframe on the visible Gaussians; fallback = field at the camera centre.

    src_poses: {uid: c2w} source poses (V-A: P*, V-B: P); G_src: state before deformation.
    field: dict(g, R, t, t_nodes, r_node, delta_t, K, K_cand, beta) for the fallback.
    Returns ({uid: new c2w (float64 cpu)}, {uid: M}, log).
    """
    ps = cfg["pose_sync"]
    alpha = G_src["opacity"].reshape(-1)
    Ms, n_fb = {}, 0
    a_all = G_src["xyz"].double()
    b_all = xyz_new.double()
    for uid in kf_uids:
        cam = make_cam(uid, src_poses[uid], intr)
        C = render_subset(cam, G_src)["contrib_full"]
        V = (C > 0.5) & (alpha > 0.5)
        if int(V.sum()) >= ps["min_gauss"]:
            eta = C[V].double() * S[V].double().clamp_min(0.05)
            Ms[uid] = kabsch(a_all[V], b_all[V], eta)
        else:
            n_fb += 1
            o = torch.as_tensor(src_poses[uid], dtype=torch.float64)[:3, 3].to(DEV)[None]
            idx, w = influence(o.float(), torch.tensor([float(uid)], device=DEV), field["g"], field["t_nodes"],
                               field["r_node"], field["delta_t"], field["K"], field["K_cand"], field["beta"])
            po = phi(o, idx, w, field["g"].double(), field["R"].double(), field["t"].double())[0]
            qb = blend_quat(idx, w, quat_of(field["R"]))
            from deform.field import quat_to_rotmat
            Rb = quat_to_rotmat(qb.double())[0]
            M = torch.eye(4, dtype=torch.float64, device=DEV)
            M[:3, :3] = Rb
            M[:3, 3] = po - Rb @ o[0]
            Ms[uid] = M
    kf_sorted = sorted(kf_uids)
    kf_t = torch.tensor(kf_sorted, dtype=torch.float64)
    new = {}
    for uid in all_uids:
        if uid in Ms:
            j = uid
        else:
            dist = (kf_t - float(uid)).abs()
            j = kf_sorted[int(torch.nonzero(dist == dist.min())[0])]  # tie -> earlier keyframe
        P = torch.as_tensor(src_poses[uid], dtype=torch.float64).to(DEV)
        new[uid] = (Ms[j] @ P).cpu()
    tr = [float(M[:3, 3].norm()) for M in Ms.values()]
    ang = [float(torch.arccos(((torch.trace(M[:3, :3]) - 1) / 2).clamp(-1, 1))) for M in Ms.values()]
    t_med = float(torch.tensor(tr).median()) if tr else None
    a_med = float(torch.tensor(ang).median()) if ang else None
    log = {"median_t_mm": None if t_med is None else 1e3 * t_med,
           "median_rot_deg": None if a_med is None else float(torch.rad2deg(torch.tensor(a_med))),
           "n_fallback": n_fb, "n_kf": len(kf_uids)}
    return new, Ms, log


@torch.no_grad()
def stretch_diag(x_pre, idx, w, dmax, g, R, t, n=10000, h=0.01, seed=0):
    """e_str = max_m |sigma_m(J) - 1| with J by central differences (I fixed, weights re-evaluated)."""
    N = x_pre.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    sel = torch.randperm(N, generator=gen)[: min(n, N)].to(DEV)
    x = x_pre[sel].double()
    i, dm = idx[sel], dmax[sel]
    gd, Rd, td = g.double(), R.double(), t.double()
    cols = []
    for a in range(3):
        e = torch.zeros(3, dtype=torch.float64, device=DEV)
        e[a] = h
        wp = weights_at(x + e, i, gd, dm.double())
        wm = weights_at(x - e, i, gd, dm.double())
        cols.append((phi(x + e, i, wp, gd, Rd, td) - phi(x - e, i, wm, gd, Rd, td)) / (2 * h))
    J = torch.stack(cols, -1)
    sv = torch.linalg.svdvals(J)
    es = (sv - 1).abs().max(-1).values
    return {"p50": float(es.median()), "p90": float(torch.quantile(es, 0.9))}
