"""P3 gate G3 (a)-(d): deformation pipeline on synthetic two-scan scenes, through the real rasterizer.

(a) identity: aligned layers -> max Gaussian displacement < 0.1 mm
(b) pull: new layer bent by a smooth known field (1 deg + 2 cm linear) -> e_dl drops >= 80 %
(c) seam: plane with two keyframe increments (1 cm step) -> V-B lowers e_step, V-A does not
(d) V-B init: phi_0 equals the rigid shift for Gaussians deep inside one keyframe's region
Run: python tests/test_solver_synthetic.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.field import phi  # noqa: E402
from deform.influence import influence  # noqa: E402
from deform.nodes import build_edges, build_nodes  # noqa: E402
from deform.pipeline import correct_map, delta_T_dict, node_init  # noqa: E402
from deform.render_utils import DEV, make_cam, render_subset  # noqa: E402
from deform.rigid import apply_rigid  # noqa: E402
from deform.solver import Problem, solve  # noqa: E402
from gaussian_splatting.utils.general_utils import rotmat2quaternion  # noqa: E402

INTR = {"fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 240.0, "W": 640, "H": 480}
TR = {"depth_min_threshold": 0.1, "depth_max_threshold": 6.0, "old_than_N_keyframe": 12,
      "prune_size_threshold": 0.25, "depth_type": "median"}


def cfg_tum(**over):
    c = dcfg.resolve({}, {"Dataset": {"type": "tum"}, "Training": TR})
    c["diagnostics"]["stretch_samples"] = 2000
    for k, v in over.items():
        a, b = k.split(".")
        c[a][b] = v
    return c


def plane(origin, u, v, nrm, nu, nv, spacing, scale=None):
    o, u, v, n = (torch.tensor(a, dtype=torch.float32, device=DEV) for a in (origin, u, v, nrm))
    iu = torch.arange(nu, device=DEV, dtype=torch.float32) * spacing
    iv = torch.arange(nv, device=DEV, dtype=torch.float32) * spacing
    A, B = torch.meshgrid(iu, iv, indexing="ij")
    xyz = o + A.reshape(-1, 1) * u + B.reshape(-1, 1) * v
    R = torch.stack([u, v, n], 1)[None].repeat(xyz.shape[0], 1, 1)
    q = rotmat2quaternion(R, normalize=True)
    s = scale if scale is not None else 0.6 * spacing
    N = xyz.shape[0]
    return {"xyz": xyz, "rot": q, "scale": torch.full((N, 2), s, device=DEV),
            "opacity": torch.full((N, 1), 0.98, device=DEV), "f_dc": torch.rand((N, 1, 3), device=DEV)}


def corner(spacing=0.02):
    n = int(2.0 / spacing)
    parts = [plane((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), n, n, spacing),
             plane((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), n, n, spacing),
             plane((0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), n, n, spacing)]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return torch.tensor(T, dtype=torch.float64)


def cams_arc(uids, radius=3.0, height=1.6, a0=15, a1=75, jitter=0.0):
    out = {}
    for i, u in enumerate(uids):
        a = math.radians(a0 + (a1 - a0) * i / max(1, len(uids) - 1)) + jitter
        out[u] = look_at((radius * math.cos(a), radius * math.sin(a), height), (0.5, 0.5, 0.5))
    return out


def nearest_cam(xyz, poses):
    uids = sorted(poses)
    C = torch.stack([poses[u][:3, 3].float() for u in uids]).to(DEV)
    return torch.tensor(uids, device=DEV)[torch.cdist(xyz, C).argmin(1)]


def cat(a, b):
    return {k: torch.cat([a[k], b[k]]) for k in a}


def bend(G, deg=1.0, shift=0.02):
    """Smooth known field: rotation about z through (1,1,0) + translation growing linearly with x."""
    a = math.radians(deg)
    Rz = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], device=DEV)
    c = torch.tensor([1.0, 1.0, 0.0], device=DEV)
    d = torch.tensor([0.6, 0.0, 0.8], device=DEV)
    x = (G["xyz"] - c) @ Rz.T + c + shift * (G["xyz"][:, :1] / 2.0) * d
    qR = rotmat2quaternion(Rz[None], normalize=True)[0]
    from deform.rigid import quat_mul
    q = torch.nn.functional.normalize(quat_mul(qR.expand_as(G["rot"]), G["rot"]), dim=-1)
    out = dict(G)
    out["xyz"], out["rot"] = x, q
    return out


def scene(new_layer_fn=None):
    old_u = list(range(0, 101, 10))
    new_u = list(range(400, 501, 10))
    P_old = cams_arc(old_u)
    P_new = cams_arc(new_u, jitter=0.03)
    Go = corner()
    Gn = corner()
    if new_layer_fn is not None:
        Gn = new_layer_fn(Gn)
    G = cat(Go, Gn)
    No = Go["xyz"].shape[0]
    t_old = nearest_cam(Go["xyz"], P_old)
    t_new = nearest_cam(Gn["xyz"], P_new)
    t0 = torch.cat([t_old, t_new]).long()
    active = torch.cat([torch.zeros(No), torch.ones(Gn["xyz"].shape[0])]).bool().to(DEV)
    poses = {**P_old, **P_new}

    def frame_fn(uid):
        m = ~active if uid < 300 else active  # each scan's frames were mapped into its own layer
        p = render_subset(make_cam(uid, poses[uid], INTR), G, m)
        D = p["rend_depth_median"][0]
        D = torch.where(p["rend_alpha"][0] > 0.5, D, torch.zeros_like(D))
        return p["render"].clamp(0, 1), D

    uids = sorted(poses)
    inp = {"g": G, "t0": t0, "tc": t0.clone(), "active": active, "kf_uids": uids, "all_cam_ids": uids,
           "poses_pre": poses, "poses_pgo": poses, "intr": INTR, "frame_fn": frame_fn, "tr": TR, "W": 12, "seed": 0}
    return inp


class St:
    def __init__(self, inp):
        self.intr, self.kf_uids = INTR, sorted(inp["kf_uids"])

    def cam(self, uid, which="pgo", poses=None):
        return make_cam(uid, poses[uid], INTR)


def edl(inp, res, cfg):
    st = St(inp)
    m_old, m_new = M.layer_masks(inp["active"])
    G_rig, P_rig = res["G_rig"], res["poses_rig"]
    G_def = dict(inp["g"])
    G_def["xyz"], G_def["rot"] = res["xyz"], res["rot"]
    JL, _ = M.select_loop_kfs(st, G_rig, P_rig, m_old, m_new, cfg)
    _, J_eval = M.split_opt_eval(JL, INTR["H"], INTR["W"])
    Pi = M.eval_pixel_set(st, G_rig, P_rig, J_eval, m_old, m_new, cfg)
    maps = {"rigid": M.gap_maps(st, G_rig, P_rig, Pi, m_old, m_new),
            "deform": M.gap_maps(st, G_def, res["poses"], Pi, m_old, m_new)}
    out, npix = M.edl_paired(Pi, maps)
    return out["rigid"]["median"], out["deform"]["median"], npix


def test_a_identity():
    torch.manual_seed(0)
    inp = scene()
    cfg = cfg_tum()
    for var in ("A", "B"):
        res = correct_map(inp, cfg, var)
        disp = (res["xyz"] - inp["g"]["xyz"]).norm(dim=1).max().item()
        print(f"(a) V-{var}: accepted={res['accepted']} reason={res['reason']} max disp = {disp * 1e3:.4f} mm")
        assert disp < 1e-4


def test_b_pull_layers():
    torch.manual_seed(0)
    inp = scene(bend)
    cfg = cfg_tum()
    res = correct_map(inp, cfg, "A")
    lg = res["log"]
    print(f"(b) V-A: accepted={res['accepted']} reason={res['reason']} pairs={lg.get('corr_capped')} "
          f"gate={ {k: round(v, 3) for k, v in lg.get('gate_frac', {}).items()} } iters={lg.get('lbfgs_iters')} "
          f"opt edl {lg.get('edl_opt_init_mm')} -> {lg.get('edl_opt_final_mm')} mm")
    assert res["accepted"], res["reason"]
    e_r, e_d, npix = edl(inp, res, cfg)
    print(f"(b) e_dl on J_eval: rigid {e_r * 1e3:.2f} mm -> deform {e_d * 1e3:.2f} mm ({npix} px), "
          f"reduction {100 * (1 - e_d / e_r):.1f} %")
    assert e_d <= 0.2 * e_r
    res_b = correct_map(inp, cfg, "B")
    e_r2, e_d2, _ = edl(inp, res_b, cfg)
    print(f"(b) V-B: accepted={res_b['accepted']} e_dl {e_r2 * 1e3:.2f} -> {e_d2 * 1e3:.2f} mm")


def seam_case(variant, w_p=10.0):
    """Plane z=0 split in two keyframe regions; keyframe 20 moved +1 cm along z by PGO."""
    G = plane((-1.0, -0.5, 0.0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 200, 100, 0.01)
    tc = torch.where(G["xyz"][:, 0] < 0, 10, 20).long()
    P = {10: look_at((-0.5, 0.0, 2.0), (-0.5, 0.01, 0.0)), 20: look_at((0.5, 0.0, 2.0), (0.5, 0.01, 0.0))}
    P_pgo = {10: P[10].clone(), 20: P[20].clone()}
    P_pgo[20][2, 3] += 0.01
    dT = delta_T_dict([10, 20], [10, 20], P, P_pgo)
    xyz_r, rot_r = apply_rigid(G["xyz"], G["rot"], tc, dT)
    cfg = cfg_tum(**{"energy.w_p": w_p, "energy.w_con": 0.0})
    x_pre = xyz_r if variant == "A" else G["xyz"]
    S = torch.ones(G["xyz"].shape[0], device=DEV)
    R = torch.ones_like(S).bool()
    nd = build_nodes(x_pre, tc, tc, S, R, G["opacity"].reshape(-1), 0.3, 120.0)
    E = build_edges(nd, 0.3, 120.0)
    R0, t_init = node_init(nd, dT, variant)
    idx, w = influence(x_pre, tc.float(), nd["g"], nd["t"], 0.3, 120.0)
    prob = Problem(nd["g"], R0, t_init, E, None, None, None, cfg)
    theta0 = torch.cat([torch.zeros_like(t_init), t_init], 1)
    with torch.enable_grad():
        theta, _ = solve(prob, theta0, cfg)
    Rn, tn, _ = prob.unpack(theta)
    xyz = phi(x_pre, idx, w, nd["g"].double(), Rn, tn).float()
    # e_step on fixed sets from the pre state (camera above the plane)
    cam_pose = {0: look_at((0.0, 0.0, 2.5), (0.0, 0.01, 0.0))}

    class S1:
        intr = INTR
        kf_uids = [0]
        gpre = {"tc": tc.cpu().int()}

        def cam(self, uid, which="pgo", poses=None):
            return make_cam(uid, poses[uid], INTR)

    sets = M.step_sets(S1(), G, cam_pose, [0], seed=0)
    return M.e_step(xyz_r, sets)["e_step"], M.e_step(xyz, sets)["e_step"], (xyz - xyz_r).norm(dim=1).max().item()


def test_c_seam():
    r_a, d_a, mv_a = seam_case("A")
    r_b, d_b, mv_b = seam_case("B")
    print(f"(c) e_step rigid {r_a * 1e3:.3f} mm | V-A {d_a * 1e3:.3f} mm (max move {mv_a * 1e3:.3f} mm) | "
          f"V-B {d_b * 1e3:.3f} mm (max move vs rigid {mv_b * 1e3:.2f} mm)")
    assert abs(d_a - r_a) < 0.1 * r_a  # V-A keeps the baked-in step
    assert d_b < 0.8 * r_b  # V-B smooths it


def test_d_vb_init_equals_rigid():
    G = plane((-1.0, -0.5, 0.0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 200, 100, 0.01)
    tc = torch.where(G["xyz"][:, 0] < 0, 10, 20).long()
    P = {10: look_at((-0.5, 0.0, 2.0), (-0.5, 0.01, 0.0)), 20: look_at((0.5, 0.0, 2.0), (0.5, 0.01, 0.0))}
    P_pgo = {u: p.clone() for u, p in P.items()}
    P_pgo[10][0, 3] += 0.03
    a = math.radians(2.0)
    Rz = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], dtype=torch.float64)
    P_pgo[20] = P_pgo[20].clone()
    P_pgo[20][:3, :3] = Rz @ P_pgo[20][:3, :3]
    dT = delta_T_dict([10, 20], [10, 20], P, P_pgo)
    xyz_r, _ = apply_rigid(G["xyz"], G["rot"], tc, dT)
    S = torch.ones(G["xyz"].shape[0], device=DEV)
    nd = build_nodes(G["xyz"], tc, tc, S, S.bool(), G["opacity"].reshape(-1), 0.3, 120.0)
    R0, t_init = node_init(nd, dT, "B")
    idx, w = influence(G["xyz"], tc.float(), nd["g"], nd["t"], 0.3, 120.0)
    pos0 = phi(G["xyz"], idx, w, nd["g"].double(), R0, t_init).float()
    deep = (nd["kappa"][idx] == tc[:, None]).all(1)
    err = (pos0 - xyz_r).norm(dim=1)[deep]
    print(f"(d) deep Gaussians: {int(deep.sum())}/{deep.numel()}, max |phi0 - rigid| = {err.max().item() * 1e3:.5f} mm")
    assert int(deep.sum()) > 1000 and err.max().item() < 1e-5


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.06)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
