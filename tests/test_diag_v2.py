"""Plan v2 tests T2-T6 (T1/T7: tools/check_t1.py + tools/check_t1_strict.py).

T2  weight mode with S == 1 everywhere == hard mode with tau = 0          (real dump)
T3  checkerboard split / D3-all: pair source pixels never in a cell B of J_eval (= Pi*)   (real dump)
T4  oracle on a synthetic two-scan corner (new layer bent by a known smooth field): e_dl -80 %
T5  ICP recovers a known rigid offset (20 mm, 1 deg) of a box seen from inside: <= 0.5 mm, <= 0.05 deg, Q1-Q4 pass
T6  one plane, new layer slid along the plane: Q3 must reject (unobservable translation)
usage: python tests/test_diag_v2.py [DUMP.pt]   (T2/T3 need a dump; default: first TUM tuning dump)
"""
import copy
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform import pipeline as PL  # noqa: E402
from deform.registration import register, se3_exp  # noqa: E402
from deform.render_utils import DEV, make_cam  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- T5 / T6: analytic renders
class Cam:
    def __init__(self, intr):
        self.fx, self.fy, self.cx, self.cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        self.image_width, self.image_height = intr["W"], intr["H"]


def raycast(planes, intr):
    """Camera at the origin looking down +z. planes: [(n, d)] with n.x = d. The camera is inside the
    convex region, so the visible surface is the closest positive hit. Returns depth, normal, alpha."""
    H, W = intr["H"], intr["W"]
    v, u = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float64),
                          torch.arange(W, device=DEV, dtype=torch.float64), indexing="ij")
    r = torch.stack(((u - intr["cx"]) / intr["fx"], (v - intr["cy"]) / intr["fy"], torch.ones_like(u)), 0)
    best = torch.full((H, W), float("inf"), dtype=torch.float64, device=DEV)
    N = torch.zeros((3, H, W), dtype=torch.float64, device=DEV)
    for n, d in planes:
        n = torch.as_tensor(n, dtype=torch.float64, device=DEV)
        den = (n[:, None, None] * r).sum(0)
        t = torch.where(den.abs() > 1e-12, d / den, torch.full_like(den, float("inf")))
        hit = (t > 0) & (t < best)
        best = torch.where(hit, t, best)
        N = torch.where(hit[None], -n[:, None, None].expand(3, H, W) * torch.sign(den)[None], N)
    alpha = torch.isfinite(best) & (best < 20.0)
    D = torch.where(alpha, best, torch.zeros_like(best))
    return D.float(), N.float(), alpha.float()


def moved(planes, Hm):
    """Planes of the scene transformed by Hm (points x -> R x + t)."""
    R, t = Hm[:3, :3], Hm[:3, 3]
    out = []
    for n, d in planes:
        n = torch.as_tensor(n, dtype=torch.float64, device=DEV)
        n2 = R @ n
        out.append((n2.tolist(), float(d + n2 @ t)))
    return out


def box_planes():
    # a 4 x 3 x 5 m room around the camera, looking into a corner region (several non-parallel planes)
    return [((1, 0, 0), 1.6), ((-1, 0, 0), 2.4), ((0, 1, 0), 1.0), ((0, -1, 0), 2.0),
            ((0, 0, 1), 3.0), ((0, 0, -1), 2.0), ((0.6, 0, 0.8), 2.6)]


def layers(planes_old, planes_new, intr):
    Do, No, Oo = raycast(planes_old, intr)
    Dn, Nn, On = raycast(planes_new, intr)
    return {"Do": Do, "Dn": Dn, "No": No, "Nn": Nn, "Oo": Oo, "On": On}


def rc_tum():
    c = dcfg.resolve({}, {"Dataset": {"type": "tum"}, "Training": {"prune_size_threshold": 0.25}})
    return c["reg"], c["energy"]["sigma_c"]


INTR = {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3, "W": 640, "H": 480}


def test_t5_icp_recovers_rigid():
    a = math.radians(1.0)
    Hm = torch.eye(4, dtype=torch.float64, device=DEV)
    Hm[:3, :3] = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]],
                              dtype=torch.float64, device=DEV)
    Hm[:3, 3] = torch.tensor([0.012, -0.010, 0.012], dtype=torch.float64, device=DEV)  # |t| = 19.7 mm
    P = box_planes()
    # new layer = old scene moved by Hm^-1, so H = Hm maps the new layer onto the old one
    L = layers(P, moved(P, torch.linalg.inv(Hm)), INTR)
    rc, sc = rc_tum()
    res = register(L, Cam(INTR), rc, sc)
    E = torch.linalg.inv(Hm) @ res["H_c"]
    et = float(E[:3, 3].norm()) * 1e3
    er = math.degrees(math.acos(max(-1.0, min(1.0, (float(torch.trace(E[:3, :3])) - 1) / 2))))
    s = res["stats"]
    print(f"T5: |t_err| {et:.4f} mm, rot err {er:.5f} deg, Q {res['Q']}, inliers {s['n_inlier']}/{s['n_overlap']}, "
          f"rms {1e3 * s['rms_before_m']:.2f} -> {1e3 * s['rms_after_m']:.4f} mm, sig_t {1e3 * s['sig_t_m']:.4f} mm, "
          f"sig_r {s['sig_r_deg']:.5f} deg, eig ratio {s['eig_ratio']:.2e}, iters {s['iters']}")
    assert et <= 0.5 and er <= 0.05, "T5 accuracy"
    assert res["passed"], f"T5 Q1-Q4 failed: {res['fail']}"


def test_t6_degenerate_plane_rejected():
    P = [((0, 0, 1), 2.0)]  # one wall, perpendicular to the optical axis
    Hm = torch.eye(4, dtype=torch.float64, device=DEV)
    Hm[:3, 3] = torch.tensor([0.020, 0.0, 0.0], dtype=torch.float64, device=DEV)  # slide along the wall
    L = layers(P, moved(P, torch.linalg.inv(Hm)), INTR)
    rc, sc = rc_tum()
    res = register(L, Cam(INTR), rc, sc)
    s = res["stats"]
    print(f"T6: Q {res['Q']}, sig_t {s['sig_t_m']:.3e} m, sig_r {s['sig_r_deg']:.3e} deg, eig ratio {s['eig_ratio']:.2e}")
    assert not res["Q"]["Q3"], "T6: Q3 accepted a degenerate (single-plane) registration"


# ---------------------------------------------------------------- T4: synthetic oracle
def test_t4_oracle_synthetic():
    import test_solver_synthetic as TS
    torch.manual_seed(0)
    inp = TS.scene(TS.bend)
    base = TS.cfg_tum()
    base["energy"]["w_p"] = 1.0
    cfg = copy.deepcopy(base)
    cfg["corr"].update({"mode": "oracle", "stride": 1, "eps_d": base["corr"]["eps_eval"],
                        "s_gate": {"mode": "off"}, "max_per_node": 10 ** 9, "max_total": 10 ** 9})
    cfg["accept"]["enforce"] = False
    cfg["solver"]["tol_mode"] = "energy"
    res = PL.correct_map(inp, cfg, "A")
    st = TS.St(inp)
    m_old, m_new = M.layer_masks(inp["active"])
    Pi, _ = M.pi_star(st, res["G_rig"], res["poses_rig"], res["J_eval"], m_old, m_new, base)
    Gd = dict(inp["g"])
    Gd["xyz"], Gd["rot"] = res["xyz"], res["rot"]
    maps = {"rigid": M.gap_maps(st, res["G_rig"], res["poses_rig"], Pi, m_old, m_new),
            "O1": M.gap_maps(st, Gd, res["poses"], Pi, m_old, m_new)}
    e, npix = M.edl_paired(Pi, maps)
    red = 1 - e["O1"]["median"] / e["rigid"]["median"]
    print(f"T4: oracle pairs {res['log'].get('corr_capped')}, e_dl on Pi* rigid {1e3 * e['rigid']['median']:.2f} -> "
          f"O1 {1e3 * e['O1']['median']:.2f} mm ({npix} px), reduction {100 * red:.1f} %")
    assert red >= 0.80


# ---------------------------------------------------------------- T2 / T3 on a real dump
def _dump_inp(path):
    from deform.dump import EventState, load_dump
    from utils.loop_dump import load_deform_config
    d = load_dump(path)
    st = EventState(d)
    inp = PL.inp_from_dump(d, st.frame)
    base = dcfg.resolve(load_deform_config(os.path.join(ROOT, "configs", "deform", "selected.yaml")), st.config)
    base["seed"] = int(st.meta.get("seed", 0))
    return st, inp, base


def test_t2_weight_equals_hard_tau0(path):
    st, inp, base = _dump_inp(path)
    orig = PL.compute_reliability

    def ones(g, *a, **k):
        r = orig(g, *a, **k)
        r["S"] = torch.ones_like(r["S"])
        r["R"] = torch.ones_like(r["R"], dtype=torch.bool)
        return r

    PL.compute_reliability = ones
    try:
        cache = {}
        hard = copy.deepcopy(base)
        hard["corr"]["s_gate"] = {"mode": "hard", "tau": 0.0, "floor": 0.05, "layers": "both"}
        hard2 = copy.deepcopy(hard)
        hard2["corr"]["s_gate"]["tau"] = -0.0  # same semantics, different cache key -> stages [2]-[5] rebuilt
        wt = copy.deepcopy(base)
        wt["corr"]["s_gate"] = {"mode": "weight", "tau": None, "floor": 0.05, "layers": "both"}
        a = PL.correct_map(inp, hard, "A", cache)
        a2 = PL.correct_map(inp, hard2, "A", cache)  # rebuild noise of the existing code (atomics)
        b = PL.correct_map(inp, wt, "A", cache)  # same cached S; pairs rebuilt for the new key
        wts = cache[PL.prep_key(wt, "A")]["pairs"].get("omega")
    finally:
        PL.compute_reliability = orig
    noise = float((a["xyz"] - a2["xyz"]).norm(dim=1).max())
    dx = float((a["xyz"] - b["xyz"]).norm(dim=1).max())
    la, lb = a["log"], b["log"]
    same = (la["corr_capped"] == lb["corr_capped"] and a["accepted"] == b["accepted"]
            and la["lbfgs_iters"] == lb["lbfgs_iters"])
    print(f"T2: omega unique values {torch.unique(wts).tolist()[:5]}; pairs {la['corr_capped']} vs {lb['corr_capped']}, "
          f"accepted {a['accepted']} vs {b['accepted']}, iters {la['lbfgs_iters']} vs {lb['lbfgs_iters']}, "
          f"max |dxyz| weight-vs-hard {dx:.3e} m (hard-vs-hard rebuild noise {noise:.3e} m)")
    assert same and dx <= max(1e-9, 2.0 * noise)


def test_t3_split_disjoint(path):
    import run_diag_v2 as RD
    st, inp, base = _dump_inp(path)
    H, W = int(st.intr["H"]), int(st.intr["W"])
    cache = {}
    ref = PL.correct_map(inp, base, "A", cache)
    regB = {u: M.checker(H, W, base["corr"]["checker"], 1) for u in ref["J_eval"]}
    bad = {}
    for name in ("F3", "ALL-1", "D3-all"):
        cfg = RD.make_cfg(base, name)
        PL.correct_map(inp, cfg, "A", cache)
        ctx = cache[PL.prep_key(cfg, "A")]
        p = ctx["pairs"]
        n_eval_src = int(sum(int((p["src_uid"] == u).sum()) for u in regB)) if p.get("P", 0) else 0
        bad[name] = RD.pair_split_overlap(ctx, regB)
        print(f"T3: {name}: pairs {p.get('P', 0)}, from J_eval keyframes {n_eval_src}, in cells B {bad[name]}")
    assert all(v == 0 for v in bad.values())


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    dumps = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "results_exp/tum/room/20261002013837_rigid_s0/loop_dumps/*.pt")))[:1]
    which = os.environ.get("TESTS", "T5,T6,T4,T2,T3").split(",")
    fns = {"T5": test_t5_icp_recovers_rigid, "T6": test_t6_degenerate_plane_rejected, "T4": test_t4_oracle_synthetic,
           "T2": lambda: test_t2_weight_equals_hard_tau0(dumps[0]), "T3": lambda: test_t3_split_disjoint(dumps[0])}
    for k in which:
        fns[k]()
        print("PASS", k, flush=True)
