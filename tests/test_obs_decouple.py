"""PLAN E1c §6 unit tests T10–T13 (synthetic, no dataset).
run: python tests/run_tests.py tests/test_obs_decouple.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("E1C_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gaussian_splatting"))
from gaussian_splatting.utils.camera_utils import SE3_exp  # noqa: E402
from utils.plane_anchor import Anchor, PlaneAnchorMap, PlaneMeas, plane_w2c  # noqa: E402

_p = os.environ.get("E1C_OBS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "obs_decouple.py"))
_spec = importlib.util.spec_from_file_location("obs_decouple", _p)
od = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(od)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROT_LR, TRANS_LR = 0.003, 0.0015          # configs/tum/base_config.yaml Training.lr
CFG = dict(sigma_normal=0.02, sigma_dist=0.01, min_singular_value=0.3, activate_after_kf=2, cell_size=20)
K = np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot(axis, deg):
    a = unit(axis)
    t = np.deg2rad(deg)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * Kx @ Kx


def anchor_map(scene):
    m = PlaneAnchorMap(CFG, K, 480, 640)
    for i, (n, d) in enumerate(scene):
        m.anchors[i] = Anchor(id=i, n_w=torch.as_tensor(n, dtype=torch.float64), d_w=float(d),
                              samples_w=torch.zeros(0, 3, dtype=torch.float64), n_obs_kf=2, active=True,
                              created_kf=0, weight=1.0)
    return m


def meas(n, d):
    return PlaneMeas(n=torch.as_tensor(n, dtype=torch.float64), d=float(d),
                     mask_cells=torch.zeros(24, 32, dtype=torch.bool), n_px=10000, mse=0.0)


# ---------------------------------------------------------------- T10
def test_T10_manual_adam_equals_adamw():
    """V = I, M = 1: DecoupledPoseStep must reproduce the repo's AdamW on the two pose deltas."""
    g = torch.Generator().manual_seed(10)
    worst = 0.0
    for s in range(200):
        tr = torch.nn.Parameter(torch.zeros(3, device=DEV))
        ro = torch.nn.Parameter(torch.zeros(3, device=DEV))
        opt = torch.optim.AdamW([{"params": [ro], "lr": ROT_LR}, {"params": [tr], "lr": TRANS_LR}], weight_decay=0.01)
        dec = od.DecoupledPoseStep(np.zeros((0, 3)), 0.3, TRANS_LR, ROT_LR, device=DEV)
        dec.Vt = torch.eye(3, device=DEV)
        dec.Vr = torch.eye(3, device=DEV)
        dec.Mt = torch.ones(3, device=DEV)
        dec.Mr = torch.ones(3, device=DEV)
        scale = 10 ** float(torch.empty(1).uniform_(-4, 1, generator=g))
        for it in range(120):
            gt_ = (torch.randn(3, generator=g) * scale).to(DEV)
            gr_ = (torch.randn(3, generator=g) * scale).to(DEV)
            tr.grad, ro.grad = gt_.clone(), gr_.clone()
            opt.step()
            ref_t, ref_r = tr.detach().clone(), ro.detach().clone()
            with torch.no_grad():
                tr.zero_()
                ro.zero_()
            d_t, d_r = dec.step(gt_, gr_)
            for a, b in ((d_t, ref_t), (d_r, ref_r)):
                rel = float(((a - b).abs() / b.abs().clamp_min(1e-30)).max())
                worst = max(worst, rel)
    print(f"T10: worst relative difference {worst:.3e} over 200 x 120 steps on {DEV}")
    assert worst < 1e-7


# ---------------------------------------------------------------- T11
def test_T11_blocks_decouple():
    """d(distance residual)/d(rot) and d(normal residual)/d(trans) vanish at tau = 0 (left update)."""
    rng = np.random.default_rng(11)
    worst_dr, worst_nt = 0.0, 0.0
    for _ in range(20):
        R = rot(rng.normal(size=3), rng.uniform(0, 180))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, rng.uniform(-2, 2, 3)
        T = torch.from_numpy(T)
        n_w = torch.from_numpy(unit(rng.normal(size=3)))
        d_w = float(rng.uniform(0.5, 3))

        def resid(tau):
            Tc = SE3_exp(tau) @ T
            Rn = Tc[:3, :3] @ n_w
            dist = -(d_w - Rn @ Tc[:3, 3])          # d_meas term is constant
            return torch.cat([dist.reshape(1), -Rn])  # [distance, normal (m - Rn) up to a constant]

        J = torch.autograd.functional.jacobian(resid, torch.zeros(6, dtype=torch.float64))
        worst_dr = max(worst_dr, float(J[0, 3:].abs().max()))
        worst_nt = max(worst_nt, float(J[1:, :3].abs().max()))
    print(f"T11: max |d dist/d rot| {worst_dr:.2e}, max |d normal/d trans| {worst_nt:.2e}")
    assert worst_dr < 1e-6 and worst_nt < 1e-6


# ---------------------------------------------------------------- T12
def test_T12_obs_basis_configs():
    f, p = np.array([0.0, -1.0, 0.0]), unit([1.0, 0.0, 0.2])
    b = od.obs_basis([f, p, p], 0.3)                         # floor + two parallel panels
    free = b["Vt"][:, b["Mt"] == 0]
    ang = np.degrees(np.arccos(min(1.0, abs(float(free[:, 0] @ unit(np.cross(f, p)))))))
    print(f"T12: floor+2 parallel: n_obs_t {int(b['Mt'].sum())}, n_obs_r {int(b['Mr'].sum())}, free dir err {ang:.2e} deg")
    assert int(b["Mt"].sum()) == 2 and free.shape[1] == 1 and ang < 0.1 and int(b["Mr"].sum()) == 3
    b = od.obs_basis(np.eye(3), 0.3)                         # corner of three perpendicular planes
    assert int(b["Mt"].sum()) == 3 and int(b["Mr"].sum()) == 3
    b = od.obs_basis([f], 0.3)                               # floor only
    free_r = b["Vr"][:, b["Mr"] == 0]
    angr = np.degrees(np.arccos(min(1.0, abs(float(free_r[:, 0] @ f)))))
    print(f"T12: floor only: n_obs_t {int(b['Mt'].sum())}, n_obs_r {int(b['Mr'].sum())}, free rot err {angr:.2e} deg")
    assert int(b["Mt"].sum()) == 1 and int(b["Mr"].sum()) == 2 and angr < 0.1
    # two normals at angle theta: smallest non-zero sigma of both blocks = sqrt(1 - cos theta)
    th = np.deg2rad(25.0)
    b = od.obs_basis([[1.0, 0, 0], [np.cos(th), np.sin(th), 0]], 0.0)
    s = np.sqrt(1 - np.cos(th))
    assert abs(sorted(b["sig_t"])[1] - s) < 1e-9 and abs(sorted(b["sig_r"])[0] - s) < 1e-9


# ---------------------------------------------------------------- T13
def _run_plane_only(decouple):
    n1, n2 = unit([0.1, -0.99, 0.1]), unit([0.8, 0.0, 0.6])
    scene = [(n1, 1.0), (n2, 1.5)]
    T_gt = np.eye(4)
    free = unit(np.cross(n1, n2))
    obs = unit(n1 + n2)                                       # inside span{n1, n2}
    m = anchor_map(scene)
    planes = [meas(*plane_w2c(n, d, T_gt)) for n, d in scene]
    pairs = [(0, 0), (1, 1)]
    T = np.eye(4)
    T[:3, 3] = 0.02 * obs + 0.01 * free                       # w2c translation offset (camera frame)
    T = torch.from_numpy(T).float().to(DEV)
    ctx = m.make_loss_ctx(planes, pairs, DEV, torch.float32)
    tr = torch.nn.Parameter(torch.zeros(3, device=DEV))
    ro = torch.nn.Parameter(torch.zeros(3, device=DEV))
    if decouple:
        dec = od.DecoupledPoseStep(ctx["m"].double().cpu().numpy(), 0.3, TRANS_LR, ROT_LR, device=DEV)
    else:
        opt = torch.optim.AdamW([{"params": [ro], "lr": ROT_LR}, {"params": [tr], "lr": TRANS_LR}], weight_decay=0.01)
    for it in range(120):
        tau = torch.cat([tr, ro])
        loss = m.plane_loss_ctx(SE3_exp(tau) @ T, ctx)
        g_t, g_r = torch.autograd.grad(loss, [tr, ro])
        with torch.no_grad():
            if decouple:
                d_t, d_r = dec.step(None, None, g_t, g_r)
                tr.copy_(d_t)
                ro.copy_(d_r)
            else:
                tr.grad, ro.grad = g_t, g_r
                opt.step()
            T = SE3_exp(torch.cat([tr, ro])) @ T
            tr.zero_()
            ro.zero_()
    t = T[:3, 3].double().cpu().numpy()
    return float(t @ free) - 0.01, abs(float(t @ obs)), dec if decouple else None


def test_T13_plane_only_free_direction():
    """Criteria revised by the user (18/10/2026): (1) sum |d_xi| on the free translation axis == 0,
    (2) position along the free direction changes < 0,1 mm; observed direction converges < 1 mm;
    E1 (ours) must move the free direction > 0,5 mm (test is sensitive). The original < 1e-6 m failed at
    2,9 um because Adam-amplified rotation noise rotates t through the left update (tests/diag_t10_t13.py)."""
    d_free_dec, obs_left_dec, dec = _run_plane_only(True)
    d_free_e1, obs_left_e1, _ = _run_plane_only(False)
    free_axes = (dec.basis["Mt"] == 0)
    dxi_free = float(dec.sum_abs_dxi_t.cpu().numpy()[free_axes].sum())
    print(f"T13: dec  sum|d_xi| free axis {dxi_free:.3e}, free change {abs(d_free_dec) * 1000:.2e} mm, "
          f"observed residual {obs_left_dec * 1000:.3f} mm, n_obs_t {int(dec.basis['Mt'].sum())}")
    print(f"T13: ours free change {abs(d_free_e1) * 1000:.3f} mm, observed residual {obs_left_e1 * 1000:.3f} mm")
    assert free_axes.sum() == 1 and dxi_free == 0.0
    assert abs(d_free_dec) < 1e-4 and obs_left_dec < 1e-3
    assert abs(d_free_e1) > 5e-4
