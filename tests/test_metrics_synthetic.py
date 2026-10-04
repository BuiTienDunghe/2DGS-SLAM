"""Synthetic checks for M1 (e_dl) and M2 (e_step) through the real rasterizer.

Run: python tests/test_metrics_synthetic.py   (VRAM-capped to ~1 GB)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform.metrics import (edl_paired, e_step, eval_pixel_set, gap_maps, layer_masks,  # noqa: E402
                            step_sets)
from deform.render_utils import DEV, make_cam  # noqa: E402

INTR = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "W": 640, "H": 480}


class FakeState:
    def __init__(self, tc, active, poses):
        self.intr = INTR
        self.kf_uids = sorted(poses)
        self.gpre = {"tc": torch.as_tensor(tc).int()}
        self.active = torch.as_tensor(active).bool()
        self.poses_gt = {}
        self.tr = {"depth_type": "median", "depth_min_threshold": 0.1, "depth_max_threshold": 10.0}

    def cam(self, uid, which="pgo", poses=None):
        return make_cam(uid, poses[uid], self.intr)


def plane(n_side=101, spacing=0.01, z=2.0, scale=0.006):
    t = (torch.arange(n_side, device=DEV, dtype=torch.float32) - n_side // 2) * spacing
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    xyz = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.full_like(xx.reshape(-1), z)], -1)
    N = xyz.shape[0]
    return {
        "xyz": xyz,
        "rot": torch.tensor([1.0, 0, 0, 0], device=DEV).repeat(N, 1),
        "scale": torch.full((N, 2), scale, device=DEV),
        "opacity": torch.full((N, 1), 0.99, device=DEV),
        "f_dc": torch.full((N, 1, 3), 0.5, device=DEV),
    }


def cat(a, b):
    return {k: torch.cat([a[k], b[k]], 0) for k in a}


def cfg():
    c = dcfg.resolve({}, {"Dataset": {"type": "tum"}, "Training": {"prune_size_threshold": 0.25}})
    c["corr"]["stride"] = 2
    return c


def test_edl_two_layers_1cm():
    old, new = plane(z=2.0), plane(z=2.01)
    G = cat(old, new)
    n = old["xyz"].shape[0]
    active = torch.cat([torch.zeros(n), torch.ones(n)]).bool()
    poses = {0: torch.eye(4, dtype=torch.float64)}
    st = FakeState(torch.zeros(2 * n), active, poses)
    m_old, m_new = layer_masks(active)
    Pi = eval_pixel_set(st, G, poses, [(0, None)], m_old, m_new, cfg())
    res, npix = edl_paired(Pi, {"m": gap_maps(st, G, poses, Pi, m_old, m_new)})
    e = res["m"]["median"]
    print(f"e_dl = {e * 1e3:.3f} mm on {npix} px")
    assert npix > 1000
    assert abs(e - 0.01) < 1e-3

    # identical layers -> ~0
    G0 = cat(old, plane(z=2.0))
    res0, _ = edl_paired(Pi, {"m": gap_maps(st, G0, poses, Pi, m_old, m_new)})
    print(f"e_dl (aligned) = {res0['m']['median'] * 1e3:.4f} mm")
    assert res0["m"]["median"] < 1e-4


def test_edl_tilted_camera_normals_in_camera_frame():
    """F10: rend_normal is in the camera frame -> gap stays 1 cm when the camera is rotated 25 deg."""
    old, new = plane(n_side=301, z=2.0), plane(n_side=301, z=2.01)
    G = cat(old, new)
    n = old["xyz"].shape[0]
    active = torch.cat([torch.zeros(n), torch.ones(n)]).bool()
    a = np.radians(25.0)
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, :3] = torch.tensor([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    c2w[0, 3] = -0.9  # keep the plane in view
    poses = {0: c2w}
    st = FakeState(torch.zeros(2 * n), active, poses)
    m_old, m_new = layer_masks(active)
    Pi = eval_pixel_set(st, G, poses, [(0, None)], m_old, m_new, cfg())
    res, npix = edl_paired(Pi, {"m": gap_maps(st, G, poses, Pi, m_old, m_new)})
    e = res["m"]["median"]
    print(f"tilted: e_dl = {e * 1e3:.3f} mm on {npix} px")
    assert npix > 1000 and abs(e - 0.01) < 1e-3


def _step_setup(step):
    G = plane(n_side=101)
    right = G["xyz"][:, 0] >= 0
    G["xyz"][right, 2] += step
    tc = torch.where(right, 2, 1)
    active = torch.ones(G["xyz"].shape[0]).bool()
    poses = {0: torch.eye(4, dtype=torch.float64)}
    st = FakeState(tc.cpu(), active, poses)
    return st, G, poses


def test_estep_detects_step():
    st, G, poses = _step_setup(0.01)
    sets = step_sets(st, G, poses, [0], seed=0)
    r = e_step(G["xyz"], sets)
    print(f"step 1 cm: e_step = {r['e_step'] * 1e3:.3f} mm (B={r['n_B']})")
    assert r["n_B"] > 50 and r["e_step"] > 1e-3

    st, G, poses = _step_setup(0.0)
    sets = step_sets(st, G, poses, [0], seed=0)
    r = e_step(G["xyz"], sets)
    print(f"no step:   e_step = {r['e_step'] * 1e3:.4f} mm")
    assert abs(r["e_step"]) < 1e-4


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.06)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
