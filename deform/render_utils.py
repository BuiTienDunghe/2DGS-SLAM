"""Rendering of raw Gaussian tensors (subsets, attribute images) without touching GaussianModel.

plan v4 A3: inside det_scope(True) every render here uses diff_surfel_rasterization_det, a copy of the
rasterizer whose per-Gaussian `contributions` are accumulated in int64 fixed point (scale 2^34) instead of
float atomicAdd, so the sum does not depend on the thread order. SLAM itself keeps the original build.
"""
import contextlib
import math
import os
import warnings

import numpy as np
import torch
import diff_surfel_rasterization as _RAST_FLOAT

try:
    import diff_surfel_rasterization_det as _RAST_DET
except ImportError:  # the det build is optional; det_scope(True) then fails loudly
    _RAST_DET = None

from utils.camera_utils import Camera

DEV = os.environ.get("DEFORM_DEVICE", "cuda")  # "cpu" for CPU-only checks while a SLAM run owns the GPU (R6)
_STATE = {"det_render": False}


def det_render_active():
    return _STATE["det_render"]


@contextlib.contextmanager
def det_scope(render):
    """Use the deterministic rasterizer build for every render issued inside the block."""
    if render and _RAST_DET is None:
        raise RuntimeError("det.render requested but diff_surfel_rasterization_det is not installed")
    prev = _STATE["det_render"]
    _STATE["det_render"] = bool(render)
    try:
        yield
    finally:
        _STATE["det_render"] = prev


@contextlib.contextmanager
def torch_det_scope(enabled, sink=None):
    """torch.use_deterministic_algorithms(True, warn_only=True) inside the block (restored after). Ops that
    still have no deterministic implementation warn; the unique messages are appended to `sink` (a list)."""
    if not enabled:
        yield
        return
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
        if sink is not None:
            seen = set(sink)
            for w in rec:
                m = str(w.message).split("\n")[0][:200]
                if m not in seen:
                    sink.append(m)
                    seen.add(m)


def make_cam(uid, c2w, intr, gt_c2w=None):
    """Camera at c2w pose (4x4, any dtype/device). intr: {fx, fy, cx, cy, W, H}."""
    c2w = torch.as_tensor(c2w, dtype=torch.float64)
    w2c = torch.linalg.inv(c2w).float().to(DEV)
    gt = None
    if gt_c2w is not None:
        gt = torch.linalg.inv(torch.as_tensor(gt_c2w, dtype=torch.float64)).float().to(DEV)
    K = np.array([[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]])
    return Camera(uid, w2c, gt, K, int(intr["H"]), int(intr["W"]), device=DEV)


def _empty(cam):
    H, W = int(cam.image_height), int(cam.image_width)
    z1 = torch.zeros((1, H, W), device=DEV)
    return {
        "render": torch.zeros((3, H, W), device=DEV), "rend_alpha": z1, "rend_normal": torch.zeros((3, H, W), device=DEV),
        "rend_depth_median": z1, "rend_depth_expected": z1, "rend_dist": z1,
        "contributions": torch.zeros((0,), device=DEV),
    }


@torch.no_grad()
def rasterize(cam, xyz, rot, scale, opacity, shs=None, colors=None, error_img=None, bg=None):
    """All tensors already selected (subset). Exactly one of shs [N,1,3] / colors [N,3]."""
    if xyz.shape[0] == 0:
        return _empty(cam)
    H, W = int(cam.image_height), int(cam.image_width)
    if error_img is None:
        error_img = torch.ones((H, W), dtype=torch.float32, device=DEV)
    if bg is None:
        bg = torch.zeros(3, dtype=torch.float32, device=DEV)
    mod = _RAST_DET if _STATE["det_render"] else _RAST_FLOAT
    settings = mod.GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        projmatrix_raw=cam.projection_matrix, error_img=error_img.float().contiguous(),
        sh_degree=0, campos=cam.camera_center, prefiltered=False, debug=False,
    )
    means2D = torch.zeros_like(xyz)
    img, radii, allmap, contrib = mod.GaussianRasterizer(raster_settings=settings)(
        means3D=xyz.float().contiguous(), means2D=means2D,
        shs=None if shs is None else shs.float().contiguous(),
        colors_precomp=None if colors is None else colors.float().contiguous(),
        opacities=opacity.float().contiguous(), scales=scale.float().contiguous(),
        rotations=rot.float().contiguous(), cov3D_precomp=None,
        theta=cam.cam_rot_delta, rho=cam.cam_trans_delta,
    )
    alpha = allmap[1:2]
    d_exp = torch.nan_to_num(allmap[0:1] / alpha, 0, 0)
    return {
        "render": img, "rend_alpha": alpha, "rend_normal": allmap[2:5],
        "rend_depth_median": torch.nan_to_num(allmap[5:6], 0, 0), "rend_depth_expected": d_exp,
        "rend_dist": allmap[6:7], "contributions": contrib,
    }


def render_subset(cam, G, mask=None, error_img=None):
    """G: dict xyz/rot/scale/opacity/f_dc on GPU. Returns pkg; contributions scattered to [N]."""
    N = G["xyz"].shape[0]
    if mask is None:
        mask = torch.ones(N, dtype=torch.bool, device=DEV)
    pkg = rasterize(cam, G["xyz"][mask], G["rot"][mask], G["scale"][mask], G["opacity"][mask],
                    shs=G["f_dc"][mask], error_img=error_img)
    full = torch.zeros(N, device=DEV)
    if pkg["contributions"].numel():
        full[mask] = pkg["contributions"].float()
    pkg["contrib_full"] = full
    return pkg


def render_attribute(cam, G, attr, mask=None):
    """Alpha-blended attribute image [C,H,W] (C<=3), divided by alpha (0 where alpha==0)."""
    N = G["xyz"].shape[0]
    if mask is None:
        mask = torch.ones(N, dtype=torch.bool, device=DEV)
    a = attr.float()
    if a.dim() == 1:
        a = a[:, None]
    C = a.shape[1]
    colors = torch.zeros((N, 3), device=DEV)
    colors[:, :C] = a
    pkg = rasterize(cam, G["xyz"][mask], G["rot"][mask], G["scale"][mask], G["opacity"][mask],
                    colors=colors[mask])
    alpha = pkg["rend_alpha"]
    img = torch.where(alpha > 1e-6, pkg["render"][:C] / alpha.clamp_min(1e-6), torch.zeros_like(pkg["render"][:C]))
    return img, alpha
