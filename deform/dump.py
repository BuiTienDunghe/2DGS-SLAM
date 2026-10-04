"""Load loop-event dumps (Appendix B) and rebuild the offline state: Gaussians, poses, cameras, RGB-D."""
import glob
import os

import torch
from munch import munchify

from deform.render_utils import DEV, make_cam

_DATASETS = {}


def list_dumps(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "loop_dumps", "event_*.pt")))


def load_dump(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def gauss_to_dev(g, xyz=None, rot=None):
    """Gaussian dict on GPU; optional replacement xyz/rot (e.g. post state)."""
    return {
        "xyz": (g["xyz"] if xyz is None else xyz).float().to(DEV),
        "rot": (g["rot"] if rot is None else rot).float().to(DEV),
        "scale": g["scale"].float().to(DEV),
        "opacity": g["opacity"].float().to(DEV),
        "f_dc": g["f_dc"].float().to(DEV),
    }


def get_dataset(config):
    """Dataset object (cached per dataset_path) to re-read keyframe RGB-D by uid."""
    from utils.dataset import load_dataset
    key = config["Dataset"]["dataset_path"]
    if key not in _DATASETS:
        mp = munchify(config["model_params"])
        _DATASETS[key] = load_dataset(mp, mp.source_path, config=config)
    return _DATASETS[key]


def load_frame(config, uid, cam_meta=None):
    """rgb [3,H,W], sensor depth [H,W] scaled like mapping (scale*d + shift)."""
    ds = get_dataset(config)
    img, _, depth, _ = ds[int(uid)]
    if cam_meta is not None and uid in cam_meta:
        depth = cam_meta[uid]["scale"] * depth + cam_meta[uid]["shift"]
    return img.to(DEV), depth.to(DEV)


class EventState:
    """Everything one loop event needs offline."""

    def __init__(self, dump):
        self.d = dump
        self.meta = dump["meta"]
        self.config = dump["config"]
        self.intr = dump["intrinsics"]
        self.kf_uids = list(dump["keyframe_uids"])
        self.gpre = dump["gauss_pre"]
        self.N = self.gpre["xyz"].shape[0]
        self.active = self.gpre["active"].bool()
        self.cam_meta = dump["cam_meta"]
        self.poses_pre = dump["poses_pre"]
        self.poses_pgo = dump["poses_pgo"]
        self.poses_gt = dump.get("poses_gt", {})
        self.tr = self.config["Training"]
        self.dtype_name = self.config["Dataset"]["type"]

    def cam(self, uid, which="pgo", poses=None):
        P = poses if poses is not None else (self.poses_pgo if which == "pgo" else self.poses_pre)
        return make_cam(uid, P[uid], self.intr, self.poses_gt.get(uid))

    def frame(self, uid):
        return load_frame(self.config, uid, self.cam_meta)

    def delta_t(self):
        """Delta_t = W * median(u_{j+1} - u_j) over keyframes (frame-index units)."""
        u = torch.tensor(sorted(self.kf_uids), dtype=torch.float64)
        W = int(self.meta.get("old_than_N_keyframe", self.tr["old_than_N_keyframe"]))
        if u.numel() < 2:
            return float(W)
        return float(W * torch.median(u[1:] - u[:-1]).item())
