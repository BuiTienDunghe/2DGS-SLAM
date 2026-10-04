"""P4 gate G4: the online backend hook reproduces the offline pipeline on a real dump.

Builds a fake BackEnd from one loop dump (GaussianModel with optimizer, Camera dicts, key frames
re-read from the dataset, a PoseGraphManager stand-in with the dumped PGO poses), then
  1. mode rigid: BackEnd.apply_rigid_correction == offline replay (exact)
  2. deform: deform.online.backend_correct == deform.pipeline.correct_map offline
     (max ||d mu|| < 1e-4 m, e_dl difference < 0.1 mm)
usage: python tests/test_online_hook.py DUMP.pt [DEFORM_YAML]
"""
import copy
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import gtsam  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from munch import munchify  # noqa: E402
from torch import nn  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.dump import EventState, load_dump  # noqa: E402
from deform.online import backend_correct  # noqa: E402
from deform.pipeline import correct_map, inp_from_dump, with_pos  # noqa: E402
from deform.render_utils import DEV, make_cam  # noqa: E402
from deform.rigid import quat_angle, replay  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_splatting.utils.general_utils import inverse_sigmoid  # noqa: E402
from utils.slam_backend import BackEnd  # noqa: E402


class FakePGO:
    def __init__(self, d):
        self.poses = d["poses_pgo"]
        self.graph_initials = gtsam.Values()
        for u, P in self.poses.items():
            self.graph_initials.insert(gtsam.symbol("x", int(u)), gtsam.Pose3(P.double().numpy()))
        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.last_error = 0.0

    def get_optimized_node_pose(self, idx):
        return self.poses[int(idx)].double().numpy()


def fake_backend(d, st):
    cfg = d["config"]
    gp = d["gauss_pre"]
    g = GaussianModel(cfg["model_params"]["sh_degree"], cfg["model_params"]["initial_opacity"])
    N = gp["xyz"].shape[0]
    g._xyz = nn.Parameter(gp["xyz"].float().to(DEV).clone())
    if "scale_raw" in gp:  # plan v4 dumps: raw parameters -> the backend's activations reproduce exactly
        g._rotation = nn.Parameter(gp["rot_raw"].float().to(DEV).clone())
        g._scaling = nn.Parameter(gp["scale_raw"].float().to(DEV).clone())
        g._opacity = nn.Parameter(gp["opacity_raw"].float().to(DEV).clone())
    else:
        g._rotation = nn.Parameter(gp["rot"].float().to(DEV).clone())
        g._scaling = nn.Parameter(torch.log(gp["scale"].float().to(DEV)))
        g._opacity = nn.Parameter(inverse_sigmoid(gp["opacity"].float().to(DEV).clamp(1e-6, 1 - 1e-6)))
    g._features_dc = nn.Parameter(gp["f_dc"].float().to(DEV).clone())
    g._features_rest = nn.Parameter(torch.zeros((N, 0, 3), device=DEV))
    g.unique_kfIDs = gp["tc"].int().to(DEV)[:, None]
    g.birth_kfIDs = gp["t0"].int().to(DEV)[:, None]
    g.last_observe_ids = gp["tl"].int().to(DEV)[:, None]
    g.min_observed_depth = gp["dc"].float().to(DEV)[:, None]
    g.active_mask = gp["active"].bool().to(DEV)[:, None]
    g.training_setup(munchify(cfg["opt_params"]))
    be = types.SimpleNamespace()
    be.gaussians, be.config, be.device, be.dtype = g, cfg, "cuda", torch.float32
    be.seed = int(d["meta"].get("seed", 0))
    be.old_than_N_keyframe = int(d["meta"].get("old_than_N_keyframe", cfg["Training"]["old_than_N_keyframe"]))
    be.all_cam_ids = list(d["all_cam_ids"])
    be.all_cameras, be.key_cameras, be.key_frames = {}, {}, {}
    for u, P in d["poses_pre"].items():
        cam = make_cam(u, P, d["intrinsics"])
        be.all_cameras[u] = cam
    from utils.slam_frontend import Frame
    for u in d["keyframe_uids"]:
        cam = be.all_cameras[u]
        cm = d["cam_meta"][u]
        cam.scale = nn.Parameter(torch.tensor([cm["scale"]], device=DEV))
        cam.shift = nn.Parameter(torch.tensor([cm["shift"]], device=DEV))
        be.key_cameras[u] = cam
        ds = __import__("deform.dump", fromlist=["get_dataset"]).get_dataset(cfg)
        img, _, depth, _ = ds[int(u)]
        be.key_frames[u] = Frame(camera_id=u, rgb=img.to(DEV), depth=depth.to(DEV))
    be.pgo = FakePGO(d)
    return be


def main(dump_path, deform_yaml=None):
    torch.manual_seed(0)
    d = load_dump(dump_path)
    st = EventState(d)
    ok = True
    # 1. rigid: backend method on the fake backend vs offline replay
    be = fake_backend(d, st)
    BackEnd.apply_rigid_correction(be)
    xyz_r, rot_r, _ = replay(d, use_online=False)  # same fp32 computation from the dumped poses
    dx = (be.gaussians.get_xyz.detach() - xyz_r).norm(dim=1).max().item()
    da = quat_angle(be.gaussians.get_rotation.detach(), rot_r).max().item()
    print(f"[rigid] hook vs offline replay: max dxyz {dx:.3e} m, max dangle {da:.3e} rad")
    ok &= dx < 1e-6 and da < 1e-5
    # 2. deform
    user = {"mode": "deform", "variant": "B"}
    if deform_yaml:
        import yaml
        with open(deform_yaml) as f:
            user = yaml.safe_load(f)
    cfg = dcfg.resolve(user, d["config"])
    cfg["seed"] = int(d["meta"].get("seed", 0))
    be = fake_backend(d, st)
    cur = be.key_cameras[int(d["meta"]["cur_uid"])]
    loop = be.key_cameras[int(d["meta"]["loop_uid"])]
    applied, log = backend_correct(be, cur, loop, cfg)
    res = correct_map(inp_from_dump(d, st.frame), cfg, cfg["variant"])
    print(f"[deform] online accepted={applied} ({log.get('fallback_reason')}), offline accepted={res['accepted']} "
          f"({res['reason']}), t_online={log.get('t_hook_s')}")
    if not applied:
        BackEnd.apply_rigid_correction(be)
    xyz_on = be.gaussians.get_xyz.detach()
    dmu = (xyz_on - res["xyz"]).norm(dim=1).max().item()
    print(f"[deform] online vs offline max ||d mu|| = {dmu:.3e} m")
    ok &= (applied == res["accepted"]) and dmu < 1e-4
    poses_on = {u: torch.linalg.inv(c.T.double()).cpu() for u, c in be.all_cameras.items()}
    G = res["G_rig"]
    m_old, m_new = M.layer_masks(st.active)
    JL = sorted(set(res["J_opt"]) | set(res["J_eval"]))
    _, J_eval = M.split_opt_eval(JL, int(st.intr["H"]), int(st.intr["W"])) if JL else ([], [])
    if J_eval:
        Pi = M.eval_pixel_set(st, G, res["poses_rig"], J_eval, m_old, m_new, cfg)
        maps = {"off": M.gap_maps(st, with_pos(G, res["xyz"], res["rot"]), res["poses"], Pi, m_old, m_new),
                "on": M.gap_maps(st, with_pos(G, xyz_on, be.gaussians.get_rotation.detach()), poses_on, Pi, m_old, m_new)}
        e, _ = M.edl_paired(Pi, maps)
        diff = abs((e["on"]["median"] or 0) - (e["off"]["median"] or 0))
        print(f"[deform] e_dl offline {1e3 * (e['off']['median'] or 0):.3f} mm, online {1e3 * (e['on']['median'] or 0):.3f} mm, diff {1e3 * diff:.4f} mm")
        ok &= diff < 1e-4
    print("G4_REPLAY_PASS" if ok else "G4_REPLAY_FAIL")
    return ok


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
