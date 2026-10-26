"""PLAN MONOGS-E1C §3: reproducibility helpers and write-only outputs (added; MonoGS behaviour unchanged).

configure     : --seed / --range / --method -> config["Repro"]; --method e1c|e1c_s loads the PlaneTracking block
seed_process  : seed random / numpy / torch (+cuda) / open3d, cudnn deterministic; logs TF32 + process name
apply_range   : keep dataset frames [BEGIN, END) (frame BEGIN becomes frame 0, initialised at GT like MonoGS)
write_outputs : traj_kf.txt, traj_all.txt, traj_gt_all.txt (TUM, camera->world), kf_indices.txt, timing.json
"""
import json
import os
import random

import numpy as np
import torch

from utils.logging_utils import Log

E1C_CONFIG = "configs/rgbd/tum/e1c.yaml"


PW_CONFIG = "configs/rgbd/tum/pw.yaml"
PW_CALIB = "configs/rgbd/tum/pw_calib.yaml"
# PLAN MONOGS-PW §2.7: (match_mode, robust, anchor_mode, kf_trigger_t1, scale_mode, pgo_mode)
PW_METHODS = {
    "full": ("nn", True, "attach", True, "info", "ours"),
    "full_nob": ("nn", True, "attach", True, "e1c", "ours"),
    "full_nopgo": ("nn", True, "attach", True, "info", "none"),
    "full_nomatch": ("e1c", False, "attach", True, "info", "ours"),
    "full_nolife": ("nn", True, "e1c", True, "info", "ours"),
    "full_nokf": ("nn", True, "attach", False, "info", "ours"),
    "sgs": ("nn", True, "attach", True, "info", "sgs"),
}
PW2_CONFIG = "configs/rgbd/tum/pw2.yaml"
PW2_CALIB = "configs/rgbd/tum/pw2_calib.yaml"
# PLAN MONOGS-PW2 §2.7: method -> PGO mode (everything else from pw2.yaml)
PW2_METHODS = {"pw2": "ours", "pw2_nopgo": "none"}
METHODS = ["base", "e1c", "e1c_s"] + list(PW_METHODS) + list(PW2_METHODS) + ["base_kf"]


def configure(config, args):
    config["Repro"] = dict(seed=int(args.seed), range=list(args.range) if args.range else None, method=args.method,
                           frame_offset=int(args.range[0]) if args.range else 0,
                           stiffness_every=int(getattr(args, "stiffness_every", 0) or 0))
    if args.method == "base_kf":   # PLAN MONOGS-PW2 §2.6: base + keyframe every N frames (OR with MonoGS)
        n = int(getattr(args, "kf_force_every", 0) or 0)
        if n <= 0:
            raise ValueError("--method base_kf needs --kf-force-every N > 0")
        config["Training"]["kf_force_every"] = n
        config["PlaneTracking"] = dict(enable=False)
        return config
    if args.method in PW2_METHODS:   # PLAN MONOGS-PW2 §2.7
        import yaml
        with open(E1C_CONFIG) as f:
            pt = yaml.safe_load(f)["PlaneTracking"]
        with open(PW_CONFIG) as f:
            pw = yaml.safe_load(f)
        with open(PW2_CONFIG) as f:
            pw2 = yaml.safe_load(f)
        with open(PW2_CALIB) as f:
            cal = yaml.safe_load(f)
        pt.update(pw["PlaneTracking"])
        pt.update(pw2["PlaneTracking"])
        pt.update(enable=True, pw=True, kf_trigger_t1=True)
        config["PlaneTracking"] = pt
        config["PGO"] = dict(pw["PGO"], **pw2.get("PGO", {}))
        config["PGO"]["mode"] = PW2_METHODS[args.method]
        config["Noise"] = cal["Noise"]
        config["Stiffness"] = dict(pw["Stiffness"], at_keyframes=True)
        return config
    if args.method in PW_METHODS:
        import yaml
        with open(E1C_CONFIG) as f:
            pt = yaml.safe_load(f)["PlaneTracking"]
        with open(PW_CONFIG) as f:
            pw = yaml.safe_load(f)
        with open(PW_CALIB) as f:
            cal = yaml.safe_load(f)
        pt.update(pw["PlaneTracking"])
        mm, rob, am, t1, sm, pg = PW_METHODS[args.method]
        pt.update(enable=True, pw=True, match_mode=mm, robust=rob, anchor_mode=am, kf_trigger_t1=t1, scale_mode=sm)
        config["PlaneTracking"] = pt
        config["PGO"] = dict(pw["PGO"], mode=pg)
        if int(getattr(args, "pgo_force_every", 0) or 0) > 0:
            config["PGO"]["force_every_kf"] = int(args.pgo_force_every)
        config["Noise"] = cal["Noise"]
        config["Stiffness"] = dict(pw["Stiffness"], at_keyframes=True)
        return config
    if args.method in ("e1c", "e1c_s"):
        import yaml
        with open(E1C_CONFIG) as f:
            pt = yaml.safe_load(f)["PlaneTracking"]
        pt["enable"] = True
        if args.method == "e1c_s":
            pt["lambda_plane"] = 0.1     # PLAN §5.3: depth-loss weight 1 - alpha = 0.1 (TUM) vs 1 in 2DGS-SLAM
        config["PlaneTracking"] = pt
    else:
        config["PlaneTracking"] = dict(enable=False)
    return config


def seed_process(seed, proc):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if os.environ.get("MONOGS_DETERMINISTIC", "0") == "1":   # PLAN §4.2 search for the remaining source (off by default)
        torch.use_deterministic_algorithms(True, warn_only=True)
    o3d_seed = False
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
        o3d_seed = True
    except Exception as e:  # noqa: BLE001
        Log(f"[Repro] open3d seed unavailable: {e}", tag="Repro")
    Log(f"[Repro] proc={proc} seed={seed} open3d_seed={o3d_seed} "
        f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32} tf32_cudnn={torch.backends.cudnn.allow_tf32}", tag="Repro")


def single_thread_flags(config):
    """(frontend value, backend value): frontend reads Training.single_thread; backend now reads the same key and
    falls back to Dataset.single_thread only when Training lacks it (PLAN §3.1)."""
    fe = config["Training"].get("single_thread", False)
    if "single_thread" in config["Training"]:
        be = config["Training"]["single_thread"]
    else:
        be = config["Dataset"].get("single_thread", False)
    return bool(fe), bool(be)


def apply_range(ds, config):
    rg = config.get("Repro", {}).get("range")
    if not rg:
        return
    b, e = int(rg[0]), int(rg[1])
    for name in ("color_paths", "depth_paths", "poses"):
        if hasattr(ds, name):
            setattr(ds, name, getattr(ds, name)[b:e])
    ds.num_imgs = len(ds.color_paths)
    Log(f"[Repro] range [{b}, {e}) -> {ds.num_imgs} frames", tag="Repro")


def _timestamp(ds, i):
    try:
        return float(os.path.splitext(os.path.basename(ds.color_paths[i]))[0])
    except Exception:  # noqa: BLE001
        return float(i)


def _c2w(R, T):
    M = np.eye(4)
    M[:3, :3] = R.detach().cpu().double().numpy()
    M[:3, 3] = T.detach().cpu().double().numpy()
    return np.linalg.inv(M), M


def _write_tum(path, rows):
    from scipy.spatial.transform import Rotation
    with open(path, "w") as f:
        for t, P in rows:
            U, _, Vt = np.linalg.svd(P[:3, :3])      # quaternion of the nearest rotation (MonoGS R may drift)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            q = Rotation.from_matrix(R).as_quat()
            f.write(f"{t:.6f} {P[0, 3]:.6f} {P[1, 3]:.6f} {P[2, 3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def write_outputs(frontend, ds, save_dir, total_s):
    os.makedirs(save_dir, exist_ok=True)
    ids = sorted(frontend.cameras.keys())
    est, gt, orth = [], [], []
    for i in ids:
        cam = frontend.cameras[i]
        Pe, W = _c2w(cam.R, cam.T)
        Pg, _ = _c2w(cam.R_gt, cam.T_gt)
        R = W[:3, :3]
        orth.append((float(np.linalg.norm(R @ R.T - np.eye(3))), float(np.linalg.det(R))))
        est.append((_timestamp(ds, i), Pe))
        gt.append((_timestamp(ds, i), Pg))
    kf = list(frontend.kf_indices)
    pos = {i: n for n, i in enumerate(ids)}
    _write_tum(os.path.join(save_dir, "traj_all.txt"), est)
    _write_tum(os.path.join(save_dir, "traj_gt_all.txt"), gt)
    _write_tum(os.path.join(save_dir, "traj_kf.txt"), [est[pos[i]] for i in kf if i in pos])
    _write_tum(os.path.join(save_dir, "traj_gt_kf.txt"), [gt[pos[i]] for i in kf if i in pos])
    # PLAN MONOGS-PW §3.4: corrected trajectory, every frame re-attached to the final pose of its reference keyframe
    corr = []
    ref = getattr(frontend, "ref_rel", {})
    kf_set = set(kf)
    for n, i in enumerate(ids):
        if i in kf_set or i not in ref or ref[i][0] not in frontend.cameras:
            corr.append(est[n])
            continue
        rk, Trel = ref[i]
        _, Wk = _c2w(frontend.cameras[rk].R, frontend.cameras[rk].T)
        corr.append((est[n][0], np.linalg.inv(Trel @ Wk)))
    _write_tum(os.path.join(save_dir, "traj_all_corr.txt"), corr)
    with open(os.path.join(save_dir, "kf_indices.txt"), "w") as f:
        f.write("\n".join(str(i) for i in kf) + "\n")
    tr = list(getattr(frontend, "track_ms", []))
    pl = list(getattr(frontend, "plane_ms", []))
    tim = dict(total_s=float(total_s), n_frames=len(ids), n_kf=len(kf), frame_offset=frontend.config.get("Repro", {}).get("frame_offset", 0),
               ms_track_mean=float(np.mean(tr)) if tr else None, ms_track_median=float(np.median(tr)) if tr else None,
               ms_plane_mean=float(np.mean(pl)) if pl else None, n_tracked=len(tr),
               max_orth_dev=max(o[0] for o in orth) if orth else None, max_abs_det_minus_1=max(abs(o[1] - 1) for o in orth) if orth else None,
               repro=frontend.config.get("Repro"))
    with open(os.path.join(save_dir, "timing.json"), "w") as f:
        json.dump(tim, f, indent=1)
    np.savetxt(os.path.join(save_dir, "ortho.csv"), np.array([[i, o[0], o[1]] for i, o in zip(ids, orth)]),
               delimiter=",", header="frame,orth_dev_fro,det", comments="", fmt=["%d", "%.6e", "%.8f"])
    Log(f"[Repro] wrote trajectories ({len(ids)} frames, {len(kf)} kf) and timing.json to {save_dir}", tag="Repro")
