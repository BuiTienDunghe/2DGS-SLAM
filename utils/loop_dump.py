"""Instrumentation for loop events (handoff plan P0, Appendix B/C).

Everything here only *reads* backend state, except that the caller decides when to apply
the correction. Dumps are plain dicts of CPU tensors written with torch.save.
"""
import json
import os
import time

import numpy as np
import torch
import yaml


VALID_MODES = ("rigid", "deform")


def load_deform_config(path):
    """Deformation config; None means rigid (= baseline)."""
    if path is None:
        return {"mode": "rigid", "variant": None, "_path": None}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    mode = cfg.get("mode", "rigid")
    if mode not in VALID_MODES:
        raise ValueError(f"deform config {path}: mode must be one of {VALID_MODES}, got {mode!r}")
    if mode == "rigid":
        cfg["variant"] = None
    elif cfg.get("variant") not in ("A", "B"):
        raise ValueError(f"deform config {path}: variant must be 'A' or 'B'")
    cfg["mode"] = mode
    cfg["_path"] = str(path)
    return cfg


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            return x.item()
        except Exception:
            return str(x)
    return x


def write_json(path, obj):
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def append_jsonl(run_dir, name, obj):
    if not run_dir:
        return
    with open(os.path.join(run_dir, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(obj)) + "\n")


def gaussian_state(g):
    """Per-Gaussian tensors on CPU (activated scale/opacity, normalized quaternion w,x,y,z)."""
    with torch.no_grad():
        return {
            "xyz": g.get_xyz.detach().float().cpu().clone(),
            "rot": g.get_rotation.detach().float().cpu().clone(),
            "scale": g.get_scaling.detach().float().cpu().clone(),
            "opacity": g.get_opacity.detach().float().cpu().clone(),
            "f_dc": g._features_dc.detach().float().cpu().clone(),
            "t0": g.birth_kfIDs.detach().reshape(-1).int().cpu().clone(),
            "tc": g.unique_kfIDs.detach().reshape(-1).int().cpu().clone(),
            "tl": g.last_observe_ids.detach().reshape(-1).int().cpu().clone(),
            "dc": g.min_observed_depth.detach().reshape(-1).float().cpu().clone(),
            "active": g.active_mask.detach().reshape(-1).bool().cpu().clone(),
            # plan v4: raw parameters (exp / sigmoid / normalize give the activated values above bit-exactly)
            "scale_raw": g._scaling.detach().float().cpu().clone(),
            "opacity_raw": g._opacity.detach().float().cpu().clone(),
            "rot_raw": g._rotation.detach().float().cpu().clone(),
        }


def c2w(cam):
    return torch.linalg.inv(cam.T.detach().double()).cpu()


def gt_c2w(cam):
    if cam.gt_pose is None:
        return None
    return torch.linalg.inv(cam.gt_pose.detach().double()).cpu()


def cam_meta(cam):
    return {
        "scale": float(cam.scale.detach().reshape(-1)[0].item()),
        "shift": float(cam.shift.detach().reshape(-1)[0].item()),
        "exposure_a": float(cam.exposure_a.detach().reshape(-1)[0].item()),
        "exposure_b": float(cam.exposure_b.detach().reshape(-1)[0].item()),
    }


def intrinsics(cam):
    return {
        "fx": float(cam.fx), "fy": float(cam.fy), "cx": float(cam.cx), "cy": float(cam.cy),
        "W": int(cam.image_width), "H": int(cam.image_height),
    }


def snapshot_pre(backend, cur_camera, loop_camera):
    """State right after PGO succeeded and BEFORE the map is corrected / reactivated."""
    poses_pgo = {}
    for uid in backend.all_cam_ids:
        poses_pgo[int(uid)] = torch.from_numpy(backend.pgo.get_optimized_node_pose(uid)).double()
    return {
        "gauss_pre": gaussian_state(backend.gaussians),
        "poses_pre": {int(u): c2w(c) for u, c in backend.all_cameras.items()},
        # the rigid fix reads key_cameras[uid].T; keep it separately (may differ from all_cameras[uid])
        "poses_pre_kf": {int(u): c2w(c) for u, c in backend.key_cameras.items()},
        "kf_cam_is_all_cam": {int(u): (backend.all_cameras.get(u) is c) for u, c in backend.key_cameras.items()},
        "poses_pgo": poses_pgo,
        "poses_gt": {int(u): gt_c2w(c) for u, c in backend.all_cameras.items()},
        "keyframe_uids": sorted(int(u) for u in backend.key_cameras.keys()),
        "all_cam_ids": [int(u) for u in backend.all_cam_ids],
        "cam_meta": {int(u): cam_meta(c) for u, c in backend.key_cameras.items()},
        "intrinsics": intrinsics(cur_camera),
        "active_cam_id_list": [int(u) for u in backend.active_cam_id_list],
        "inactive_cam_id_list": [int(u) for u in backend.inactive_cam_id_list],
        "cam_sliding_window": [int(u) for u in backend.cam_sliding_window],
        "anchor_uids": _anchor_uids(backend),
        # plan v5: what the offline PGO replay needs (relocalised loop camera, measurement, all factors)
        "loop_cam_pose": c2w(loop_camera),
        "loop_transform": torch.from_numpy(np.linalg.inv(c2w(loop_camera).numpy()) @ c2w(cur_camera).numpy()),
        "graph_factors": _factor_list(backend),
    }


def _factor_list(backend):
    """[(type, [uids], measured 4x4 (list), sigmas (list))] for every non-null factor, in graph order."""
    try:
        import gtsam
        g = backend.pgo.graph_factors
        out = []
        for i in range(g.size()):
            f = g.at(i)
            if f is None:
                out.append(None)
                continue
            ks = [int(gtsam.Symbol(k).index()) for k in f.keys()]
            t = type(f).__name__
            meas = f.prior().matrix() if t.startswith("PriorFactor") else f.measured().matrix()
            out.append((t, ks, meas.tolist(), f.noiseModel().sigmas().tolist()))
        return out
    except Exception as e:
        return [("error", str(e))]


def _anchor_uids(backend):
    """Frames pinned by a unary prior factor (plan v4 A4); [] if that cannot be determined."""
    try:
        from deform.online import anchored_uids
        return anchored_uids(backend.pgo)
    except Exception:
        return []


def event_record(backend, cur_camera, loop_camera, pgo_err_before, t_fix_s, deform_log=None):
    """One loop_events.jsonl line (Appendix C). Deformation fields stay None in rigid mode."""
    cfg = backend.deform_cfg
    rec = {
        "event_id": backend.loop_event_count,
        "frame_idx": int(cur_camera.uid),
        "cur_uid": int(cur_camera.uid),
        "loop_uid": int(loop_camera.uid),
        "mode": cfg.get("mode", "rigid"),
        "variant": cfg.get("variant"),
        "accepted": None,
        "fallback_reason": None,
        "pgo_err_before": pgo_err_before,
        "pgo_err_after": float(backend.pgo.last_error),
        "gtsam_err_after_sync": float(backend.pgo.graph_factors.error(backend.pgo.graph_initials)),
        "n_gauss": int(backend.gaussians.get_xyz.shape[0]),
        "n_keyframes": len(backend.key_cameras),
        "n_frames": len(backend.all_cameras),
        "t_stage_s": {"total": float(t_fix_s)},
        "t_event_s": float(t_fix_s),
        "wall_time": time.time(),
    }
    if deform_log:
        rec.update(deform_log)
    return rec


def save_revisit_dump(backend, data):
    """plan v6 quick (E5): map + poses at a revisit burst (no PGO, no correction), written between two mapping
    iterations. Same layout as a loop-event dump (gauss_pre, poses_pre = poses_pgo, meta.cur_uid = the requesting
    frame, meta.loop_uid = the candidate keyframe) so that deform.dump.EventState / inp_from_dump read it; the tracked
    pose of the requesting frame (it need not be a keyframe) is in dump["request"]."""
    if not backend.run_dir or not backend.key_cameras:
        return None
    t0 = time.perf_counter()
    _, uid, tag, pose_c2w, gt, info = data
    d = os.path.join(backend.run_dir, "revisit_dumps")
    os.makedirs(d, exist_ok=True)
    cfg = backend.config
    n = getattr(backend, "revisit_dump_count", 0) + 1
    backend.revisit_dump_count = n
    cand = info.get("cand_kf")
    kf_uids = sorted(int(u) for u in backend.key_cameras.keys())
    poses = {int(u): c2w(c) for u, c in backend.all_cameras.items()}
    dump = {
        "gauss_pre": gaussian_state(backend.gaussians),
        "poses_pre": poses,
        "poses_pgo": poses,
        "poses_pre_kf": {int(u): c2w(c) for u, c in backend.key_cameras.items()},
        "poses_gt": {int(u): gt_c2w(c) for u, c in backend.all_cameras.items()},
        "keyframe_uids": kf_uids,
        "all_cam_ids": [int(u) for u in backend.all_cam_ids],
        "cam_meta": {int(u): cam_meta(c) for u, c in backend.key_cameras.items()},
        "intrinsics": intrinsics(backend.key_cameras[kf_uids[-1]]),
        "active_cam_id_list": [int(u) for u in backend.active_cam_id_list],
        "inactive_cam_id_list": [int(u) for u in backend.inactive_cam_id_list],
        "cam_sliding_window": [int(u) for u in backend.cam_sliding_window],
        "anchor_uids": _anchor_uids(backend),
        "request": {"uid": int(uid), "tag": tag, "pose_c2w": torch.as_tensor(pose_c2w).double(),
                    "gt_c2w": None if gt is None else torch.as_tensor(gt).double(), "cand_kf": cand,
                    "observed_ratio": info.get("observed_ratio"), "latest_kf_uid": kf_uids[-1]},
        "meta": {"event_id": n, "frame_idx": int(uid), "cur_uid": int(uid), "loop_uid": -1 if cand is None else int(cand),
                 "scene": f'{cfg["Dataset"]["type"]}/{cfg["Dataset"]["sequence_name"]}', "seed": backend.seed,
                 "mode": "revisit", "variant": None, "wall_time": time.time(),
                 "old_than_N_keyframe": int(backend.old_than_N_keyframe)},
        "config": cfg,
    }
    path = os.path.join(d, f'rv_{n:03d}_{tag}_f{int(uid)}_k{dump["meta"]["loop_uid"]}.pt')
    torch.save(dump, path)
    dt = time.perf_counter() - t0
    backend.revisit_dump_time_s = getattr(backend, "revisit_dump_time_s", 0.0) + dt
    append_jsonl(backend.run_dir, "revisit_dumps.jsonl", {
        "n": n, "uid": int(uid), "tag": tag, "cand_kf": cand, "observed_ratio": info.get("observed_ratio"),
        "latest_kf_uid": kf_uids[-1], "n_gauss": int(dump["gauss_pre"]["xyz"].shape[0]), "t_dump_s": dt,
        "t_total_s": backend.revisit_dump_time_s, "file": os.path.basename(path), "wall_time": time.time()})
    return path


def save_dump(backend, pre, event):
    """Write pre + post (after correction, BEFORE update_state) to loop_dumps/."""
    if not backend.run_dir:
        return None
    d = os.path.join(backend.run_dir, "loop_dumps")
    os.makedirs(d, exist_ok=True)
    g = backend.gaussians
    with torch.no_grad():
        post = {
            "xyz": g.get_xyz.detach().float().cpu().clone(),
            "rot": g.get_rotation.detach().float().cpu().clone(),
        }
    cfg = backend.config
    dump = dict(pre)
    dump["gauss_post"] = post
    dump["poses_final"] = {int(u): c2w(c) for u, c in backend.all_cameras.items()}
    dump["meta"] = {
        "event_id": event["event_id"],
        "frame_idx": event["frame_idx"],
        "cur_uid": event["cur_uid"],
        "loop_uid": event["loop_uid"],
        "scene": f'{cfg["Dataset"]["type"]}/{cfg["Dataset"]["sequence_name"]}',
        "seed": backend.seed,
        "mode": event["mode"],
        "variant": event["variant"],
        "wall_time": event["wall_time"],
        "pgo_err_before": event["pgo_err_before"],
        "pgo_err_after": event["pgo_err_after"],
        "config_path": backend.deform_cfg.get("_path"),
        "old_than_N_keyframe": int(backend.old_than_N_keyframe),
    }
    dump["config"] = cfg
    dump["deform_log"] = {k: v for k, v in event.items() if k not in dump["meta"]}
    path = os.path.join(
        d, f'event_{event["event_id"]:03d}_c{event["cur_uid"]}_h{event["loop_uid"]}.pt')
    torch.save(dump, path)
    return path
