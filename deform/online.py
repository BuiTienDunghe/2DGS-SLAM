"""Backend hook (plan P4): run correct_map() on the live backend state and write the result back.

Called in the "pgo" branch right after optimize_pose_graph() returned True and BEFORE update_state().
On rejection the caller falls back to BackEnd.apply_rigid_correction() (exact baseline behaviour).
"""
import time

import torch

from deform.pipeline import correct_map


def anchored_uids(pgo):
    """Frames pinned by a unary factor (the fixed prior on the first frame, slam_backend.py:682)."""
    import gtsam

    out = set()
    g = pgo.graph_factors
    for i in range(g.size()):
        try:
            f = g.at(i)
        except Exception:
            continue
        if f is None:
            continue
        ks = f.keys()
        if len(ks) == 1:
            out.add(int(gtsam.Symbol(ks[0]).index()))
    return sorted(out)


def backend_inp(be, cur_camera):
    g = be.gaussians
    with torch.no_grad():
        G = {"xyz": g.get_xyz.detach().float().clone(), "rot": g.get_rotation.detach().float().clone(),
             "scale": g.get_scaling.detach().float().clone(), "opacity": g.get_opacity.detach().float().clone(),
             "f_dc": g._features_dc.detach().float().clone()}
    poses_pre = {int(u): torch.linalg.inv(c.T.detach().double()).cpu() for u, c in be.all_cameras.items()}
    poses_pgo = {int(u): torch.from_numpy(be.pgo.get_optimized_node_pose(u)).double() for u in be.all_cam_ids}

    def frame_fn(uid):
        cam, fr = be.key_cameras[uid], be.key_frames[uid]
        with torch.no_grad():
            depth = cam.scale.detach() * fr.depth + cam.shift.detach()
        return fr.rgb, depth

    return {
        "g": G,
        "t0": g.birth_kfIDs.detach().reshape(-1).long(),
        "tc": g.unique_kfIDs.detach().reshape(-1).long(),
        "active": g.active_mask.detach().reshape(-1).bool().clone(),
        "kf_uids": sorted(int(u) for u in be.key_cameras.keys()),
        "all_cam_ids": [int(u) for u in be.all_cam_ids],
        "poses_pre": poses_pre, "poses_pgo": poses_pgo,
        "intr": {"fx": float(cur_camera.fx), "fy": float(cur_camera.fy), "cx": float(cur_camera.cx),
                 "cy": float(cur_camera.cy), "W": int(cur_camera.image_width), "H": int(cur_camera.image_height)},
        "frame_fn": frame_fn, "tr": be.config["Training"], "W": int(be.old_than_N_keyframe), "seed": int(be.seed),
        "anchor_uids": anchored_uids(be.pgo),
    }


def write_back(be, res):
    """Gaussians (xyz, rotation) via replace_tensors_in_optimizer; poses into cameras and gtsam."""
    import gtsam

    g = be.gaussians
    opt = g.replace_tensors_in_optimizer({"xyz": res["xyz"].detach().float().contiguous(),
                                          "rotation": res["rot"].detach().float().contiguous()})
    g._xyz = opt["xyz"]
    g._rotation = opt["rotation"]
    err_before = float(be.pgo.graph_factors.error(be.pgo.graph_initials))
    # plan v4 A4: frames pinned by a prior keep the value gtsam holds (the pipeline anchored the result on them,
    # so the camera pose written below agrees with it to float precision); report the residual
    anchored = set(anchored_uids(be.pgo))
    resid = {}
    for uid, P in res["poses"].items():
        w2c = torch.linalg.inv(P.double()).float().to(be.device)
        if uid in be.all_cameras:
            be.all_cameras[uid].T = w2c
        if uid in be.key_cameras:
            be.key_cameras[uid].T = w2c
        key = gtsam.symbol("x", int(uid))
        if not be.pgo.graph_initials.exists(key):
            continue
        if int(uid) in anchored:
            held = be.pgo.graph_initials.atPose3(key).matrix()
            resid[int(uid)] = float(((P.double().numpy() - held)[:3, 3] ** 2).sum() ** 0.5)
            continue
        be.pgo.graph_initials.update(key, gtsam.Pose3(P.double().numpy()))
    be.pgo.last_error = be.pgo.graph_factors.error(be.pgo.graph_initials)
    return {"gtsam_err_before_sync": err_before, "gtsam_err_after_sync": float(be.pgo.last_error),
            "anchored_uids": sorted(anchored), "anchor_resid_m": resid}


def backend_correct(be, cur_camera, loop_camera, dcfg_resolved):
    """Returns (applied: bool, deform_log dict). If not applied the caller must run the rigid fallback."""
    t0 = time.perf_counter()
    # plan v4 A5: peak VRAM of this event. reset_peak_memory_stats would erase the run-level peak that
    # resources_backend.json reads at the end, so the running maximum is kept here and re-reported.
    vram_before = torch.cuda.memory_allocated()
    run_peak = max(getattr(be, "_vram_run_peak", 0), torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    # The backend inherits allow_tf32=True from dust3r/croco; TF32 matmuls lose ~1 mm on room-scale coordinates,
    # the same order as sigma_c. Run the method in full fp32 (offline tools do too) and restore the flag after.
    tf32_prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        inp = backend_inp(be, cur_camera)
        res = correct_map(inp, dcfg_resolved, dcfg_resolved["variant"])
    except Exception as e:  # any failure -> rigid fallback (never crash the SLAM run)
        import traceback
        torch.backends.cuda.matmul.allow_tf32 = tf32_prev
        log = {"accepted": False, "fallback_reason": f"exception: {type(e).__name__}: {e}",
               "traceback": traceback.format_exc()[-2000:]}
        _vram_log(log, be, vram_before, run_peak)
        return False, log
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_prev
    log = dict(res["log"])
    if res["accepted"]:
        # snapshot everything write_back mutates; on any failure restore it so the caller's rigid fallback
        # starts from the exact pre-correction state (plan: safe fallback in P4 is non-skippable)
        import gtsam
        g = be.gaussians
        snap_xyz, snap_rot = g._xyz.detach().clone(), g._rotation.detach().clone()
        snap_T = {u: c.T for u, c in be.all_cameras.items()}
        snap_T.update({u: c.T for u, c in be.key_cameras.items()})
        snap_vals = gtsam.Values(be.pgo.graph_initials)
        snap_err = be.pgo.last_error
        try:
            log.update(write_back(be, res))
        except Exception as e:
            import traceback
            opt = g.replace_tensors_in_optimizer({"xyz": snap_xyz, "rotation": snap_rot})
            g._xyz, g._rotation = opt["xyz"], opt["rotation"]
            for u, T in snap_T.items():
                if u in be.all_cameras:
                    be.all_cameras[u].T = T
                if u in be.key_cameras:
                    be.key_cameras[u].T = T
            # after optimize_pose_graph graph_initials IS graph_optimized (same object) -> restore both
            be.pgo.graph_initials = snap_vals
            be.pgo.graph_optimized = snap_vals
            be.pgo.last_error = snap_err
            log.update({"accepted": False, "fallback_reason": f"exception_writeback: {type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-2000:]})
            log["t_hook_s"] = time.perf_counter() - t0
            _vram_log(log, be, vram_before, run_peak)
            return False, log
    log["t_hook_s"] = time.perf_counter() - t0
    del inp
    torch.cuda.empty_cache()
    _vram_log(log, be, vram_before, run_peak)
    return bool(res["accepted"]), log


def _vram_log(log, be, vram_before, run_peak):
    """Per-event peak (plan v4 A5). The torch peak counter was reset for this event, so the run-level peak is
    carried on the backend object; BackEnd._write_backend_resources reads it back (log only)."""
    ev_peak = torch.cuda.max_memory_allocated()
    be._vram_run_peak = max(run_peak, ev_peak)
    log["vram_event_peak_gb"] = ev_peak / 1024 ** 3
    log["vram_before_gb"] = vram_before / 1024 ** 3
    log["vram_run_peak_gb"] = be._vram_run_peak / 1024 ** 3
