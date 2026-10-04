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


def backend_inp(be, cur_camera, loop_camera=None):
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
        "cur_uid": int(cur_camera.uid), "loop_uid": None if loop_camera is None else int(loop_camera.uid),
    }


def factor_classes(pgo, all_cam_ids):
    """{index: 'prior' | 'odom' | 'loop'} for the non-null factors. Odometry factors join two frames that are
    consecutive in all_cam_ids (add_odom_node_to_graph); loop factors join (loop_uid, cur_uid)."""
    import gtsam

    nxt = {int(a): int(b) for a, b in zip(all_cam_ids[:-1], all_cam_ids[1:])}
    out = {}
    g = pgo.graph_factors
    for i in range(g.size()):
        f = g.at(i)
        if f is None:
            continue
        ks = [int(gtsam.Symbol(k).index()) for k in f.keys()]
        if len(ks) == 1:
            out[i] = "prior"
        elif len(ks) == 2 and nxt.get(ks[0]) == ks[1]:
            out[i] = "odom"
        else:
            out[i] = "loop"
    return out


def graph_error_by_class(pgo, classes, values):
    tot = {"prior": 0.0, "odom": 0.0, "loop": 0.0}
    g = pgo.graph_factors
    for i, c in classes.items():
        tot[c] += float(g.at(i).error(values))
    return tot


def regen_odometry(be, poses_new, anchored, tol_t=1e-6, tol_r=1e-6):
    """plan v5 A2: re-measure odometry factors that touch a frame whose written pose differs from the value the
    graph holds (> tol), with inv(T'_i) T'_j from the written c2w poses. Returns the log dict."""
    import gtsam
    import numpy as np

    g = be.pgo.graph_factors
    vals = be.pgo.graph_initials
    moved = set()
    T_new = {}
    for uid, P in poses_new.items():
        key = gtsam.symbol("x", int(uid))
        if not vals.exists(key):
            continue
        Pn = P.double().numpy()
        T_new[int(uid)] = Pn
        if int(uid) in anchored:
            continue
        held = vals.atPose3(key).matrix()
        d = np.linalg.inv(held) @ Pn
        ang = float(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
        if float(np.linalg.norm(d[:3, 3])) > tol_t or ang > tol_r:
            moved.add(int(uid))
    classes = factor_classes(be.pgo, be.all_cam_ids)
    n_rep, max_dm, max_da = 0, 0.0, 0.0
    for i, c in classes.items():
        if c != "odom":
            continue
        f = g.at(i)
        ki, kj = [int(gtsam.Symbol(k).index()) for k in f.keys()]
        if ki not in moved and kj not in moved:
            continue
        if ki not in T_new or kj not in T_new:
            continue
        meas = np.linalg.inv(T_new[ki]) @ T_new[kj]
        old = f.measured().matrix()
        dm = np.linalg.inv(old) @ meas
        max_dm = max(max_dm, float(np.linalg.norm(dm[:3, 3])))
        max_da = max(max_da, float(np.arccos(np.clip((np.trace(dm[:3, :3]) - 1) / 2, -1, 1))))
        g.replace(i, gtsam.BetweenFactorPose3(f.keys()[0], f.keys()[1], gtsam.Pose3(meas), f.noiseModel()))
        n_rep += 1
    return {"regen_n_moved": len(moved), "regen_n_replaced": n_rep, "regen_n_odom": sum(1 for c in classes.values() if c == "odom"),
            "regen_n_loop": sum(1 for c in classes.values() if c == "loop"), "regen_max_dmeas_m": max_dm,
            "regen_max_dmeas_rad": max_da}, classes


def write_back(be, res, regen_odom=False):
    """Gaussians (xyz, rotation) via replace_tensors_in_optimizer; poses into cameras and gtsam.
    plan v5 A2 (regen_odom): odometry factors are re-measured from the written poses BEFORE the values are
    updated, so the reported error after the write is that of the consistent graph."""
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
    extra = {}
    classes = None
    if regen_odom:
        rlog, classes = regen_odometry(be, res["poses"], anchored)
        extra.update(rlog)
        extra["gtsam_err_before_sync_regen_graph"] = float(be.pgo.graph_factors.error(be.pgo.graph_initials))
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
    if classes is None:
        classes = factor_classes(be.pgo, be.all_cam_ids)
    extra["gtsam_err_after_sync_by_class"] = graph_error_by_class(be.pgo, classes, be.pgo.graph_initials)
    # residual of the loop factor of THIS event (the newest loop factor in the graph), after the write
    loop_idx = [i for i, c in classes.items() if c == "loop"]
    if loop_idx:
        extra["loop_factor_k_error_after_sync"] = float(be.pgo.graph_factors.at(max(loop_idx)).error(be.pgo.graph_initials))
    return {"gtsam_err_before_sync": err_before, "gtsam_err_after_sync": float(be.pgo.last_error),
            "anchored_uids": sorted(anchored), "anchor_resid_m": resid, **extra}


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
        inp = backend_inp(be, cur_camera, loop_camera)
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
            log.update(write_back(be, res, regen_odom=bool(dcfg_resolved["pose_sync"].get("regen_odom", False))))
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
