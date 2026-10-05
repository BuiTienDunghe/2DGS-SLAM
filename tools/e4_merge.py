"""plan v6 quick, E4: merge the duplicated Gaussians of the two layers after the deformation (offline, no
re-optimisation).

usage: python tools/e4_merge.py --out DIR RUN_DIR [RUN_DIR ...]      (runs with deformation dumps, e.g. deform6_s0/s1)
Per accepted loop event: the map AFTER the deformation (gauss_post, poses_final of the dump), layers by birth time
(old: t0 < s_k, new: t0 >= s_k), Pi*(k) as in v5 (rigid replay of the dump).
Region: Gaussians whose centre projects, in at least one J_eval keyframe, onto a pixel where both layers are
rendered (alpha > 0,95), agree within eps_eval and in normal, away from depth edges (the dense Pi* gating), and
lies within 5 cm of its own layer's rendered depth there.
Rule: every new-layer Gaussian of the region is paired with the nearest old-layer Gaussian of the region; the pair
is kept when |n_old . (x_new - x_old)| < tau_d (5 and 10 mm), the normals differ by < 30 deg and the tangential
distance is below the larger of the two largest scales. A Gaussian is in at most one pair (pairs visited by
increasing normal distance). Two ways to merge:
  A  keep the member with the higher reliability S (S as in the pipeline, on this state), delete the other
  B  one Gaussian replaces the pair: opacity-weighted position, normal and tangential scales, opacity = max,
     rotation = tangent axis of the more reliable member projected on the new plane, colour and everything else of
     the more reliable member, birth keyframe of the old-layer member
Deletion goes through GaussianModel.prune_points (the upstream routine) on a model rebuilt from the dump.
Before / after: Gaussians removed (region, whole map), PSNR and depth L1 on the J_eval keyframes and on every 5th
keyframe, share of Pi* pixels that still show two layers and the median gap on them.
Writes DIR/e4.jsonl.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402


# ------------------------------------------------------------------ pure part (numpy, tested on CPU)
def quat_to_R(q):
    """(w,x,y,z) -> rotation matrices [N,3,3]."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], 1)).as_matrix()


def R_to_quat(R):
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(R).as_quat()
    q = np.concatenate([q[:, 3:], q[:, :3]], 1)
    return np.where(q[:, :1] < 0, -q, q)


def match_pairs(x_old, n_old, smax_old, x_new, n_new, smax_new, tau_d, max_angle_deg=30.0, funnel=None):
    """Indices (into the new / old arrays) of the accepted pairs and their normal distance."""
    from scipy.spatial import cKDTree

    e = np.zeros(0, dtype=np.int64)
    if len(x_old) == 0 or len(x_new) == 0:
        return e, e, np.zeros(0)
    dist, j = cKDTree(x_old).query(x_new, k=1)
    d = x_new - x_old[j]
    dn = np.abs(np.einsum("nd,nd->n", n_old[j], d))
    dt = np.sqrt(np.clip(dist ** 2 - dn ** 2, 0, None))
    ok_d = dn < tau_d
    ok_n = np.einsum("nd,nd->n", n_old[j], n_new) > np.cos(np.radians(max_angle_deg))
    ok_t = dt < np.maximum(smax_old[j], smax_new)
    ok = ok_d & ok_n & ok_t
    i = np.nonzero(ok)[0]
    i = i[np.argsort(dn[i], kind="stable")]
    _, first = np.unique(j[i], return_index=True)  # the closest (in normal distance) new Gaussian of every old one
    n_ok = len(i)
    i = i[np.sort(first)]
    if funnel is not None:  # how many new Gaussians each gate lets through (gates applied one after the other)
        funnel.update({"new": int(len(x_new)), "normal_dist": int(ok_d.sum()), "normal_angle": int((ok_d & ok_n).sum()), "tangential": int(n_ok),
                       "one_per_old": int(len(i)), "nn_dist_med_mm": 1e3 * float(np.median(dist)), "normal_dist_med_mm": 1e3 * float(np.median(dn)),
                       "tangential_med_mm": 1e3 * float(np.median(dt))})
    return i, j[i], dn[i]


def merge_b(xyz, quat, scale, opacity, n_or, i_keep, i_del):
    """Mode B values for the kept slots: (xyz, quat (w,x,y,z), scale, opacity). i_keep = the more reliable member."""
    w1, w2 = opacity[i_keep], opacity[i_del]
    ws = np.clip(w1 + w2, 1e-12, None)
    x = (w1[:, None] * xyz[i_keep] + w2[:, None] * xyz[i_del]) / ws[:, None]
    n = w1[:, None] * n_or[i_keep] + w2[:, None] * n_or[i_del]
    n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)
    t1 = quat_to_R(quat[i_keep])[:, :, 0]
    t1 = t1 - np.einsum("nd,nd->n", t1, n)[:, None] * n
    t1 /= np.clip(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12, None)
    R = np.stack([t1, np.cross(n, t1), n], axis=-1)
    s = (w1[:, None] * scale[i_keep] + w2[:, None] * scale[i_del]) / ws[:, None]
    return x, R_to_quat(R), s, np.maximum(w1, w2)


def oriented_normals(xyz, quat, t0, centers):
    """Third rotation axis, turned towards the camera centre of the birth keyframe (centers: {uid: [3]})."""
    n = quat_to_R(quat)[:, :, 2]
    c = np.stack([centers.get(int(u), np.full(3, np.nan)) for u in t0])
    s = np.sign(np.einsum("nd,nd->n", n, c - xyz))
    s[~np.isfinite(s) | (s == 0)] = 1.0
    return n * s[:, None]


# ------------------------------------------------------------------ GPU part
def model_from_state(g, opt):
    """GaussianModel (upstream class) rebuilt from the raw parameters of a dump, with its optimizer."""
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    from gaussian_splatting.scene.gaussian_model import GaussianModel

    m = GaussianModel(sh_degree=0)
    par = lambda t: nn.Parameter(t.float().to("cuda").contiguous().requires_grad_(True))  # noqa: E731
    col = lambda t, dt: t.reshape(-1, 1).to("cuda").to(dt)  # noqa: E731
    N = g["xyz"].shape[0]
    m._xyz, m._features_dc = par(g["xyz"]), par(g["f_dc"])
    m._features_rest = par(torch.zeros((N, 0, 3)))
    m._opacity, m._scaling, m._rotation = par(g["opacity_raw"]), par(g["scale_raw"]), par(g["rot_raw"])
    m.unique_kfIDs, m.last_observe_ids, m.birth_kfIDs = col(g["tc"], torch.int), col(g["tl"], torch.int), col(g["t0"], torch.int)
    m.min_observed_depth, m.active_mask = col(g["dc"], torch.float32), col(g["active"], torch.bool)
    m.training_setup(SimpleNamespace(**{k: opt[k] for k in ("position_lr", "feature_lr", "opacity_lr", "scaling_lr", "rotation_lr", "percent_dense")}))
    return m


def main():
    import deform  # noqa: F401
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from deform import config as dcfg
    from deform import metrics as M
    from deform.dump import EventState, list_dumps, load_dump
    from deform.pipeline import delta_T_dict, delta_t_frames, gauss_from_dump, inp_from_dump, rigid_result, with_pos
    from deform.reliability import compute_reliability
    from deform.render_utils import DEV, det_scope, make_cam
    from e_v4_durability import measure
    from utils import loop_dump
    from utils.loop_dump import load_deform_config

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(root, "configs", "deform", "selected_v6.yaml"))
    ap.add_argument("--taus", type=float, nargs="+", default=[0.005, 0.010])
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--only-event", type=int, default=None)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    user = load_deform_config(a.config)
    out_path = os.path.join(a.out, "e4.jsonl")
    open(out_path, "w").close()

    def region_gaussians(st, G, poses, uids, m_old, m_new, cfg, tol=0.05):
        c = cfg["corr"]
        Hh, Wd = int(st.intr["H"]), int(st.intr["W"])
        inreg = torch.zeros(G["xyz"].shape[0], dtype=torch.bool, device=DEV)
        for u in uids:
            L = M.render_layers(st.cam(u, poses=poses), G, m_old, m_new)
            m = (L["Oo"] > M.ALPHA_THR) & (L["On"] > M.ALPHA_THR) & ((L["Dn"] - L["Do"]).abs() < c["eps_eval"])
            m &= (L["No"] * L["Nn"]).sum(0) > c["cos_theta_n"]
            m &= ~(M.edge_mask(L["Do"], c["g_max"]) | M.edge_mask(L["Dn"], c["g_max"]))
            w2c = torch.linalg.inv(torch.as_tensor(poses[u], dtype=torch.float64)).float().to(DEV)
            pc = G["xyz"] @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            px = torch.round(st.intr["fx"] * pc[:, 0] / z.clamp_min(1e-6) + st.intr["cx"]).long()
            py = torch.round(st.intr["fy"] * pc[:, 1] / z.clamp_min(1e-6) + st.intr["cy"]).long()
            idx = torch.nonzero((z > 0.05) & (px >= 0) & (px < Wd) & (py >= 0) & (py < Hh)).reshape(-1)
            D = torch.where(m_new[idx], L["Dn"][py[idx], px[idx]], L["Do"][py[idx], px[idx]])
            vis = m[py[idx], px[idx]] & ((z[idx] - D).abs() < tol)
            inreg.index_fill_(0, idx[vis], True)
        return inreg

    for rd in a.runs:
        for p in list_dumps(rd):
            d = load_dump(p)
            st = EventState(d)
            C, Lu = int(st.meta["cur_uid"]), int(st.meta["loop_uid"])
            if a.only_event is not None and C != a.only_event:
                continue
            acc = (d.get("deform_log") or {}).get("accepted")
            row = {"run": os.path.basename(rd), "event": f"{C}<->{Lu}", "mode": st.meta.get("mode"), "deform_accepted": acc}
            if st.meta.get("mode") != "deform" or not acc:
                row["note"] = "event without an accepted deformation: skipped"
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                continue
            cfg = dcfg.resolve(user, st.config)
            inp = inp_from_dump(d, st.frame)
            kf_uids = sorted(inp["kf_uids"])
            split = M.birth_split(C, Lu)
            Hh, Wd = int(st.intr["H"]), int(st.intr["W"])
            with torch.no_grad(), det_scope(True):
                # Pi*(k) from the rigid replay, as everywhere in v5
                dT = delta_T_dict(kf_uids, inp["all_cam_ids"], inp["poses_pre"], inp["poses_pgo"])
                xyz_rig, rot_rig, poses_rig = rigid_result(inp, dT)
                G_rig = with_pos(inp["g"], xyz_rig, rot_rig)
                mo, mn = M.layer_masks_birth(inp["t0"], split)
                JL, _ = M.select_loop_kfs(st, G_rig, poses_rig, mo, mn, cfg)
                if not JL:
                    row["note"] = "no two-layer keyframe"
                    with open(out_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    continue
                _, J_eval = M.split_opt_eval(JL, Hh, Wd, cfg["corr"]["checker"])
                J_eval_u = sorted(u for u, _ in J_eval)
                Pi, _ = M.pi_star(st, G_rig, poses_rig, J_eval_u, mo, mn, cfg)
                row["n_pi"] = int(sum(int(m.sum()) for m, _ in Pi.values()))
                del G_rig
                # the map after the deformation
                gs = dict(d["gauss_pre"])
                gs["xyz"], gs["rot"], gs["rot_raw"] = d["gauss_post"]["xyz"], d["gauss_post"]["rot"], d["gauss_post"]["rot"]
                poses = {int(u): torch.as_tensor(v).double() for u, v in d["poses_final"].items()}
                G0 = gauss_from_dump(gs)
                t0_0 = gs["t0"].reshape(-1).long()
                N0 = int(t0_0.shape[0])
                all_u = kf_uids[:: a.every]

                def metrics(G, t0):
                    masks = M.layer_masks_birth(t0, split)
                    r, valid = measure(st, G, poses, Pi, masks)
                    qe, qa = M.render_quality(st, G, poses, J_eval_u), M.render_quality(st, G, poses, all_u)
                    return {"n_gauss": int(G["xyz"].shape[0]), "two_layer_frac": r["valid_frac"], "gap_mm": r["median_mm"], "gap_n": r["n"],
                            "psnr_eval": qe["psnr"], "l1_eval_mm": None if qe["depth_l1"] is None else 1e3 * qe["depth_l1"],
                            "psnr_all": qa["psnr"], "l1_all_mm": None if qa["depth_l1"] is None else 1e3 * qa["depth_l1"]}, valid, masks

                base, valid0, masks0 = metrics(G0, t0_0)
                row["before"] = base
                dt = delta_t_frames(kf_uids, inp["W"])
                cams = {u: make_cam(u, poses[u], inp["intr"]) for u in kf_uids}
                rel = compute_reliability(G0, cams, st.frame, t0_0.to(DEV), dt, cfg, inp["tr"], kf_uids)
                del cams
                S = rel["S"].reshape(-1).double().cpu().numpy()
                inreg = region_gaussians(st, G0, poses, J_eval_u, masks0[0], masks0[1], cfg)
                io = torch.nonzero(inreg & masks0[0]).reshape(-1).cpu().numpy()
                inw = torch.nonzero(inreg & masks0[1]).reshape(-1).cpu().numpy()
                row.update({"n_region_old": int(len(io)), "n_region_new": int(len(inw)), "J_eval": J_eval_u})
                xyz = G0["xyz"].double().cpu().numpy()
                quat = G0["rot"].double().cpu().numpy()
                scale = G0["scale"].double().cpu().numpy()
                opac = G0["opacity"].reshape(-1).double().cpu().numpy()
                centers = {int(u): np.asarray(P, dtype=np.float64)[:3, 3] for u, P in poses.items()}
                n_or = oriented_normals(xyz, quat, t0_0.numpy(), centers)
                smax = scale.max(1)
                row["variants"] = {}
                for tau in a.taus:
                    fun = {}
                    i_n, i_o, dn = match_pairs(xyz[io], n_or[io], smax[io], xyz[inw], n_or[inw], smax[inw], tau, funnel=fun)
                    g_new, g_old = inw[i_n], io[i_o]
                    keep_old = S[g_old] >= S[g_new]  # the more reliable member survives (ties: the old one)
                    i_keep, i_del = np.where(keep_old, g_old, g_new), np.where(keep_old, g_new, g_old)
                    for mode in ("A", "B"):
                        model = model_from_state(gs, st.config["opt_params"])
                        if mode == "B" and len(i_keep):
                            x_m, q_m, s_m, o_m = merge_b(xyz, quat, scale, opac, n_or, i_keep, i_del)
                            ik = torch.from_numpy(i_keep).to("cuda")
                            model._xyz.data[ik] = torch.from_numpy(x_m).float().to("cuda")
                            model._rotation.data[ik] = torch.from_numpy(q_m).float().to("cuda")
                            model._scaling.data[ik] = torch.log(torch.from_numpy(s_m).float().to("cuda"))
                            o_c = torch.from_numpy(o_m).float().to("cuda").clamp(1e-6, 1 - 1e-6)
                            model._opacity.data[ik] = torch.log(o_c / (1 - o_c))[:, None]
                            model.birth_kfIDs[ik] = torch.from_numpy(t0_0.numpy()[g_old]).int().to("cuda")[:, None]
                        dele = torch.zeros(N0, dtype=torch.bool, device="cuda")
                        if len(i_del):
                            dele[torch.from_numpy(i_del).to("cuda")] = True
                        model.prune_points(dele)
                        model.check_bookkeeping()
                        g1 = loop_dump.gaussian_state(model)
                        G1 = gauss_from_dump(g1)
                        m1, valid1, _ = metrics(G1, g1["t0"].reshape(-1).long())
                        # the gap before the merge on the pixels that still show two layers after it
                        vals = []
                        maps0 = M.gap_maps(st, G0, poses, Pi, masks0[0], masks0[1])
                        for u in Pi:
                            vals.append(maps0[u]["gap"][valid1[u] & valid0[u]])
                        vals = torch.cat(vals)
                        n_reg = len(io) + len(inw)
                        v = {"n_pairs": int(len(i_del)), "removed_frac_region": len(i_del) / max(1, n_reg), "removed_frac_map": len(i_del) / max(1, N0),
                             "new_layer_pairs_frac": len(i_del) / max(1, len(inw)), "kept_old_frac": float(keep_old.mean()) if len(keep_old) else None,
                             "normal_dist_med_mm": 1e3 * float(np.median(dn)) if len(dn) else None, "after": m1, "funnel": fun,
                             "gap_before_same_px_mm": None if vals.numel() == 0 else 1e3 * float(vals.median()),
                             "d_psnr_eval": m1["psnr_eval"] - base["psnr_eval"], "d_psnr_all": m1["psnr_all"] - base["psnr_all"],
                             "d_l1_eval_mm": m1["l1_eval_mm"] - base["l1_eval_mm"], "d_l1_all_mm": m1["l1_all_mm"] - base["l1_all_mm"]}
                        v["feasible"] = bool(v["removed_frac_region"] >= 0.20 and v["d_psnr_eval"] >= -0.1 and v["d_psnr_all"] >= -0.1
                                             and v["d_l1_eval_mm"] <= 0.0 and v["d_l1_all_mm"] <= 0.0)
                        row["variants"][f"tau{int(round(1e3 * tau))}_{mode}"] = v
                        del model, G1, g1
                        torch.cuda.empty_cache()
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            b = row["before"]
            print(f"== {row['run']} {row['event']}  N {b['n_gauss']}  region old/new {row['n_region_old']}/{row['n_region_new']}  Pi* {row['n_pi']}  "
                  f"before: two-layer {b['two_layer_frac']:.2f} gap {b['gap_mm']:.1f} mm  PSNR eval/all {b['psnr_eval']:.2f}/{b['psnr_all']:.2f}  L1 eval/all {b['l1_eval_mm']:.1f}/{b['l1_all_mm']:.1f} mm", flush=True)
            for k, v in row["variants"].items():
                m1 = v["after"]
                print(f"   {k:8s} pairs {v['n_pairs']:6d} = {100 * v['removed_frac_region']:.1f} % of region, {100 * v['removed_frac_map']:.1f} % of map  "
                      f"dPSNR eval/all {v['d_psnr_eval']:+.3f}/{v['d_psnr_all']:+.3f} dB  dL1 eval/all {v['d_l1_eval_mm']:+.2f}/{v['d_l1_all_mm']:+.2f} mm  "
                      f"two-layer {m1['two_layer_frac']:.2f}  gap {m1['gap_mm'] if m1['gap_mm'] is None else round(m1['gap_mm'], 1)} mm (before on the same px {v['gap_before_same_px_mm'] if v['gap_before_same_px_mm'] is None else round(v['gap_before_same_px_mm'], 1)})  feasible={v['feasible']}", flush=True)
            del G0
            torch.cuda.empty_cache()
    print("[e4] done ->", out_path)


if __name__ == "__main__":
    main()
