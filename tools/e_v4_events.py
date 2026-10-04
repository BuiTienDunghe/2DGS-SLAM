"""plan v4 step E (1): per online loop event, (a) exactness of the online hook against an offline recomputation
from the dump, (b) e_dl on Pi* (cells B of J_eval, active-based layers as in v2/v3) for rigid, online, offline.

usage: python tools/e_v4_events.py --out DIR [--config configs/deform/selected_v5.yaml] RUN_DIR [RUN_DIR ...]
For every dump of every run: rigid replay (dT from the dumped poses) -> J_L / J_opt / J_eval on the rigid state ->
Pi*; online result = gauss_post + poses_final of the dump; offline = correct_map(selected_v5) on the dump.
Rows: run, event, pairs, accepted, R_online, R_offline, |online - offline| (Gaussians, poses), node translation
difference (offline recompute vs the online log is not stored -> only Gaussians / poses), hook time, VRAM,
anchor, gtsam errors. For a rigid run (no deform config in the dump meta) only rigid / online(=rigid) rows.
Writes DIR/events.jsonl and prints a table.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import torch  # noqa: E402
import yaml  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.dump import EventState, gauss_to_dev, list_dumps, load_dump  # noqa: E402
from deform.pipeline import correct_map, gauss_from_dump, inp_from_dump, with_pos  # noqa: E402
from deform.render_utils import DEV  # noqa: E402
from deform.rigid import replay  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel_red(r, x):
    return None if r is None or x is None or r <= 0 else (r - x) / r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "deform", "selected_v5.yaml"))
    ap.add_argument("--no-offline", action="store_true", help="skip the offline recomputation (rigid runs)")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    user = load_deform_config(a.config)
    rows = []
    for rd in a.runs:
        for p in list_dumps(rd):
            t0 = time.time()
            d = load_dump(p)
            st = EventState(d)
            cfg = dcfg.resolve(user, st.config)
            cfg["seed"] = int(st.meta.get("seed", 0))
            m_old, m_new = M.layer_masks(st.active)
            H, W = int(st.intr["H"]), int(st.intr["W"])
            G_pre = gauss_from_dump(st.gpre)
            # rigid reference exactly like the pipeline (same dT formula)
            inp = inp_from_dump(d, st.frame)
            from deform.pipeline import delta_T_dict, rigid_result
            dT = delta_T_dict(sorted(inp["kf_uids"]), inp["all_cam_ids"], inp["poses_pre"], inp["poses_pgo"])
            xyz_rig, rot_rig, poses_rig = rigid_result(inp, dT)
            G_rig = with_pos(G_pre, xyz_rig, rot_rig)
            JL, _ = M.select_loop_kfs(st, G_rig, poses_rig, m_old, m_new, cfg)
            row = {"run": os.path.basename(rd), "dump": os.path.basename(p), "event_id": st.meta["event_id"],
                   "cur": st.meta["cur_uid"], "loop": st.meta["loop_uid"], "mode": st.meta.get("mode"),
                   "n_loop_kfs": len(JL), "raw_params": "scale_raw" in st.gpre}
            lg = d.get("deform_log") or {}
            row.update({k: lg.get(k) for k in ("accepted", "fallback_reason", "corr_gated", "corr_capped", "lbfgs_iters",
                                               "edl_opt_init_mm", "edl_opt_final_mm", "max_node_disp_m", "t_hook_s",
                                               "vram_event_peak_gb", "vram_before_gb", "anchor", "anchored_uids", "anchor_resid_m",
                                               "gtsam_err_before_sync", "gtsam_err_after_sync", "pgo_err_before", "pgo_err_after",
                                               "det_warnings", "frac_res_gt30mm", "accept_checks")})
            if not JL:
                row["note"] = "no_overlap"
                rows.append(row)
                print(json.dumps(row, default=str)[:300], flush=True)
                continue
            J_opt, J_eval = M.split_opt_eval(JL, H, W, cfg["corr"]["checker"])
            J_eval_u = [u for u, _ in J_eval]
            Pi, _ = M.pi_star(st, G_rig, poses_rig, J_eval_u, m_old, m_new, cfg)
            # online result from the dump
            G_on = with_pos(G_pre, d["gauss_post"]["xyz"].float().to(DEV), d["gauss_post"]["rot"].float().to(DEV))
            poses_on = {int(u): torch.as_tensor(v).double() for u, v in d["poses_final"].items()}
            maps = {"rigid": M.gap_maps(st, G_rig, poses_rig, Pi, m_old, m_new),
                    "online": M.gap_maps(st, G_on, poses_on, Pi, m_old, m_new)}
            res = None
            if not a.no_offline and st.meta.get("mode") == "deform":
                res = correct_map(inp, cfg, cfg["variant"])
                G_off = with_pos(G_pre, res["xyz"], res["rot"])
                maps["offline"] = M.gap_maps(st, G_off, res["poses"], Pi, m_old, m_new)
                row["off_accepted"], row["off_reason"] = res["accepted"], res["reason"]
                row["off_iters"] = res["log"].get("lbfgs_iters")
                row["on_off_dmu_max_m"] = float((G_on["xyz"] - res["xyz"]).norm(dim=1).max())
                row["on_off_dmu_n_gt_1e-6"] = int(((G_on["xyz"] - res["xyz"]).norm(dim=1) > 1e-6).sum())
                row["on_off_dpose_max_m"] = max(float((poses_on[u][:3, 3] - res["poses"][u].double()[:3, 3]).norm()) for u in res["poses"] if u in poses_on)
                row["off_edl_opt_final_mm"] = res["log"].get("edl_opt_final_mm")
                row["off_t_total_s"] = res["log"]["t_stage_s"].get("total")
            e, n_pix = M.edl_paired(Pi, maps)
            row["n_pix"] = n_pix
            for k, v in e.items():
                row[f"edl_{k}_mm"] = None if v["median"] is None else 1e3 * v["median"]
                row[f"p90_{k}_mm"] = None if v["p90"] is None else 1e3 * v["p90"]
            row["R_online"] = rel_red(e["rigid"]["median"], e["online"]["median"])
            row["R_offline"] = None if "offline" not in e else rel_red(e["rigid"]["median"], e["offline"]["median"])
            # ATE / PSNR at the event (keyframes J_eval)
            row["ate_rigid_m"], row["ate_online_m"] = M.ate_kf(st, poses_rig), M.ate_kf(st, poses_on)
            rq_r, rq_o = M.render_quality(st, G_rig, poses_rig, sorted(J_eval_u)), M.render_quality(st, G_on, poses_on, sorted(J_eval_u))
            row["psnr_rigid"], row["psnr_online"] = rq_r.get("psnr"), rq_o.get("psnr")
            row["eval_s"] = time.time() - t0
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("run", "event_id", "cur", "loop", "accepted", "corr_capped", "n_pix", "edl_rigid_mm",
                                                      "edl_online_mm", "edl_offline_mm", "R_online", "R_offline", "on_off_dmu_max_m",
                                                      "on_off_dpose_max_m", "lbfgs_iters", "off_iters", "t_hook_s", "eval_s")}, default=str), flush=True)
            del res, maps, G_on, G_rig, G_pre
            torch.cuda.empty_cache()
    with open(os.path.join(a.out, "events.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"[e_v4_events] {len(rows)} rows -> {a.out}/events.jsonl")


if __name__ == "__main__":
    main()
