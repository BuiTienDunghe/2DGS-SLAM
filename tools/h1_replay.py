"""plan v5 step B, H1: does re-measuring the odometry factors keep the deformation of event k through PGO(k+1)?

usage: python tools/h1_replay.py --out DIR RUN_DIR [RUN_DIR ...]
The backend's pose graph is rebuilt offline (tools/pgo_replay.py: odometry from tracked poses, loop measurements
recovered from the stationarity of the real PGO solutions, verified per loop). For every consecutive pair of loop
events (k, k+1) where event k applied a deformation (rigid runs: every pair, scenario (a)), PGO(k+1) is replayed
from the state the run really had before it (poses_pre of dump k+1) under four graphs:
  (a) rigid run data (baseline runs): tracked odometry + recovered loops
  (b) v5 behaviour: tracked odometry, the written poses only as initial values
  (c) plan v5 A2: odometry re-measured from the written poses (= relative poses of poses_pre(k+1))
  (d) (c) + the loop factor of event k re-measured from the written poses
Fidelity: the run's own scenario ((a) rigid, (b) v5) is compared with the real poses_pgo(k+1); the graph error at
poses_final(k) is compared with the logged gtsam_err_after_sync (v5 dumps). After PGO(k+1) the map before the
correction (gauss_pre of dump k+1) is moved rigidly per keyframe like BackEnd.apply_rigid_correction and the two-layer
gap is measured on Pi*(k) (birth-time layers, split s_k). Upstream's loop rules at k+1 are evaluated per scenario.
Writes DIR/h1.jsonl and DIR/loops.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.dump import EventState, list_dumps, load_dump  # noqa: E402
from deform.pipeline import delta_T_dict, inp_from_dump, rigid_result, with_pos  # noqa: E402
from deform.render_utils import DEV  # noqa: E402
from deform.rigid import apply_rigid  # noqa: E402
from e_v4_durability import measure, state_from_gauss, t0_masks  # noqa: E402
from pgo_replay import Replay, load_attempts, mat, rel, sym  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "deform", "selected_v5.yaml"))
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    user = load_deform_config(a.config)
    rows, loop_report = [], {}
    for rd in a.runs:
        dumps = list_dumps(rd)
        D = [load_dump(p) for p in dumps]
        R = Replay(D, load_attempts(rd), D[0]["config"]["Training"])
        loops, checks = R.recover_all()
        loop_report[os.path.basename(rd)] = {f"{l}<->{c}": v for (l, c), v in checks.items()}
        for (l, c), v in checks.items():
            print(f"[loops] {os.path.basename(rd)} {c}<->{l}: {v['source']} grad {v['grad_norm_rest']:.3g} -> {v['grad_norm_full']:.3g}, "
                  f"LM drift from x* {v['lm_drift_from_xstar_m']:.2e} m, err(x*) {v['err_at_xstar']:.2f} vs logged {v['logged_pgo_err_after']}, "
                  f"loop residual {v['loop_residual_t_m']} m / {v['loop_residual_rot_rad']} rad")
        is_deform_run = any((d.get("deform_log") or {}).get("accepted") for d in D)
        for k in range(len(D) - 1):
            dk, dn = D[k], D[k + 1]
            acc_k = bool((dk.get("deform_log") or {}).get("accepted"))
            if is_deform_run and not acc_k:
                continue
            st = EventState(dk)
            cfg = dcfg.resolve(user, st.config)
            split = M.birth_split(dk["meta"]["cur_uid"], dk["meta"]["loop_uid"])
            H, W = int(st.intr["H"]), int(st.intr["W"])
            ids = [int(u) for u in dn["all_cam_ids"]]
            ids_k = [int(u) for u in dk["all_cam_ids"]]
            cu1, lu1 = int(dn["meta"]["cur_uid"]), int(dn["meta"]["loop_uid"])
            pre1 = {int(u): v for u, v in dn["poses_pre"].items()}
            fin_k = {int(u): v for u, v in dk["poses_final"].items()}
            odom_tracked = R.odom_tracked(ids)
            odom_written = {(a_, b_): rel(pre1[a_], pre1[b_]) for a_, b_ in zip(ids[:-1], ids[1:])}
            loops_old = R.loops_upto(k)
            loop_new = R.loop_of(dn)
            row = {"run": os.path.basename(rd), "k": dk["meta"]["event_id"], "pair": f'{dk["meta"]["cur_uid"]}<->{dk["meta"]["loop_uid"]}',
                   "next": f"{cu1}<->{lu1}", "n_frames": len(ids), "n_loops_old": len(loops_old), "loop_k1_in_graph": loop_new is not None, "scenarios": {}}
            # fidelity 1: graph error right after the write at k (v5 dumps log it)
            g_k = R.build(ids_k, R.odom_tracked(ids_k), loops_old)
            row["err_after_write_k_replay"] = float(g_k.error(R.values(ids_k, fin_k)))
            row["err_after_write_k_logged"] = (dk.get("deform_log") or {}).get("gtsam_err_after_sync")
            own = "a" if not is_deform_run else "b"
            scen = {own: (odom_tracked, loops_old)}
            if is_deform_run:
                scen["c"] = (odom_written, loops_old)
                ck, lk = int(dk["meta"]["cur_uid"]), int(dk["meta"]["loop_uid"])
                scen["d"] = (odom_written, [((lc, rel(pre1[lk], pre1[ck])) if lc == (lk, ck) else (lc, m)) for lc, m in loops_old])
            # Pi*(k), reference states (birth layers as the v4 durability tool)
            G_pre_k = state_from_gauss(st.gpre)
            inp_k = inp_from_dump(dk, st.frame)
            dT_k = delta_T_dict(sorted(inp_k["kf_uids"]), inp_k["all_cam_ids"], inp_k["poses_pre"], inp_k["poses_pgo"])
            xyz_rig, rot_rig, poses_rig = rigid_result(inp_k, dT_k)
            G_rig = with_pos(G_pre_k, xyz_rig, rot_rig)
            mk = t0_masks(st.gpre["t0"], split)
            JL, _ = M.select_loop_kfs(st, G_rig, poses_rig, mk[0], mk[1], cfg)
            if not JL:
                row["note"] = "no_overlap at k"
                rows.append(row)
                continue
            _, J_eval = M.split_opt_eval(JL, H, W, cfg["corr"]["checker"])
            Pi, _ = M.pi_star(st, G_rig, poses_rig, [u for u, _ in J_eval], mk[0], mk[1], cfg)
            r_rig, valid_ref = measure(st, G_rig, poses_rig, Pi, mk)
            G_post = state_from_gauss(st.gpre, dk["gauss_post"]["xyz"], dk["gauss_post"]["rot"])
            r_post, _ = measure(st, G_post, {u: torch.as_tensor(v).double() for u, v in fin_k.items()}, Pi, mk, valid_ref)
            G_pre1 = state_from_gauss(dn["gauss_pre"])
            mk1 = t0_masks(dn["gauss_pre"]["t0"], split)
            r_pre1, _ = measure(st, G_pre1, {u: torch.as_tensor(v).double() for u, v in pre1.items()}, Pi, mk1, valid_ref)
            r_real1, _ = measure(st, state_from_gauss(dn["gauss_pre"], dn["gauss_post"]["xyz"], dn["gauss_post"]["rot"]),
                                 {int(u): torch.as_tensor(v).double() for u, v in dn["poses_final"].items()}, Pi, mk1, valid_ref)
            row.update({"rigid_k_mm": r_rig["median_mm"], "post_k_mm": r_post["median_mm"], "pre_k1_mm": r_pre1["median_mm"],
                        "real_post_k1_mm": r_real1["median_mm"], "n_pi": int(sum(int(m.sum()) for m, _ in Pi.values()))})
            gain_k = (r_rig["median_mm"] or 0) - (r_post["median_mm"] or 0)
            kf_uids = sorted(int(u) for u in dn["keyframe_uids"])
            for name, (odom, lp) in scen.items():
                g_no = R.build(ids, odom, lp)
                v0 = R.values(ids, pre1)
                e_no = float(g_no.error(v0))
                if loop_new is None:
                    e_with, thr, reject, remove, keep_loop = e_no, None, False, True, False
                    g_run = g_no
                else:
                    g_with = R.build(ids, odom, lp + [loop_new])
                    e_with = float(g_with.error(v0))
                    thr = e_no + (cu1 - 0) * R.thr_frame
                    reject = e_with > thr
                    remove = (not reject) and e_with < 50.0
                    keep_loop = (not reject) and (not remove)
                    g_run = g_with if keep_loop else g_no
                e0 = float(g_run.error(v0))
                v1 = v0 if e0 < 1e-4 else R.optimize(g_run, v0)
                e1 = float(g_run.error(v1))
                after = {u: torch.from_numpy(v1.atPose3(sym(u)).matrix()) for u in ids}
                dev = max(float(np.linalg.norm(after[u][:3, 3].numpy() - mat(dn["poses_pgo"][u])[:3, 3])) for u in ids)
                dT = delta_T_dict(kf_uids, ids, pre1, after)
                xyz1, rot1 = apply_rigid(G_pre1["xyz"], G_pre1["rot"], dn["gauss_pre"]["tc"].to(DEV), dT)
                r_s, _ = measure(st, with_pos(G_pre1, xyz1, rot1), after, Pi, mk1, valid_ref)
                retained = None if gain_k <= 0 or r_s["median_mm"] is None else (r_rig["median_mm"] - r_s["median_mm"]) / gain_k
                row["scenarios"][name] = {"err_no_loop": e_no, "err_with_loop": e_with, "thr": thr, "loop_rejected": reject, "loop_removed_lt50": remove,
                                          "loop_kept": keep_loop, "pgo_err_before": e0, "pgo_err_after": e1, "max_dev_vs_real_m": dev, "own_scenario": name == own,
                                          "gap_after_k1_mm": r_s["median_mm"], "valid_frac": r_s["valid_frac"], "retained": retained}
            rows.append(row)
            print(json.dumps({kk: row.get(kk) for kk in ("run", "k", "pair", "next", "rigid_k_mm", "post_k_mm", "pre_k1_mm", "real_post_k1_mm",
                                                         "err_after_write_k_replay", "err_after_write_k_logged")}, default=str))
            for name, sc in row["scenarios"].items():
                print(f"   ({name}) loop kept={sc['loop_kept']} err {sc['err_no_loop']:.1f}/{sc['err_with_loop']:.1f} pgo {sc['pgo_err_before']:.1f}->{sc['pgo_err_after']:.1f} "
                      f"dev_vs_real {sc['max_dev_vs_real_m']:.2e} m gap {sc['gap_after_k1_mm']} mm retained {sc['retained']}")
            torch.cuda.empty_cache()
    with open(os.path.join(a.out, "h1.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    json.dump(loop_report, open(os.path.join(a.out, "loops.json"), "w"), indent=1, default=str)
    print(f"[h1_replay] {len(rows)} rows -> {a.out}/h1.jsonl")


if __name__ == "__main__":
    main()
