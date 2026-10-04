"""plan v5 step B, H2 (report only): variant B (node init from the keyframe pose increments, pre coordinates) against
the rigid fix at the event, on the baseline dumps.

usage: python tools/h2_variant_b.py --out DIR RUN_DIR [RUN_DIR ...] [--config configs/deform/selected_v5.yaml]
Per dump: correct_map with the frozen config but variant B, R on Pi* (active-mask layers, as v2/v3) vs rigid, and the
same for variant A for reference. Writes DIR/h2.jsonl.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.dump import EventState, list_dumps, load_dump  # noqa: E402
from deform.pipeline import correct_map, delta_T_dict, gauss_from_dump, inp_from_dump, rigid_result, with_pos  # noqa: E402
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
    rows = []
    for rd in a.runs:
        for p in list_dumps(rd):
            d = load_dump(p)
            st = EventState(d)
            cfg = dcfg.resolve(user, st.config)
            cfg["seed"] = int(d["meta"].get("seed", 0))
            m_old, m_new = M.layer_masks(st.active)
            H, W = int(st.intr["H"]), int(st.intr["W"])
            inp = inp_from_dump(d, st.frame)
            G_pre = gauss_from_dump(st.gpre)
            dT = delta_T_dict(sorted(inp["kf_uids"]), inp["all_cam_ids"], inp["poses_pre"], inp["poses_pgo"])
            xyz_rig, rot_rig, poses_rig = rigid_result(inp, dT)
            G_rig = with_pos(G_pre, xyz_rig, rot_rig)
            JL, _ = M.select_loop_kfs(st, G_rig, poses_rig, m_old, m_new, cfg)
            row = {"run": os.path.basename(rd), "dump": os.path.basename(p), "cur": st.meta["cur_uid"], "loop": st.meta["loop_uid"], "n_loop_kfs": len(JL)}
            if not JL:
                rows.append(row)
                print(json.dumps(row))
                continue
            _, J_eval = M.split_opt_eval(JL, H, W, cfg["corr"]["checker"])
            Pi, _ = M.pi_star(st, G_rig, poses_rig, [u for u, _ in J_eval], m_old, m_new, cfg)
            maps = {"rigid": M.gap_maps(st, G_rig, poses_rig, Pi, m_old, m_new)}
            for var in ("A", "B"):
                c = json.loads(json.dumps(cfg))
                c["variant"] = var
                try:
                    res = correct_map(inp_from_dump(d, st.frame), c, var)
                except Exception as ex:
                    import traceback
                    row[f"{var}_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
                    row[f"{var}_traceback"] = traceback.format_exc()[-1500:]
                    print(json.dumps({"run": row["run"], "cur": row["cur"], "variant": var, "error": row[f"{var}_error"]}))
                    continue
                maps[var] = M.gap_maps(st, with_pos(G_pre, res["xyz"], res["rot"]), res["poses"], Pi, m_old, m_new)
                row[f"{var}_accepted"], row[f"{var}_reason"] = res["accepted"], res["reason"]
                row[f"{var}_iters"] = res["log"].get("lbfgs_iters")
                row[f"{var}_edl_opt_mm"] = (res["log"].get("edl_opt_init_mm"), res["log"].get("edl_opt_final_mm"))
                row[f"{var}_max_node_disp_mm"] = None if res["log"].get("max_node_disp_m") is None else 1e3 * res["log"]["max_node_disp_m"]
                row[f"{var}_t_s"] = res["log"]["t_stage_s"].get("total")
                del res
                torch.cuda.empty_cache()
            e, n = M.edl_paired(Pi, maps)
            row["n_pix"] = n
            for k, v in e.items():
                row[f"edl_{k}_mm"] = None if v["median"] is None else 1e3 * v["median"]
            r = e["rigid"]["median"]
            for var in ("A", "B"):
                row[f"R_{var}"] = None if (var not in e or r is None or e[var]["median"] is None) else (r - e[var]["median"]) / r
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("run", "cur", "loop", "n_pix", "edl_rigid_mm", "edl_A_mm", "edl_B_mm", "R_A", "R_B", "A_accepted", "B_accepted", "B_reason", "B_max_node_disp_mm")}, default=str))
    with open(os.path.join(a.out, "h2.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()
