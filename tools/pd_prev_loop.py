"""P vs D diagnostic: does D (no PGO) re-open the regions that EARLIER loops had closed?

usage: python tools/pd_prev_loop.py --out DIR RUN_DIR [RUN_DIR ...]
For event k of a run and every earlier event j with a two-layer region: Pi*(j) is built on the rigid replay of
dump j (birth-time layers, split s_j); at event k the same pixels are measured with the layers t0 < s_j / t0 >= s_j
of the map of event k in three states: pre (before the correction of event k), P0 (PGO + rigid) and D (coarse
deformation graph of tools/pd_experiment.py, same solve). Median |gap| on Pi*(j), pixels valid in P0 and D.
Writes DIR/pd_prev_loop.jsonl.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pd_experiment as PD  # noqa: E402
from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.dump import EventState, list_dumps, load_dump  # noqa: E402
from deform.pipeline import delta_T_dict, delta_t_frames, inp_from_dump, rigid_result, with_pos  # noqa: E402
from deform.reliability import compute_reliability  # noqa: E402
from deform.render_utils import det_scope, make_cam, torch_det_scope  # noqa: E402
from pgo_replay import Replay, load_attempts, mat  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402


def gap_on(st, Pi, states, masks):
    maps = {n: M.gap_maps(st, G, P, Pi, masks[0], masks[1]) for n, (G, P) in states.items()}
    out = {}
    for n in states:
        own, com = [], []
        for u, (m, _) in Pi.items():
            v = m & maps[n][u]["valid"]
            own.append(maps[n][u]["gap"][v])
            for o in ("P0", "D"):
                v = v & maps[o][u]["valid"]
            com.append(maps[n][u]["gap"][v])
        own, com = torch.cat(own), torch.cat(com)
        out[n] = {"gap_mm": None if com.numel() == 0 else 1e3 * float(com.median()), "n": int(com.numel()),
                  "gap_own_mm": None if own.numel() == 0 else 1e3 * float(own.median()), "n_own": int(own.numel())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(PD.ROOT, "configs", "deform", "selected_v6.yaml"))
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    user = load_deform_config(a.config)
    out_path = os.path.join(a.out, "pd_prev_loop.jsonl")
    open(out_path, "w").close()
    for rd in a.runs:
        D = [load_dump(p) for p in list_dumps(rd)]
        loops, _ = Replay(D, load_attempts(rd), D[0]["config"]["Training"]).recover_all()
        prev = []  # (event name, split, Pi, rigid gap at its own event)
        for d in D:
            t_ev = time.time()
            st = EventState(d)
            C, L = int(d["meta"]["cur_uid"]), int(d["meta"]["loop_uid"])
            cfg = dcfg.resolve(user, st.config)
            cfg["seed"] = int(d["meta"].get("seed", 0))
            inp = inp_from_dump(d, st.frame)
            G_pre = inp["g"]
            kf_uids = sorted(inp["kf_uids"])
            Hh, Wd = int(st.intr["H"]), int(st.intr["W"])
            split = M.birth_split(C, L)
            m_old, m_new = M.layer_masks_birth(inp["t0"], split)
            dT = delta_T_dict(kf_uids, inp["all_cam_ids"], inp["poses_pre"], inp["poses_pgo"])
            xyz_rig, rot_rig, poses_rig = rigid_result(inp, dT)
            G_rig = with_pos(G_pre, xyz_rig, rot_rig)
            sink = []
            with torch.no_grad(), det_scope(True), torch_det_scope(True, sink):
                JL, _ = M.select_loop_kfs(st, G_rig, poses_rig, m_old, m_new, cfg)
                Pi_k = None
                if JL:
                    _, J_eval = M.split_opt_eval(JL, Hh, Wd, cfg["corr"]["checker"])
                    Pi_k, _ = M.pi_star(st, G_rig, poses_rig, [u for u, _ in J_eval], m_old, m_new, cfg)
                if prev and (L, C) in loops:
                    dt = delta_t_frames(kf_uids, inp["W"])
                    cams_pre = {u: make_cam(u, inp["poses_pre"][u], inp["intr"]) for u in kf_uids}
                    rel = compute_reliability(G_pre, cams_pre, inp["frame_fn"], inp["t0"], dt, cfg, inp["tr"], kf_uids)
                    del cams_pre
                    P_L, P_C = mat(inp["poses_pre"][L]), mat(inp["poses_pre"][C])
                    H_np = P_L @ loops[(L, C)] @ np.linalg.inv(P_C)
                    H_np[:3, :3] = PD.so3(H_np[:3, :3])
                    graph = PD.build_graph(inp, rel, cfg)
                    r = PD.coarse(inp, H_np, rel, cfg, graph, m_new, "time")[""]
                    if r.get("xyz") is not None:
                        states = {"pre": (G_pre, {u: torch.as_tensor(v).double().cpu() for u, v in inp["poses_pre"].items()}),
                                  "P0": (G_rig, poses_rig), "D": (with_pos(G_pre, r["xyz"], r["rot"]), r["poses"])}
                        row = {"run": os.path.basename(rd), "event": f"{C}<->{L}", "cur": C, "loop": L,
                               "ate": {n: M.ate_kf(st, P) for n, (_, P) in states.items()}, "lbfgs_iters": r["log"]["lbfgs_iters"], "regions": {}}
                        # where the nodes of the earlier loops' frames went: a = share of H given by the birth-time init
                        row["own"] = None if Pi_k is None else gap_on(st, Pi_k, states, (m_old, m_new))
                        for name, split_j, Pi_j, C_j, L_j in prev:
                            if any(u not in poses_rig for u in Pi_j):
                                continue
                            masks = M.layer_masks_birth(inp["t0"], split_j)
                            g_ = gap_on(st, Pi_j, states, masks)
                            g_["init_share_of_H_at_Cj"] = float(np.clip((C_j - L) / float(C - L), 0, 1))
                            g_["init_share_of_H_at_Lj"] = float(np.clip((L_j - L) / float(C - L), 0, 1))
                            row["regions"][name] = g_
                        row["wall_s"] = time.time() - t_ev
                        with open(out_path, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        print(f"== {row['run']} {row['event']}  ATE pre/P0/D {row['ate']['pre'] * 100:.2f}/{row['ate']['P0'] * 100:.2f}/{row['ate']['D'] * 100:.2f} cm  "
                              f"own Pi* gap pre/P0/D: " + ("—" if row["own"] is None else "/".join(PD.fmt(row["own"][n]["gap_mm"]) for n in ("pre", "P0", "D"))) + f" mm  [{row['wall_s']:.0f} s]", flush=True)
                        for name, g_ in row["regions"].items():
                            print(f"     region of {name}: gap pre/P0/D {PD.fmt(g_['pre']['gap_own_mm'])}/{PD.fmt(g_['P0']['gap_mm'])}/{PD.fmt(g_['D']['gap_mm'])} mm ({g_['P0']['n']} px)  "
                                  f"init share of H at its new/old side: {g_['init_share_of_H_at_Cj']:.2f}/{g_['init_share_of_H_at_Lj']:.2f}", flush=True)
                        del states, r, graph, rel
            if Pi_k is not None:
                prev.append((f"{C}<->{L}", split, Pi_k, C, L))
            torch.cuda.empty_cache()
    print("[pd_prev_loop] done ->", out_path)


if __name__ == "__main__":
    main()
