"""plan v4 step E (2) = plan v3 step F.2: durability of the correction of each loop event.

usage: python tools/e_v4_durability.py --out DIR RUN_DIR [RUN_DIR ...]
For event k of a run, the two layers are defined by BIRTH TIME with the split s_k = (loop_uid + cur_uid) / 2
(old: t0 <= s_k, new: t0 > s_k), so that the same region can be measured again on later states, where the
active mask of event k no longer exists. The evaluation keyframes J_eval(k) and pixels Pi*(k) are built on the
rigid replay of event k with these layers (same recipe as v2/v3 otherwise). States measured on Pi*(k):
  rigid@k   rigid replay of the dump (what the baseline fix gives)
  post@k    the map and poses the run really had right after the event (gauss_post, poses_final of the dump)
  pre@k+1   gauss_pre / poses_pre of the next dump (mapping continued, before the next correction)
  final     final_state.pt (end of SLAM, before refinement)
Per state: median |gap| (mm) on Pi*(k) ∩ valid(state) ∩ valid(rigid@k) and the valid fraction of Pi*(k).
Writes DIR/durability.jsonl and prints a table.
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
from deform.pipeline import delta_T_dict, gauss_from_dump, inp_from_dump, rigid_result, with_pos  # noqa: E402
from deform.render_utils import DEV  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def t0_masks(t0, split):
    t = t0.to(DEV).reshape(-1).float()
    return t <= split, t > split


def state_from_gauss(g, xyz=None, rot=None):
    G = gauss_from_dump(g) if "scale_raw" in g else {
        "xyz": g["xyz"].float().to(DEV), "rot": g["rot"].float().to(DEV), "scale": g["scale"].float().to(DEV),
        "opacity": g["opacity"].float().to(DEV), "f_dc": g["f_dc"].float().to(DEV)}
    if xyz is not None:
        G = with_pos(G, xyz.float().to(DEV), rot.float().to(DEV))
    return G


def measure(st, G, poses, Pi, masks, ref_valid=None):
    maps = M.gap_maps(st, G, poses, Pi, masks[0], masks[1])
    vals, n_valid, n_pi = [], 0, 0
    for uid, (m, _) in Pi.items():
        v = m & maps[uid]["valid"]
        n_pi += int(m.sum())
        n_valid += int(v.sum())
        if ref_valid is not None:
            v &= ref_valid[uid]
        vals.append(maps[uid]["gap"][v])
    x = torch.cat(vals) if vals else torch.zeros(0, device=DEV)
    return {"median_mm": None if x.numel() == 0 else 1e3 * float(x.median()),
            "p90_mm": None if x.numel() == 0 else 1e3 * float(torch.quantile(x.float(), 0.9)),
            "n": int(x.numel()), "valid_frac": n_valid / max(1, n_pi)}, {u: (Pi[u][0] & maps[u]["valid"]) for u in Pi}


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
        dumps = list_dumps(rd)
        fs_path = os.path.join(rd, "final_state.pt")
        final = torch.load(fs_path, map_location="cpu", weights_only=False) if os.path.exists(fs_path) else None
        loaded = [load_dump(p) for p in dumps]
        for k, (p, d) in enumerate(zip(dumps, loaded)):
            st = EventState(d)
            cfg = dcfg.resolve(user, st.config)
            split = 0.5 * (float(st.meta["loop_uid"]) + float(st.meta["cur_uid"]))
            H, W = int(st.intr["H"]), int(st.intr["W"])
            G_pre = state_from_gauss(st.gpre)
            inp = inp_from_dump(d, st.frame)
            dT = delta_T_dict(sorted(inp["kf_uids"]), inp["all_cam_ids"], inp["poses_pre"], inp["poses_pgo"])
            xyz_rig, rot_rig, poses_rig = rigid_result(inp, dT)
            G_rig = with_pos(G_pre, xyz_rig, rot_rig)
            mk = t0_masks(st.gpre["t0"], split)
            JL, _ = M.select_loop_kfs(st, G_rig, poses_rig, mk[0], mk[1], cfg)
            row = {"run": os.path.basename(rd), "dump": os.path.basename(p), "event_id": st.meta["event_id"], "cur": st.meta["cur_uid"],
                   "loop": st.meta["loop_uid"], "mode": st.meta.get("mode"), "split_t0": split, "n_loop_kfs": len(JL),
                   "accepted": (d.get("deform_log") or {}).get("accepted")}
            if not JL:
                row["note"] = "no_overlap"
                rows.append(row)
                continue
            _, J_eval = M.split_opt_eval(JL, H, W, cfg["corr"]["checker"])
            J_eval_u = [u for u, _ in J_eval]
            Pi, _ = M.pi_star(st, G_rig, poses_rig, J_eval_u, mk[0], mk[1], cfg)
            row["n_pi"] = int(sum(int(m.sum()) for m, _ in Pi.values()))
            row["J_eval"] = J_eval_u
            states = {"rigid@k": (G_rig, poses_rig, mk)}
            G_post = state_from_gauss(st.gpre, d["gauss_post"]["xyz"], d["gauss_post"]["rot"])
            poses_post = {int(u): torch.as_tensor(v).double() for u, v in d["poses_final"].items()}
            states["post@k"] = (G_post, poses_post, mk)
            if k + 1 < len(loaded):
                dn = loaded[k + 1]
                G_next = state_from_gauss(dn["gauss_pre"])
                poses_next = {int(u): torch.as_tensor(v).double() for u, v in dn["poses_pre"].items()}
                states["pre@k+1"] = (G_next, poses_next, t0_masks(dn["gauss_pre"]["t0"], split))
            if final is not None:
                gf = final["gaussians"]
                G_fin = state_from_gauss(gf)
                poses_fin = {int(u): torch.as_tensor(v).double() for u, v in final["poses"].items()}
                states["final"] = (G_fin, poses_fin, t0_masks(gf["t0"], split))
            ref_valid = None
            for name, (G, P, masks) in states.items():
                missing = [u for u in Pi if u not in P]
                if missing:
                    row[name] = {"note": f"poses missing for {missing[:3]}"}
                    continue
                r, valid = measure(st, G, P, Pi, masks, ref_valid)
                if name == "rigid@k":
                    ref_valid = valid
                row[name] = r
            rows.append(row)
            print(json.dumps({kk: row.get(kk) for kk in ("run", "event_id", "cur", "loop", "accepted", "n_pi", "rigid@k", "post@k", "pre@k+1", "final")}, default=str), flush=True)
            del states, G_post, G_rig, G_pre
            torch.cuda.empty_cache()
    with open(os.path.join(a.out, "durability.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"[e_v4_durability] {len(rows)} rows -> {a.out}/durability.jsonl")


if __name__ == "__main__":
    main()
