"""G0a/G0b: check a run folder has every output and the t0 bookkeeping invariants hold in final_state.pt.

usage: python tools/check_run_outputs.py RUN_DIR [--refine]   -> prints JSON, exit 1 on failure
"""
import json
import os
import sys

import torch

REQ = ["metrics.csv", "final_state.pt", "resources.json", "gsmap_{t}.ply", "mesh_{t}.ply",
       "traj_tum_{t}.txt", "key_pose_{t}.txt"]
REQ_REFINE = ["metrics_prerefine.csv", "final_state_refined.pt", "refined_gsmap_{t}.ply"]


def main(rd, refine):
    tag = os.path.basename(rd.rstrip("/")).split("_", 1)[1]
    files = REQ + (REQ_REFINE if refine else [])
    missing = [f.format(t=tag) for f in files if not os.path.exists(os.path.join(rd, f.format(t=tag)))]
    out = {"run": rd, "missing": missing}
    ok = not missing
    for name in ("final_state.pt", "final_state_refined.pt"):
        p = os.path.join(rd, name)
        if not os.path.exists(p):
            continue
        s = torch.load(p, map_location="cpu", weights_only=False)
        g = s["gaussians"]
        n = g["xyz"].shape[0]
        lens = {k: int(v.shape[0]) for k, v in g.items()}
        t0, tc, tl = g["t0"], g["tc"], g["tl"]
        known = t0 >= 0
        Rs = torch.stack([s["poses"][u][:3, :3].double() for u in sorted(s["poses"])])
        orth = (Rs @ Rs.transpose(1, 2) - torch.eye(3, dtype=torch.float64)).abs().amax(dim=(1, 2))
        dets = torch.linalg.det(Rs)
        chk = {"N": n, "len_ok": all(v == n for v in lens.values()),
               "t0_le_tc": bool(((t0 <= tc) | ~known).all()), "t0_le_tl": bool(((t0 <= tl) | ~known).all()),
               "t0_unknown_frac": float((~known).float().mean()) if n else None,
               "n_poses": len(s["poses"]), "n_kf": len(s["keyframe_uids"]),
               "pose_orth_err_max": float(orth.max()), "pose_det_min": float(dets.min()), "pose_det_max": float(dets.max())}
        dev = max(chk["pose_orth_err_max"], abs(chk["pose_det_min"] - 1), abs(chk["pose_det_max"] - 1))
        # fp32 accumulation over ~1000 frames reaches ~1e-3 (Replica run #2: det 1.0014); TF32 reached 0.3
        chk["pose_warn"] = dev > 1e-3
        chk["pose_ok"] = dev < 1e-2
        out[name] = chk
        ok &= chk["len_ok"] and chk["t0_le_tc"] and chk["t0_le_tl"] and chk["pose_ok"]
    ev = os.path.join(rd, "loop_events.jsonl")
    out["n_loop_events"] = sum(1 for _ in open(ev)) if os.path.exists(ev) else 0
    at = os.path.join(rd, "loop_attempts.jsonl")
    out["n_loop_attempts"] = sum(1 for _ in open(at)) if os.path.exists(at) else 0
    dd = os.path.join(rd, "loop_dumps")
    out["n_dumps"] = len(os.listdir(dd)) if os.path.isdir(dd) else 0
    out["ok"] = bool(ok)
    print(json.dumps(out, indent=1))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], "--refine" in sys.argv) else 1)
