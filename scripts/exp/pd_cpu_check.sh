#!/usr/bin/env bash
# P vs D experiment, CPU-only preparation: syntax check + loop measurements of every dump (gtsam, no GPU).
cd "$(dirname "$0")/../.."
source tools/exp_env.sh
export CUDA_VISIBLE_DEVICES=
python -m py_compile tools/pd_experiment.py && echo "py_compile ok"
python - <<'EOF'
import os, sys
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import numpy as np, torch
from pgo_replay import Replay, load_attempts, mat
import glob
R = "results_exp/tum/room/"
for rd in ("20261002013837_rigid_s0", "20261003123233_rigid_s1", "20261003105219_deform5_s0", "20261003114156_deform5_s1"):
    dumps = sorted(glob.glob(os.path.join(R, rd, "loop_dumps", "event_*.pt")))
    D = [torch.load(p, map_location="cpu", weights_only=False) for p in dumps]
    loops, checks = Replay(D, load_attempts(os.path.join(R, rd)), D[0]["config"]["Training"]).recover_all()
    for d in D:
        C, L = int(d["meta"]["cur_uid"]), int(d["meta"]["loop_uid"])
        gt = d.get("poses_gt") or {}
        keys = sorted(d["poses_pre"].keys())
        info = f"N={d['gauss_pre']['xyz'].shape[0]} kfs={len(d['keyframe_uids'])} frames={len(keys)} gt={len(gt)} raw={'scale_raw' in d['gauss_pre']} same_keys={sorted(d['poses_pgo'].keys()) == keys}"
        if (L, C) not in loops:
            print(rd, f"{C}<->{L}", "NO LOOP MEASUREMENT", info)
            continue
        m = loops[(L, C)]
        PL_, PC = mat(d["poses_pre"][L]), mat(d["poses_pre"][C])
        H = PL_ @ m @ np.linalg.inv(PC)
        dP = mat(d["poses_pgo"][C]) @ np.linalg.inv(PC)
        ang = np.degrees(np.arccos(np.clip((np.trace(H[:3, :3]) - 1) / 2, -1, 1)))
        c = checks[(L, C)]
        print(rd, f"{C}<->{L}", f"src={c['source']} lm_drift={c['lm_drift_from_xstar_m']:.2e} err*={c['err_at_xstar']:.1f} logged={c['logged_pgo_err_after']}",
              f"H: cur shift {1e3 * np.linalg.norm((H @ PC)[:3, 3] - PC[:3, 3]):.0f} mm rot {ang:.2f} deg | PGO cur shift {1e3 * np.linalg.norm((dP @ PC)[:3, 3] - PC[:3, 3]):.0f} mm"
              f" | H vs PGO at cur {1e3 * np.linalg.norm((H @ PC)[:3, 3] - (dP @ PC)[:3, 3]):.0f} mm", info)
EOF
