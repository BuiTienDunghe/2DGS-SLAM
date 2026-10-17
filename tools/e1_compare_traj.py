"""T7/T8: per-frame pose difference between runs (traj_tum_<tag>.txt: frame_id tx ty tz qx qy qz qw, c2w).
usage: python tools/e1_compare_traj.py REF_DIR RUN_DIR [RUN_DIR ...]
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def load(d):
    f = sorted(glob.glob(os.path.join(d, "traj_tum_*.txt")))[0]
    a = np.loadtxt(f)
    return {int(r[0]): (r[1:4], Rotation.from_quat(r[4:8])) for r in a}


def ate(d):
    m = os.path.join(d, "metrics.csv")
    if not os.path.exists(m):
        return None
    lines = open(m).read().strip().splitlines()
    hdr, val = lines[0].split(","), lines[1].split(",")
    return float(dict(zip(hdr, val))["ate_rmse_all_tracked_m"])


ref = load(sys.argv[1])
out = {"ref": sys.argv[1], "ref_ate_all_m": ate(sys.argv[1]), "runs": []}
for d in sys.argv[2:]:
    run = load(d)
    ids = sorted(set(ref) & set(run))
    dt = np.array([np.linalg.norm(ref[i][0] - run[i][0]) for i in ids])
    dr = np.array([np.degrees((ref[i][1].inv() * run[i][1]).magnitude()) for i in ids])
    first = next((i for i, x in zip(ids, dt) if x > 1e-4), None)
    out["runs"].append(dict(run=d, ate_all_m=ate(d), n=len(ids), max_dt_m=float(dt.max()), median_dt_m=float(np.median(dt)),
                            final_dt_m=float(dt[-1]), max_drot_deg=float(dr.max()), first_frame_dt_gt_1e4=first,
                            dt_at={str(i): float(x) for i, x in zip(ids, dt) if i in (2, 4, 10, 20, 40, 80, 118)}))
print(json.dumps(out, indent=2))
