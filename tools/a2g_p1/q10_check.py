"""PLAN A2G-P1 (rút gọn) Q10: run without --pw must equal the E1c run (A3) on frames 0-120:
frame_log n_planes, pairs, init_source identical; camera-centre difference <= 0.1 cm.
usage: python tools/a2g_p1/q10_check.py NEW_RUN E1C_RUN
"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from metrics import load_tum  # noqa: E402

new, ref = sys.argv[1], sys.argv[2]
a = list(csv.DictReader(open(f"{new}/frame_log.csv")))
b = {r["frame_id"]: r for r in csv.DictReader(open(f"{ref}/frame_log.csv"))}
diff = []
for r in a:
    q = b.get(r["frame_id"])
    for k in ("n_planes", "pairs", "init_source"):
        if q is None or r[k] != q[k]:
            diff.append((r["frame_id"], k, r[k], None if q is None else q[k]))
ea, eb = load_tum(f"{new}/traj_est_all.txt"), load_tum(f"{ref}/traj_est_all.txt")
n = len(ea)
dc = [float(np.linalg.norm(x[:3, 3] - y[:3, 3])) for x, y in zip(ea, eb[:n])]
out = dict(frames=len(a), column_mismatches=len(diff), first_mismatches=diff[:10], max_centre_diff_cm=100 * max(dc),
           passed=(not diff) and 100 * max(dc) <= 0.1)
print(json.dumps(out, indent=1))
json.dump(out, open(f"{new}/q10.json", "w"), indent=1)
sys.exit(0 if out["passed"] else 1)
