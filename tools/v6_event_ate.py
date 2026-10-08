"""plan v6 (extra, v4-measurement F2): what each PGO event did to the trajectory, from the loop dumps of one run.

usage: python tools/v6_event_ate.py --out DIR RUN_DIR [RUN_DIR ...]
Per event: ATE (SE(3)-aligned RMSE over the frames tracked so far) with the poses right before the PGO and with the
PGO result, and how far the camera centres moved. Paired within the run, so free of run-to-run noise; it does not
include what the corrected map does to the tracking afterwards. Writes DIR/event_ate.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402

from deform.dump import list_dumps, load_dump  # noqa: E402
from v6_ate_split import umeyama_rigid  # noqa: E402


def ate(est, gt, ids):
    pe = np.stack([np.asarray(est[u], dtype=np.float64)[:3, 3] for u in ids])
    pg = np.stack([np.asarray(gt[u], dtype=np.float64)[:3, 3] for u in ids])
    R, t = umeyama_rigid(pe, pg)
    return float(np.sqrt((np.linalg.norm(pe @ R.T + t - pg, axis=1) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {}
    for rd in a.runs:
        rd = rd.rstrip("/")
        rows = []
        for p in list_dumps(rd):
            d = load_dump(p)
            pre = {int(u): v.numpy() for u, v in d["poses_pre"].items()}
            pgo = {int(u): v.numpy() for u, v in d["poses_pgo"].items()}
            gt = {int(u): (None if v is None else v.numpy()) for u, v in d["poses_gt"].items()}
            ids = sorted(u for u in pgo if u in pre and gt.get(u) is not None)
            mv = np.array([np.linalg.norm(pgo[u][:3, 3] - pre[u][:3, 3]) for u in ids])
            m = d["meta"]
            rows.append({"cur": int(m["cur_uid"]), "loop": int(m["loop_uid"]), "n_frames": len(ids),
                         "pgo_err_before": m.get("pgo_err_before"), "pgo_err_after": m.get("pgo_err_after"),
                         "ate_pre_cm": 100 * ate(pre, gt, ids), "ate_pgo_cm": 100 * ate(pgo, gt, ids),
                         "move_cur_cm": 100 * float(np.linalg.norm(pgo[int(m["cur_uid"])][:3, 3] - pre[int(m["cur_uid"])][:3, 3])),
                         "move_mean_cm": 100 * float(mv.mean()), "move_max_cm": 100 * float(mv.max())})
            r = rows[-1]
            print("%-28s %4d<->%-4d frames %3d  graph err %8.1f -> %6.1f  ATE so far %.2f -> %.2f cm  moved: current frame %.1f cm, mean %.1f, max %.1f" % (
                os.path.basename(rd), r["cur"], r["loop"], r["n_frames"], r["pgo_err_before"], r["pgo_err_after"],
                r["ate_pre_cm"], r["ate_pgo_cm"], r["move_cur_cm"], r["move_mean_cm"], r["move_max_cm"]))
        out[os.path.basename(rd)] = rows
    json.dump(out, open(os.path.join(a.out, "event_ate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
