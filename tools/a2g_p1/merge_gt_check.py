"""Objective check of suspected false merges with GT poses (analysis only; utils/pw never reads GT).

Each anchor is taken at the keyframe it was created (anchors_kf row of t_ref, before any EKF update), mapped to
the camera of t_ref with the estimated pose and back to the world with the GT pose of t_ref. Two anchors of the
same physical surface then agree up to the creation-time measurement noise; a false merge shows an angle or an
offset well above it.
usage: python tools/a2g_p1/merge_gt_check.py RUN_DIR
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import jsonl, load_tum  # noqa: E402

run = sys.argv[1]
est = load_tum(f"{run}/traj_est_all.txt")
gt = load_tum(f"{run}/traj_gt_all.txt")
fids = [int(r["frame_id"]) for r in csv.DictReader(open(f"{run}/frame_log.csv"))]
idx = {f: i for i, f in enumerate(fids)}
tab = {r["frame_id"]: {a["id"]: a for a in r["anchors"]} for r in jsonl(f"{run}/anchors_kf.jsonl")}


def at_creation(aid):
    for f in sorted(tab):
        if aid in tab[f]:
            return f, tab[f][aid]
    return None, None


def gt_world(aid):
    f, a = at_creation(aid)
    i = idx[f]
    Te, Tg = est[i], gt[i]                       # camera->world (TUM files store c2w)
    n_w, o_w = np.array(a["n_w"]), np.array(a["o"])
    n_w = n_w / np.linalg.norm(n_w)
    n_c = Te[:3, :3].T @ n_w
    o_c = Te[:3, :3].T @ (o_w - Te[:3, 3])
    return f, Tg[:3, :3] @ n_c, Tg[:3, :3] @ o_c + Tg[:3, 3]


out = []
for e in jsonl(f"{run}/pw_events.jsonl"):
    if e["type"] != "merge":
        continue
    fk, nk, ok = gt_world(e["keep"])
    fd, nd, od = gt_world(e["drop"])
    nk, nd = nk / np.linalg.norm(nk), nd / np.linalg.norm(nd)
    ang = float(np.degrees(np.arccos(np.clip(abs(nk @ nd), -1, 1))))
    dist = float(nk @ (od - ok))
    est_ang, est_dist = e["angle_deg"], 100 * e["dist_at_o_drop"]
    out.append(dict(k=e["k"], keep=e["keep"], drop=e["drop"], created=[fk, fd], est_angle_deg=est_ang,
                    est_dist_cm=est_dist, gt_angle_deg=ang, gt_dist_cm=100 * dist,
                    suspect=est_ang > 3 or abs(est_dist) > 3, gt_false=ang > 3 or abs(100 * dist) > 3))
    print(f"k {e['k']}: keep {e['keep']} (created {fk}) drop {e['drop']} (created {fd}) | estimated {est_ang:.2f} deg "
          f"{est_dist:+.2f} cm | GT {ang:.2f} deg {100 * dist:+.2f} cm -> "
          f"{'FALSE MERGE' if out[-1]['gt_false'] else 'same surface'}")
json.dump(out, open(f"{run}/merge_gt_check.json", "w"), indent=1)
