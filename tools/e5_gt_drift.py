"""plan v6 quick, E5 cross-check: relative drift between the candidate keyframe and the requesting frame of every
revisit dump, from the ground-truth poses (CPU only).

usage: python tools/e5_gt_drift.py --out DIR E5_RUN_DIR
E = (Tgt_c^-1 Tgt_r)^-1 (Test_c^-1 Test_r): the error of the estimated relative pose candidate -> requesting frame.
Its translation norm / rotation angle are the pose inconsistency a loop between the two would have to remove.
Writes DIR/e5_gt_drift.json.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("run")
    a = ap.parse_args()
    out = {}
    for p in sorted(glob.glob(os.path.join(a.run.rstrip("/"), "revisit_dumps", "rv_*.pt"))):
        d = torch.load(p, map_location="cpu", weights_only=False)
        req = d["request"]
        c = req.get("cand_kf")
        if c is None or req.get("gt_c2w") is None or d["poses_gt"].get(int(c)) is None:
            continue
        mat = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
        rel_est = np.linalg.inv(mat(d["poses_pre"][int(c)])) @ mat(req["pose_c2w"])
        rel_gt = np.linalg.inv(mat(d["poses_gt"][int(c)])) @ mat(req["gt_c2w"])
        E = np.linalg.inv(rel_gt) @ rel_est
        ang = float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(E[:3, :3] - np.eye(3)) / (2 * np.sqrt(2))))))
        out[req["tag"]] = {"frame": int(req["uid"]), "cand_kf": int(c), "rel_err_t_mm": 1e3 * float(np.linalg.norm(E[:3, 3])), "rel_err_rot_deg": ang,
                           "dist_gt_m": float(np.linalg.norm(rel_gt[:3, 3]))}
        print(f"{req['tag']} frame {req['uid']} cand kf {c}: relative pose error {out[req['tag']]['rel_err_t_mm']:.0f} mm / {ang:.2f} deg (cameras {out[req['tag']]['dist_gt_m']:.2f} m apart)")
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "e5_gt_drift.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
