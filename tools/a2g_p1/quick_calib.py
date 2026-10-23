"""PLAN A2G-P1 (rút gọn) §7: kappa_t, kappa_r from the E1c snt_far trajectory, written into p1_common.yaml.

kappa_t = unaligned translation error at the end of the sequence / total path length (GT)
kappa_r = unaligned rotation error at the end / total rotation (GT)
Uses test-sequence data (E1c snt_far): this must be stated in the report.
usage: python tools/a2g_p1/quick_calib.py --e1c RUN_DIR [--write configs/tum/a2g_p1/p1_common.yaml]
"""
import argparse
import json
import re

import numpy as np
from scipy.spatial.transform import Rotation


def load_tum(path):
    P = []
    for ln in open(path):
        v = ln.split()
        if len(v) != 8:
            continue
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([float(x) for x in v[4:8]]).as_matrix()
        T[:3, 3] = [float(x) for x in v[1:4]]
        P.append(T)
    return P


def ang(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1c", required=True)
    ap.add_argument("--write", default=None)
    a = ap.parse_args()
    est = load_tum(f"{a.e1c}/traj_est_all.txt")
    gt = load_tum(f"{a.e1c}/traj_gt_all.txt")
    s = sum(np.linalg.norm(gt[i + 1][:3, 3] - gt[i][:3, 3]) for i in range(len(gt) - 1))
    phi = sum(ang(gt[i][:3, :3].T @ gt[i + 1][:3, :3]) for i in range(len(gt) - 1))
    e_t = float(np.linalg.norm(est[-1][:3, 3] - gt[-1][:3, 3]))
    e_r = ang(gt[-1][:3, :3].T @ est[-1][:3, :3])
    out = dict(kappa_t=e_t / s, kappa_r=e_r / phi, end_err_t_m=e_t, end_err_r_deg=float(np.degrees(e_r)),
               path_m=float(s), rot_deg=float(np.degrees(phi)), n_frames=len(gt), source=a.e1c,
               note="from the E1c run on the test sequence snt_far (mixes test data)")
    print(json.dumps(out, indent=1))
    if a.write:
        t = open(a.write).read()
        t = re.sub(r"kappa_t: *[^\s#]+", f"kappa_t: {out['kappa_t']:.6g}", t, count=1)
        t = re.sub(r"kappa_r: *[^\s#]+", f"kappa_r: {out['kappa_r']:.6g}", t, count=1)
        open(a.write, "w").write(t)
        print(f"written kappa_t, kappa_r into {a.write}")


if __name__ == "__main__":
    main()
