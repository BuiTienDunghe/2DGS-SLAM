"""G1 sanity: re-apply the rigid correction offline from the "pre" part of every dump and compare
with the "post" part written online.  usage: python tools/replay_rigid.py RUN_DIR [RUN_DIR ...]
Pass: max ||d mu|| < 1e-5 m and quaternion angle < 1e-4 rad for every Gaussian of every event.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402  (sets sys.path)
import torch  # noqa: E402

from deform.dump import list_dumps, load_dump  # noqa: E402
from deform.render_utils import DEV  # noqa: E402
from deform.rigid import quat_angle, replay  # noqa: E402

TOL_XYZ, TOL_ROT = 1e-5, 1e-4


def main(run_dirs):
    # Runs from 2026-10-02 01:40 on use full fp32 in the backend (allow_tf32=False); set REPLAY_TF32=1 only for
    # dumps of the earlier TF32 runs (smoke_dump, smoke2_dump, the first rigid_s0).
    tf32 = os.environ.get("REPLAY_TF32", "0") == "1"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    print(f"replay device={DEV} allow_tf32={tf32}")
    rows, ok_all = [], True
    for rd in run_dirs:
        for p in list_dumps(rd):
            d = load_dump(p)
            with torch.no_grad():
                xyz, rot, _ = replay(d)
                dx = (xyz - d["gauss_post"]["xyz"].to(DEV)).norm(dim=-1)
                da = quat_angle(rot, d["gauss_post"]["rot"].to(DEV))
            r = {"run": rd, "event": os.path.basename(p), "N": int(xyz.shape[0]),
                 "max_dxyz_m": float(dx.max()) if dx.numel() else 0.0,
                 "max_dangle_rad": float(da.max()) if da.numel() else 0.0,
                 "moved_frac": float(((d["gauss_post"]["xyz"] - d["gauss_pre"]["xyz"]).norm(dim=-1) > 1e-6).float().mean())}
            r["pass"] = r["max_dxyz_m"] < TOL_XYZ and r["max_dangle_rad"] < TOL_ROT
            ok_all &= r["pass"]
            rows.append(r)
            print(json.dumps(r))
    print("REPLAY_ALL_PASS" if ok_all and rows else ("REPLAY_FAIL" if rows else "NO_DUMPS"))
    return rows, ok_all


if __name__ == "__main__":
    main(sys.argv[1:])
