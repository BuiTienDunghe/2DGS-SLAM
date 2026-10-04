"""Plan v5 step A4: freeze configs/deform/selected_v6.yaml = selected_v5.yaml + layers.mode birth + pose_sync.regen_odom.

usage: python tools/freeze_v6.py RUN_DIR [RUN_DIR ...] [--out configs/deform/selected_v6.yaml]
Checks for every dump that the resolved config equals the tested candidate configs/deform/cand_v6.yaml -> FREEZE_OK;
prints the SHA-256 and the diff against selected_v5.yaml. Older frozen files are not touched.
"""
import argparse
import difflib
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import torch  # noqa: E402
import yaml  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform.dump import list_dumps  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def strip(c):
    return {k: v for k, v in c.items() if k not in ("seed", "_path", "seed_from_run", "tum", "replica")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "deform", "selected_v6.yaml"))
    a = ap.parse_args()
    v5_path = os.path.join(ROOT, "configs", "deform", "selected_v5.yaml")
    body = yaml.safe_load(open(v5_path))
    body["layers"] = {"mode": "birth"}
    body["pose_sync"] = {**body.get("pose_sync", {}), "regen_odom": True}
    sha_v5 = hashlib.sha256(open(v5_path, "rb").read()).hexdigest()
    txt = (f"# FROZEN by tools/freeze_v6.py {time.strftime('%Y-%m-%d %H:%M')} (handoffPlan_v5_persistence step A4)\n"
           f"# = selected_v5.yaml (sha256 {sha_v5[:12]}...) + layers.mode birth (A1) + pose_sync.regen_odom (A2); all datasets\n"
           + yaml.safe_dump(body, sort_keys=False))
    with open(a.out, "w") as f:
        f.write(txt)
    online = load_deform_config(a.out)
    cand = load_deform_config(os.path.join(ROOT, "configs", "deform", "cand_v6.yaml"))
    ok = True
    for rd in a.runs:
        for p in list_dumps(rd):
            d = torch.load(p, map_location="cpu", weights_only=False)
            r_new, r_cand = strip(dcfg.resolve(online, d["config"])), strip(dcfg.resolve(cand, d["config"]))
            same = r_new == r_cand
            ok &= same
            print(os.path.basename(p), "== cand_v6:", same)
            if not same:
                for k in r_new:
                    if r_new[k] != r_cand.get(k):
                        print("   differs:", k, json.dumps(r_new[k], default=str)[:160], "|", json.dumps(r_cand.get(k), default=str)[:160])
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"sha256 {sha}")
    print("".join(difflib.unified_diff(open(v5_path).read().splitlines(True), open(a.out).read().splitlines(True),
                                       "selected_v5.yaml", "selected_v6.yaml")))
    print("FREEZE_OK" if ok else "FREEZE_MISMATCH")


if __name__ == "__main__":
    main()
