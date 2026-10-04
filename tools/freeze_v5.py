"""Plan v4 step C: freeze configs/deform/selected_v5.yaml = selected_v4.yaml + the v4 switches.

usage: python tools/freeze_v5.py RUN_DIR [RUN_DIR ...] [--out configs/deform/selected_v5.yaml]
Adds to selected_v4: corr.select hash (+ explicit hash_salt), det.render / det.torch true, pose_sync.anchor
prior. Checks for every dump that the resolved config equals the K1-K4 candidate (configs/deform/cand_v5.yaml)
and the step-B K3 config "F1o-hash" of tools/run_diag_v2.py -> FREEZE_OK; prints the SHA-256 and the diff
against selected_v4.yaml. selected.yaml, selected_v3.yaml, selected_v4.yaml are not touched.
"""
import argparse
import difflib
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import torch  # noqa: E402
import yaml  # noqa: E402

import run_diag_v2 as RD  # noqa: E402
from deform import config as dcfg  # noqa: E402
from deform.correspondences import HASH_SALT  # noqa: E402
from deform.dump import list_dumps  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def strip(c):
    return {k: v for k, v in c.items() if k not in ("seed", "_path", "seed_from_run", "tum", "replica")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "deform", "selected_v5.yaml"))
    a = ap.parse_args()
    v4_path = os.path.join(ROOT, "configs", "deform", "selected_v4.yaml")
    with open(v4_path) as f:
        body = yaml.safe_load(f)
    body.setdefault("corr", {})
    body["corr"]["select"] = "hash"
    body["corr"]["hash_salt"] = int(HASH_SALT)
    body["det"] = {"render": True, "torch": True}
    body["pose_sync"] = {**body.get("pose_sync", {}), "anchor": "prior"}
    sha_v4 = hashlib.sha256(open(v4_path, "rb").read()).hexdigest()
    txt = (f"# FROZEN by tools/freeze_v5.py {time.strftime('%Y-%m-%d %H:%M')} (handoffPlan_v4_determinism step C)\n"
           f"# = selected_v4.yaml (sha256 {sha_v4[:12]}...) + corr.select hash (salt {HASH_SALT:#x}) + det.render/torch\n"
           f"# + pose_sync.anchor prior (A4 gauge anchor on the prior-fixed frame); all datasets\n"
           + yaml.safe_dump(body, sort_keys=False))
    with open(a.out, "w") as f:
        f.write(txt)
    online = load_deform_config(a.out)
    cand = load_deform_config(os.path.join(ROOT, "configs", "deform", "cand_v5.yaml"))
    ok = True
    for rd in a.runs:
        for p in list_dumps(rd):
            d = torch.load(p, map_location="cpu", weights_only=False)
            r_new = strip(dcfg.resolve(online, d["config"]))
            r_cand = strip(dcfg.resolve(cand, d["config"]))
            base = dcfg.resolve(load_deform_config(os.path.join(ROOT, "configs", "deform", "selected.yaml")), d["config"])
            base["solver"]["tol_mode"] = "grad"
            r_k3 = strip(RD.make_cfg(base, "F1o-hash"))
            r_k3["solver"]["tol_mode"] = "grad"
            same_c, same_k = r_new == r_cand, r_new == r_k3
            ok &= same_c and same_k
            print(os.path.basename(p), "== cand_v5:", same_c, "== K3 F1o-hash:", same_k)
            if not (same_c and same_k):
                for k in r_new:
                    if r_new[k] != r_cand.get(k) or r_new[k] != r_k3.get(k):
                        print("   differs:", k, json.dumps(r_new[k], default=str)[:200], "| cand", json.dumps(r_cand.get(k), default=str)[:200],
                              "| k3", json.dumps(r_k3.get(k), default=str)[:200])
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"sha256 {sha}")
    print("".join(difflib.unified_diff(open(v4_path).read().splitlines(True), open(a.out).read().splitlines(True),
                                       "selected_v4.yaml", "selected_v5.yaml")))
    print("FREEZE_OK" if ok else "FREEZE_MISMATCH")


if __name__ == "__main__":
    main()
