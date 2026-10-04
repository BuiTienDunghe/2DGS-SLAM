"""Plan v2 step 11: freeze the selected config as configs/deform/selected_v3.yaml (TUM section only).

usage: python tools/freeze_v3.py CHOSEN RUN_DIR [--out configs/deform/selected_v3.yaml]
The new switches go under `tum:` only (v2 was run on TUM only, user decision 2026-10-02), so Replica
resolves exactly as the frozen A-1. Checks that the YAML, resolved for every dump of RUN_DIR, equals the
configuration tools/run_diag_v2.py used for CHOSEN (FREEZE_OK), and prints the SHA-256.
"""
import argparse
import copy
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
from deform.dump import list_dumps  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def strip(c):
    return {k: v for k, v in c.items() if k not in ("seed", "_path", "seed_from_run", "tum", "replica")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chosen")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "deform", "selected_v3.yaml"))
    a = ap.parse_args()
    if a.chosen.endswith("+conv") or a.chosen not in RD.CONFIGS or a.chosen.startswith("O1"):
        sys.exit(f"not a selectable config: {a.chosen}")
    v1_path = os.path.join(ROOT, "configs", "deform", "selected.yaml")
    with open(v1_path) as f:
        v1 = yaml.safe_load(f)
    over = copy.deepcopy(RD.CONFIGS[a.chosen])
    body = copy.deepcopy(v1)
    tum = body.setdefault("tum", {})
    for sec, vals in over.items():
        vals = copy.deepcopy(vals)
        if sec == "corr":
            if vals.get("eps_d") == "eval":
                vals["eps_d"] = 0.10
            if vals.get("stride") == "half":
                vals["stride"] = 2  # TUM stride 4 -> 2
        tum[sec] = {**tum.get(sec, {}), **vals}
    sha_v1 = hashlib.sha256(open(v1_path, "rb").read()).hexdigest()
    txt = (f"# FROZEN by tools/freeze_v3.py {time.strftime('%Y-%m-%d %H:%M')} (plan v2 §5, TUM-only selection)\n"
           f"# chosen config: {a.chosen} on top of selected.yaml (A-1, sha256 {sha_v1[:12]}...); new switches under tum: only\n"
           + yaml.safe_dump(body, sort_keys=False))
    with open(a.out, "w") as f:
        f.write(txt)
    online = load_deform_config(a.out)
    ok = True
    for p in list_dumps(a.run_dir):
        cfg_slam = torch.load(p, map_location="cpu", weights_only=False)["config"]
        x = dcfg.resolve(online, cfg_slam)
        base = dcfg.resolve(load_deform_config(v1_path), cfg_slam)
        y = RD.make_cfg(base, a.chosen)
        same = json.dumps(strip(x), sort_keys=True, default=str) == json.dumps(strip(y), sort_keys=True, default=str)
        print(os.path.basename(p), cfg_slam["Dataset"]["type"], "s_gate", x["corr"]["s_gate"], "eps_d", x["corr"]["eps_d"],
              "stride", x["corr"]["stride"], "split", x["corr"]["split"], "mode", x["corr"]["mode"],
              "MATCH" if same else "MISMATCH")
        ok &= same
    print("sha256", hashlib.sha256(txt.encode()).hexdigest())
    print("FREEZE_OK" if ok else "FREEZE_MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
