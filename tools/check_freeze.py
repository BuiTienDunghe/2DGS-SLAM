"""Freeze check: selected.yaml resolved per dataset == the configuration P3 used for the chosen config.

usage: python tools/check_freeze.py configs/deform/selected.yaml configs/deform/repair_override.yaml CHOSEN RUN_DIR...
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import torch  # noqa: E402
import yaml  # noqa: E402

import glob  # noqa: E402

from deform import config as dcfg  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402


def list_dumps(rd):
    return sorted(glob.glob(os.path.join(rd, "loop_dumps", "event_*.pt")))


def grid_entry(name):
    """'B-10' -> ('B', 10.0, 1.0) (candidates only; the noCon sanity configs are never frozen)."""
    var, wp = name.split("-")
    return var, float(wp), 1.0


def strip(c):
    return {k: v for k, v in c.items() if k not in ("seed", "_path", "mode", "variant", "seed_from_run", "tum", "replica")}


def main(sel, ovr, chosen, runs):
    var, wp, wc = grid_entry(chosen)
    online = load_deform_config(sel)
    override = yaml.safe_load(open(ovr)) or {}
    ok = online["mode"] == "deform" and online["variant"] == var
    for rd in runs:
        d = list_dumps(rd)
        if not d:
            continue
        cfg = torch.load(d[0], map_location="cpu", weights_only=False)["config"]
        a = dcfg.resolve(online, cfg)
        b = dcfg.resolve(override, cfg)
        b["energy"]["w_p"], b["energy"]["w_con"] = wp, wc
        same = json.dumps(strip(a), sort_keys=True, default=str) == json.dumps(strip(b), sort_keys=True, default=str)
        print(cfg["Dataset"]["type"], "reliability", a["reliability"]["sigma_D"], a["reliability"]["sigma_n"],
              "energy", a["energy"]["w_p"], a["energy"]["w_con"], "MATCH" if same else "MISMATCH")
        ok &= same
    print("FREEZE_OK" if ok else "FREEZE_MISMATCH")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]) else 1)
