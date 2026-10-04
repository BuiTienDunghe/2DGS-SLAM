"""Plan v3 step D: freeze the selected config as configs/deform/selected_v4.yaml.

usage: python tools/freeze_v4.py CHOSEN RUN_DIR [RUN_DIR ...] [--out configs/deform/selected_v4.yaml]
CHOSEN is A-1c or one of the v3 candidates (F6, F1o, F1t, F1w, F1t+F6, F1o+F6). The file is selected.yaml
(A-1) + `solver.tol_mode: grad` + the candidate's switches, applied to every dataset (v3 selected on TUM and
Replica). Checks that the YAML resolved for every dump equals the configuration used in the v3 rerun
(tools/run_diag_v2.py --solver-mode grad) -> FREEZE_OK, and prints the SHA-256. selected.yaml and
selected_v3.yaml are not touched.
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
CANDIDATES = ("A-1c", "F6", "F1o", "F1t", "F1w", "F1t+F6", "F1o+F6")


def strip(c):
    return {k: v for k, v in c.items() if k not in ("seed", "_path", "seed_from_run", "tum", "replica")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chosen", choices=CANDIDATES)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "deform", "selected_v4.yaml"))
    a = ap.parse_args()
    v1_path = os.path.join(ROOT, "configs", "deform", "selected.yaml")
    with open(v1_path) as f:
        body = yaml.safe_load(f)
    body["solver"] = {"tol_mode": "grad"}
    if a.chosen != "A-1c":
        for sec, vals in copy.deepcopy(RD.CONFIGS[a.chosen]).items():
            body[sec] = {**body.get(sec, {}), **vals}
    sha_v1 = hashlib.sha256(open(v1_path, "rb").read()).hexdigest()
    txt = (f"# FROZEN by tools/freeze_v4.py {time.strftime('%Y-%m-%d %H:%M')} (handoffPlan_v3_solver step D)\n"
           f"# chosen: {a.chosen} = selected.yaml (A-1, sha256 {sha_v1[:12]}...) + solver tol_mode grad"
           f"{'' if a.chosen == 'A-1c' else ' + ' + a.chosen + ' switches'}; all datasets\n"
           + yaml.safe_dump(body, sort_keys=False))
    with open(a.out, "w") as f:
        f.write(txt)
    online = load_deform_config(a.out)
    ok = True
    for rd in a.runs:
        for p in list_dumps(rd):
            cfg_slam = torch.load(p, map_location="cpu", weights_only=False)["config"]
            x = dcfg.resolve(online, cfg_slam)
            base = dcfg.resolve(load_deform_config(v1_path), cfg_slam)
            base["solver"]["tol_mode"] = "grad"
            y = base if a.chosen == "A-1c" else RD.make_cfg(base, a.chosen)
            y["solver"]["tol_mode"] = "grad"
            same = json.dumps(strip(x), sort_keys=True, default=str) == json.dumps(strip(y), sort_keys=True, default=str)
            print(os.path.basename(p), cfg_slam["Dataset"]["type"], "solver", x["solver"]["tol_mode"],
                  "s_gate", x["corr"]["s_gate"], "max_per_node", x["corr"]["max_per_node"], "MATCH" if same else "MISMATCH")
            ok &= same
    print("sha256", hashlib.sha256(txt.encode()).hexdigest())
    print("FREEZE_OK" if ok else "FREEZE_MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
