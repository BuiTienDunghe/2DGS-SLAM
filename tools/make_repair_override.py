"""Apply the pre-registered repair rule (results_exp/reports/P3_repair_rule.md) to P2 diagnostics.

usage: python tools/make_repair_override.py P2_metrics.json OUT.yaml   -> exit 2 on the bug signature
"""
import json
import math
import sys

import numpy as np
import yaml


def main(p2, out):
    ev = json.load(open(p2))["events"]
    res, bug = {}, []
    for ds in ("tum", "replica"):
        rows = [e for e in ev if e["scene"].startswith(ds) and e.get("S_factors")]
        if not rows:
            continue
        sf = [e["S_factors"] for e in rows]
        rD50 = float(np.median([s["rD_mm"]["p50"] for s in sf if s["rD_mm"]]))
        fn50 = float(np.median([s["f_n"]["p50"] for s in sf if s["f_n"]]))
        scored = float(np.median([s["frac_scored"] for s in sf]))
        if rD50 > 100 or fn50 < 0.1 or scored < 0.3:
            bug.append({"dataset": ds, "rD_p50_mm": rD50, "f_n_p50": fn50, "frac_scored": scored})
        sD = math.ceil(float(np.median([s["rD_mm"]["p75"] for s in sf if s["rD_mm"]]))) / 1000.0
        sn = math.ceil(100 * float(np.median([s["en"]["p75"] for s in sf if s["en"]]))) / 100.0
        res[ds] = {"reliability": {"sigma_D": sD, "sigma_n": sn}}
        print(f"{ds}: events={len(rows)} rD p50={rD50:.1f} mm, f_n p50={fn50:.3f}, scored={scored:.2f} "
              f"-> sigma_D={sD * 1e3:.0f} mm, sigma_n={sn:.2f}")
    if bug:
        print("BUG_SIGNATURE", json.dumps(bug))
        sys.exit(2)
    with open(out, "w") as f:
        f.write("# repair round (P3_repair_rule.md): sigma = median over tuning events of p75 of r_D / e_n\n")
        yaml.safe_dump(res, f, sort_keys=False)
    print("WROTE", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
