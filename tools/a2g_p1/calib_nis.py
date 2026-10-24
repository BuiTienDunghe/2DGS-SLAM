"""PLAN core §6: angle / distance NIS of keyframe re-matches (T_opt, no Q_pred) of unambiguous fresh pairs."""
import json, sys
import numpy as np
run = sys.argv[1]
ang, dd = [], []
for l in open(f"{run}/pw_kf.jsonl"):
    r = json.loads(l)
    for x in r.get("reassign", []):
        if x.get("unambiguous") and x.get("fresh"):
            ang.append(x["nis_ang"]); dd.append(x["nis_d"])
print(json.dumps(dict(run=run, n=len(ang), mean_ang=float(np.mean(ang)) if ang else None,
                      mean_d=float(np.mean(dd)) if dd else None)))
