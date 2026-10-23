"""PLAN A2G-P1 (rút gọn) §3.12: re-check invariants I1–I12 on the logs of a P1 run (exit 1 on any violation).
usage: python tools/a2g_p1/check_invariants.py RUN_DIR
"""
import ast
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

CHI2_3 = chi2.ppf(0.99, 3)
ALLOWED = {("candidate", "active"), ("active", "frozen"), ("conflict", "candidate"), ("conflict", "deleted"),
           ("candidate", "deleted")}


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main(run):
    v = []
    fr = jl(os.path.join(run, "pw_frames.jsonl"))
    kf = jl(os.path.join(run, "pw_kf.jsonl"))
    ev = jl(os.path.join(run, "pw_events.jsonl"))
    tab = jl(os.path.join(run, "anchors_kf.jsonl"))
    fl = {int(r["frame_id"]): int(r["is_kf"]) for r in csv.DictReader(open(os.path.join(run, "frame_log.csv")))}
    v += [f"logged invariant event {e['id']}: {e['msg']}" for e in ev if e["type"] == "invariant"]
    prev = None
    for r in fr:
        asg = [tuple(p) for p in r["assign"]]
        if len({s for s, _ in asg}) != len(asg) or len({j for _, j in asg}) != len(asg):
            v.append(f"I1 frame {r['k']}")
        cand = {(c["s"], c["j"], c.get("at", "T_pred")): c for c in r["cand"]}
        added = {tuple(p) for p in r.get("added", [])}
        for s, j in asg:
            c = cand.get((s, j, "T_p" if (s, j) in added else "T_pred"))
            if c is None or c["nis"] > CHI2_3 + 1e-9:
                v.append(f"I2 frame {r['k']} pair {(s, j)}")
            elif c["state"] == "conflict" and c.get("blocked"):
                v.append(f"I4 frame {r['k']} pair {(s, j)}")
        states = {(c["s"], c["j"]): c["state"] for c in r["cand"]}
        for s, j in r["pose"]:
            if states.get((s, j)) not in ("active", "frozen"):
                v.append(f"I3 frame {r['k']} pose pair {(s, j)} state {states.get((s, j))}")
        src = r["init"]["source"]
        if src == "jc_jump" and r["pose"]:
            v.append(f"I3 frame {r['k']} jump but pose pairs")
        if src == "plane":
            jc = r["jc"]
            if not (jc["rank"] == 6 and jc["jump"] <= chi2.ppf(0.99, 6)
                    and (jc["dof_shape"] <= 0 or jc["shape"] <= chi2.ppf(0.99, jc["dof_shape"]))):
                v.append(f"I10 frame {r['k']}")
        if prev is not None and r["ghash"] != prev["ghash"] and not fl.get(prev["k"], 0):
            v.append(f"I6 geometry changed between frames {prev['k']} and {r['k']} (no keyframe)")
        prev = r
    for r in kf:
        for a, frm, to in r.get("transitions", []):
            if (frm, to) not in ALLOWED:
                v.append(f"I11 kf {r['k']} anchor {a} {frm}->{to}")
    for e in ev:
        if e["type"] == "merge" and "conflict" in (e["keep_par"]["state"], e["drop_par"]["state"]):
            v.append(f"I7 merge {e['keep']},{e['drop']} with conflict")
    frozen = {}
    for r in tab:
        ids = {a["id"]: a for a in r["anchors"]}
        for a in r["anchors"]:
            P = np.array(a["P"])
            if not np.allclose(P, P.T, atol=1e-12) or np.min(np.linalg.eigvalsh(0.5 * (P + P.T))) <= 0:
                v.append(f"I8 anchor {a['id']} at {r['frame_id']}")
            if a["primary_id"] != a["id"]:
                p = ids.get(a["primary_id"])
                if p is None or p["primary_id"] != p["id"]:
                    v.append(f"I9 anchor {a['id']} -> {a['primary_id']} at {r['frame_id']}")
            if a["state"] == "frozen":
                key = json.dumps([a["n_w"], a["o"], a["P"]])
                if a["id"] in frozen and frozen[a["id"]] != key:
                    v.append(f"I5 anchor {a['id']} changed after freezing (frame {r['frame_id']})")
                frozen.setdefault(a["id"], key)
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils", "pw")
    for f in sorted(os.listdir(root)):
        if f.endswith(".py"):
            names = {n.id for n in ast.walk(ast.parse(open(os.path.join(root, f)).read())) if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(ast.parse(open(os.path.join(root, f)).read())) if isinstance(n, ast.Attribute)}
            bad = names & {"gt_pose", "groundtruth", "traj_gt", "gt"}
            if bad:
                v.append(f"I12 {f} uses {sorted(bad)}")
    print(f"{run}: {len(v)} invariant violations over {len(fr)} frames, {len(kf)} keyframes")
    for x in v[:50]:
        print("  " + x)
    json.dump(dict(n_violations=len(v), violations=v), open(os.path.join(run, "invariants.json"), "w"), indent=1)
    return len(v)


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
