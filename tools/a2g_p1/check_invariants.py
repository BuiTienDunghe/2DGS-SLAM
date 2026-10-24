"""PLAN A2G-P1 core §3.8 (also reads v2 / v2.1 logs): re-check invariants on the logs of a P1 v2 run (exit 1 on any violation).
I1-I4, I6, I7, I9-I12 as v1; I5: (n, o, P) of every anchor unchanged since creation; I8: V symmetric positive
definite, Gamma >= 0; I11 without frozen; I13: b_hat samples accepted only after the test (logged m <= chi2_3);
I14: merges between anchors with fresh data; I15 checked online (logged as an invariant event if violated);
I16 (v2.1): every accepted b_hat sample touched the support region; every merge had common cells.
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
ALLOWED = {("candidate", "active"), ("candidate", "deleted")}          # PLAN core §3.6
WB = 30


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
        cand = {(c["s"], c["j"], c.get("at", "T_gate")): c for c in r["cand"]}
        added = {tuple(p) for p in r.get("added", [])}
        for s, j in asg:
            c = cand.get((s, j, "T_p" if (s, j) in added else "T_gate"))
            if c is None or c["nis"] > CHI2_3 + 1e-9:
                v.append(f"I2 frame {r['k']} pair {(s, j)}")
            elif c["state"] == "conflict" and c.get("blocked"):
                v.append(f"I4 frame {r['k']} pair {(s, j)}")
        states = {(c["s"], c["j"]): c["state"] for c in r["cand"]}
        for s, j in r["pose"]:
            if states.get((s, j)) != "active":
                v.append(f"I3 frame {r['k']} pose pair {(s, j)} state {states.get((s, j))}")
        src = r["init"]["source"]
        if src == "jc_jump" and r["pose"]:
            v.append(f"I3 frame {r['k']} jump but pose pairs")
        if src == "plane":
            jc = r["jc"]
            if not (jc["rank"] == 6 and jc["jump"] <= chi2.ppf(0.99, 6)
                    and (jc["dof_shape"] <= 0 or jc["shape"] <= chi2.ppf(0.99, jc["dof_shape"]))):
                v.append(f"I10 frame {r['k']}")
        for p in r.get("pairs", []):
            b = p.get("bias_sample") or {}
            if b.get("accepted"):
                if b.get("m") is None or b["m"] > CHI2_3 + 1e-9:
                    v.append(f"I6 frame {r['k']} accepted sample above chi2 {(p['s'], p['j'])}")
                if "unambiguous" in p and not p["unambiguous"]:
                    v.append(f"I6 frame {r['k']} sample from an ambiguous pair {(p['s'], p['j'])}")
        if prev is not None and r["ghash"] != prev["ghash"] and not fl.get(prev["k"], 0):
            v.append(f"I6 geometry changed between frames {prev['k']} and {r['k']} (no keyframe)")
        prev = r
    for r in kf:
        for a, frm, to in r.get("transitions", []):
            if (frm, to) not in ALLOWED:
                v.append(f"I11 kf {r['k']} anchor {a} {frm}->{to}")
        if "Gamma" in r and min(r["Gamma"]) < 0:
            v.append(f"I8 Gamma negative at kf {r['k']}")
    for e in ev:
        if e["type"] != "merge":
            continue
        if "conflict" in (e["keep_par"]["state"], e["drop_par"]["state"]):
            v.append(f"I7 merge {e['keep']},{e['drop']} with conflict")
        if "kt" in e and min(e["tau"]) < e["kt"] - WB:
            v.append(f"I7 merge {e['keep']},{e['drop']} without fresh data")
        if "common_cells" in e and e["common_cells"] < 1:
            v.append(f"I7 merge {e['keep']},{e['drop']} without common cells")
    first = {}
    for r in tab:
        ids = {a["id"]: a for a in r["anchors"]}
        for a in r["anchors"]:
            V = np.array(a.get("V", np.eye(3)))
            if not np.allclose(V, V.T, atol=1e-12) or np.min(np.linalg.eigvalsh(0.5 * (V + V.T))) <= 0:
                v.append(f"I5 V of anchor {a['id']} at {r['frame_id']}")
            if a["primary_id"] != a["id"]:
                p = ids.get(a["primary_id"])
                if p is None or p["primary_id"] != p["id"]:
                    v.append(f"I9 anchor {a['id']} -> {a['primary_id']} at {r['frame_id']}")
            key = json.dumps([a["n_w"], a["o"], a["P"]])
            if a["id"] in first and first[a["id"]] != key:
                v.append(f"I4 anchor {a['id']} parameters changed (frame {r['frame_id']})")
            first.setdefault(a["id"], key)
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils", "pw")
    for f in sorted(os.listdir(root)):
        if f.endswith(".py"):
            tree = ast.parse(open(os.path.join(root, f)).read())
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
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
