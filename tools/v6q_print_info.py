"""plan v6 quick: print the summary numbers of tools/v6q_summary.py --json in a compact form (helper for the report)."""
import json
import statistics as S
import sys


def r(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, list):
        return [r(x) for x in v]
    if isinstance(v, dict):
        return {k: r(x) for k, x in v.items()}
    return v


info = json.load(open(sys.argv[1]))
for k, d in info.items():
    if k == "e3_hash_groups":
        for g, v in d.items():
            print(k, g, len(v), r([min(v), S.median(v), max(v)]))
        continue
    if k in ("e3_offsets", "e3_offsets_max"):
        print(k, len(d), r([min(d), S.median(d), max(d)]))
        continue
    for kk, v in d.items():
        if kk == "funnel":
            for x in v:
                print(k, kk, r(x))
        else:
            print(k, kk, r(v) if kk not in ("kfmedian",) else [round(x, 3) for x in v])
