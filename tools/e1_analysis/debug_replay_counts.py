"""Debug for the Q4 replay check: first frame where anchor counts differ between replay and run.
usage: python tools/e1_analysis/debug_replay_counts.py"""
import os, sys
src = open("tools/e1_analysis/replay_anchors.py").read()
# record anchor counts after each frame
src = src.replace("    prev = fid\n\n# ----", "    records[-1]['na'] = len(amap.anchors); records[-1]['naa'] = amap.n_active()\n    prev = fid\n\n# ----")
src = src.split("match_flags = ")[0]
g = {"__file__": os.path.abspath("tools/e1_analysis/replay_anchors.py"), "__name__": "dbg"}
exec(compile(src, "replay_dbg", "exec"), g)
records, flog = g["records"], g["flog"]
for rec in records:
    r = flog[rec["fid"]]
    if "na" in rec and (int(r["n_anchors"]) != rec["na"] or int(r["n_anchors_active"]) != rec["naa"]):
        print("first anchor-count divergence at", rec["fid"], "kf", r["is_kf"], "run", r["n_anchors"], r["n_anchors_active"], "replay", rec["na"], rec["naa"])
        break
for rec in records[:50]:
    r = flog[rec["fid"]]
    if r["is_kf"] == "1": print(rec["fid"], "run na/naa", r["n_anchors"], r["n_anchors_active"], "rep", rec.get("na"), rec.get("naa"), "nm run/rep", r["n_matched"], rec["n_matched"])
