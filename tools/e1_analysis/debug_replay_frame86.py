import os, numpy as np, torch
"""Debug for the Q4 replay check: matching candidates at frame 86 (first frame where replay != run).
usage: python tools/e1_analysis/debug_replay_frame86.py"""
src = open("tools/e1_analysis/replay_anchors.py").read()
src = src.replace("    T_prev = w2c[prev]\n", "    T_prev = w2c[prev]\n    if fid == 86: STOP\n")
src = src.split("match_flags = ")[0]
g = {"__file__": os.path.abspath("tools/e1_analysis/replay_anchors.py"), "__name__": "dbg"}
try:
    exec(compile(src, "replay_dbg", "exec"), g)
except NameError:
    pass
amap, planes, T = g["amap"], g["planes"], g["w2c"][84]
from utils.plane_anchor import plane_w2c
R, t = T[:3, :3], T[:3, 3]
for aid, a in amap.anchors.items():
    n_c, d_c = plane_w2c(a.n_w, a.d_w, T)
    for pi, p in enumerate(planes):
        c = float(np.clip(p.n.numpy() @ n_c, -1, 1))
        if c < amap.max_cos: continue
        dist = abs(p.d - d_c)
        ov = amap._overlap(a, p, R, t)
        cost = np.arccos(c) / amap.max_angle + dist / amap.max_dist + (1 - ov)
        print(f"plane {pi} anchor {aid} active {a.active} nobs {a.n_obs_kf} ang {np.degrees(np.arccos(c)):.2f} dist {dist*100:.2f}cm ov {ov:.3f} cost {cost:.4f} pass {dist<=amap.max_dist and ov>=amap.min_overlap}")
print("pairs", amap.match(planes, torch.from_numpy(T)))
