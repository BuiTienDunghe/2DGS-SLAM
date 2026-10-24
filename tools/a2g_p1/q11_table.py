"""Q11 table (8 drift kinds x 5 seeds) for one synthetic noise model; writes JSON + markdown.
usage: python tools/a2g_p1/q11_table.py SEG_ERR OUT_PREFIX     (SEG_ERR 1: per-segment plane error at the default
floors, 0: depth noise only as in the plan text)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import pw_sim  # noqa: E402

pw_sim.SEG_ERR = float(sys.argv[1])
out = sys.argv[2]
rows = []
for kind in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
    for seed in range(5):
        r = pw_sim.run_case(kind, seed)
        r.pop("wrong_log", None)
        rows.append(r)
        print(kind, seed, round(r["nis_mean"], 2), round(r["pct_correct"], 2), r["merges"], r["false_merges"],
              r["conflicts"], r["active_per_plane"], flush=True)
json.dump(rows, open(out + ".json", "w"), indent=1)
L = ["| Kiểu | NIS IC (dữ liệu mới), trung bình 5 seed [min–max] | % gán đúng [min–max] | Gộp | Gộp nhầm | Xung đột | "
     "Seed có mặt thừa/thiếu anchor | Γ cuối (σ θ°, σ d mm mỗi frame) | Nhầm lẫn chính |", "|---|---|---|---|---|---|---|---|---|"]
for kind in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
    rs = [r for r in rows if r["kind"] == kind]
    nis = [r["nis_mean"] for r in rs]
    pc = [r["pct_correct"] for r in rs]
    g = np.mean([r["gamma"] for r in rs], 0)
    conf = {}
    for r in rs:
        for k, v in r["confusion"].items():
            kk = k.split("(")[0]
            conf[kk] = conf.get(kk, 0) + v
    top = ", ".join(f"{k} {v}" for k, v in sorted(conf.items(), key=lambda x: -x[1])[:3])
    L.append(f"| {kind} | {np.mean(nis):.2f} [{min(nis):.2f}–{max(nis):.2f}] | {np.mean(pc):.1f} [{min(pc):.1f}–{max(pc):.1f}] | "
             f"{sum(r['merges'] for r in rs)} | {sum(r['false_merges'] for r in rs)} | {sum(r['conflicts'] for r in rs)} | "
             f"{sum(not r['one_each'] for r in rs)}/5 | {np.degrees(np.sqrt(g[0])):.3f}, {1000 * np.sqrt(g[2]):.2f} | {top} |"
             .replace(".", ","))
open(out + ".md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
