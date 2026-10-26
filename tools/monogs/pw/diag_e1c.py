"""PLAN MONOGS-PW §5: offline diagnosis of the E1c plane world on results_monogs/B/{snt_far,snt_near}/e1c.
usage (repo root, env monogs): python tools/monogs/pw/diag_e1c.py   -> docs/monogs/pw/DIAG_E1C.md, diag_e1c.json
Each anchor created at keyframe k is classified: R1 (another plane of the same true plane in that frame went to
another anchor), R2 (an existing anchor of the same true plane passes the 10 deg / 10 cm gate but was not matched:
overlap), R3 (exists but outside the gate), new. ID errors: simultaneous / flip / mismatch. RPE vs events.
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pw import gt_obs, plane_w2c  # noqa: E402
from eval_runs import jsonl  # noqa: E402

RUNS = {"snt_far": "results_monogs/B/snt_far/e1c", "snt_near": "results_monogs/B/snt_near/e1c"}
D = "docs/monogs/pw"
GATE_DEG, GATE_M = 10.0, 0.10


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def diag(run):
    obs, W, G = gt_obs(run, "e1c", None)
    by_frame = collections.defaultdict(list)
    for o in obs:
        by_frame[o["frame"]].append(o)
    votes = collections.defaultdict(collections.Counter)
    for o in obs:
        if o["anchor"] is not None:
            votes[o["anchor"]][o["label"]] += 1
    alab = {a: c.most_common(1)[0][0] for a, c in votes.items()}
    an = jsonl(os.path.join(run, "anchors.jsonl"))
    kf_rows = {r["frame"]: r for r in an}
    frames = [json.loads(l) for l in open(os.path.join(run, "e1c_frames.jsonl"))]
    fr = {r["frame"]: r for r in frames}
    seen, cls = set(), collections.Counter()
    detail = []
    prev_ids = set()
    for k in sorted(kf_rows):
        cur = {a["id"]: a for a in kf_rows[k]["anchors"]}
        new = [a for i, a in cur.items() if i not in prev_ids and a["created_kf"] == k]
        prev = {i: a for i, a in cur.items() if i in prev_ids}
        if k == min(kf_rows):
            prev_ids = set(cur)
            continue
        for a in new:
            # the creating plane: the frame-k plane closest to the new anchor (both in the estimated world)
            cand = [(abs(np.dot(p["n_w"], a["n_w"]) - 1) + abs(p["d_w"] - a["d_w"]), i) for i, p in enumerate(fr[k]["planes"])]
            if not cand:
                continue
            pi = min(cand)[1]
            ob = next((o for o in by_frame[k] if o["i"] == pi), None)
            if ob is None:
                continue
            L = ob["label"]
            same_frame_other = [o for o in by_frame[k] if o["label"] == L and o["i"] != pi and o["anchor"] is not None]
            same_lab_prev = [i for i in prev if alab.get(i) == L]
            if same_frame_other:
                c = "R1"
            elif same_lab_prev:
                n_c, d_c = fr[k]["planes"][pi]["n_w"], fr[k]["planes"][pi]["d_w"]
                n_m, d_m = plane_w2c(n_c, d_c, W[k])
                gated = False
                for i in same_lab_prev:
                    n_a, d_a = plane_w2c(prev[i]["n_w"], prev[i]["d_w"], W[k])
                    if np.degrees(np.arccos(np.clip(n_m @ n_a, -1, 1))) <= GATE_DEG and abs(d_m - d_a) <= GATE_M:
                        gated = True
                c = "R2" if gated else "R3"
            else:
                c = "new"
            cls[c] += 1
            detail.append(dict(anchor=a["id"], kf=k, label=L, cls=c))
        prev_ids = set(cur)
    pairs = [o for o in obs if o["anchor"] is not None]
    mism = 100.0 * np.mean([o["label"] != alab.get(o["anchor"]) for o in pairs]) if pairs else None
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    for o in pairs:
        by[o["label"]][o["frame"]].add(o["anchor"])
    simult = flip = 0
    flip_frames = set()
    for l, fs in by.items():
        fl = sorted(fs)
        simult += sum(1 for f_ in fl if len(fs[f_]) >= 2)
        for a_, b_ in zip(fl[:-1], fl[1:]):
            if len(fs[a_]) == 1 and len(fs[b_]) == 1 and fs[a_] != fs[b_]:
                flip += 1
                flip_frames.add(b_)
    # RPE per frame (translation of the relative-motion error) vs events
    rpe = {}
    for t in range(1, len(W)):
        E = (W[t] @ np.linalg.inv(W[t - 1])) @ np.linalg.inv(G[t] @ np.linalg.inv(G[t - 1]))
        rpe[t] = float(np.linalg.norm(E[:3, 3]))
    def mean_split(sel):
        a_ = [v for t, v in rpe.items() if sel(t)]
        b_ = [v for t, v in rpe.items() if not sel(t)]
        return (100 * float(np.mean(a_)) if a_ else None, 100 * float(np.mean(b_)) if b_ else None, len(a_))
    init_plane = {r["frame"] for r in frames if r["init_source"] == "plane"}
    return dict(classes=dict(cls), n_new_anchors=sum(cls.values()), pct_mismatch=mism, id_simultaneous=simult, id_flip=flip,
                rpe_flip=mean_split(lambda t: t in flip_frames), rpe_plane_init=mean_split(lambda t: t in init_plane),
                detail=detail)


def main():
    os.makedirs(D, exist_ok=True)
    res = {s: diag(r) for s, r in RUNS.items()}
    json.dump(res, open(os.path.join(D, "diag_e1c.json"), "w"), indent=1, default=float)
    L = ["# DIAG_E1C — chẩn đoán plane world của E1c trên MonoGS (PLAN MONOGS-PW §5)\n",
         "Sinh bằng `tools/monogs/pw/diag_e1c.py` trên `results_monogs/B/{snt_far,snt_near}/e1c`. Mặt đo đổi về hệ camera bằng "
         "`traj_all`, rồi sang world bằng GT; nhãn mặt thật theo mục 8.1. Chỉ để diễn giải, không đổi thiết kế.\n",
         "| Chuỗi | Anchor mới (sau keyframe đầu) | R1: mảnh khác cùng frame | R2: trong cổng, không ghép (overlap) | R3: ngoài cổng (trôi) | Mới thật | % ghép nhầm | Đổi ID đồng thời | Đổi ID lật |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s, r in res.items():
        c = r["classes"]
        L.append(f"| {s} | {r['n_new_anchors']} | {c.get('R1', 0)} | {c.get('R2', 0)} | {c.get('R3', 0)} | {c.get('new', 0)} | "
                 f"{f(r['pct_mismatch'])} | {r['id_simultaneous']} | {r['id_flip']} |")
    L += ["", "**RPE (dịch, cm/frame) ở frame có và không có sự kiện:**\n",
          "| Chuỗi | Có lật ID | Không lật ID | Số frame lật | Khởi tạo bằng plane | Không khởi tạo bằng plane | Số frame plane |", "|---|---|---|---|---|---|---|"]
    for s, r in res.items():
        a, b = r["rpe_flip"], r["rpe_plane_init"]
        L.append(f"| {s} | {f(a[0], 2)} | {f(a[1], 2)} | {a[2]} | {f(b[0], 2)} | {f(b[1], 2)} | {b[2]} |")
    open(os.path.join(D, "DIAG_E1C.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
