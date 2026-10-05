"""plan v6 quick, E5 check: birth times of the new-layer Gaussians a revisit dump's requesting frame actually sees.

usage: python tools/e5_layer_age.py DUMP.pt [DUMP.pt ...]
Explains what the two-layer gap at that frame can and cannot measure: with the split s = (candidate + frame) / 2 the
"new" layer may be dominated by Gaussians of an earlier revisit that a previous loop already aligned.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import torch  # noqa: E402

from deform import metrics as M  # noqa: E402
from deform.dump import load_dump  # noqa: E402
from deform.pipeline import gauss_from_dump  # noqa: E402
from deform.render_utils import DEV, det_scope, make_cam, render_subset  # noqa: E402

for p in sys.argv[1:]:
    d = load_dump(p)
    req = d["request"]
    C, cand = int(req["uid"]), int(req["cand_kf"])
    G = gauss_from_dump(d["gauss_pre"])
    t0 = d["gauss_pre"]["t0"].reshape(-1).long().to(DEV)
    split = 0.5 * (C + cand)
    m_old, m_new = M.layer_masks_birth(t0, split)
    with torch.no_grad(), det_scope(True):
        cam = make_cam(C, req["pose_c2w"], d["intrinsics"])
        c = render_subset(cam, G, m_new)["contrib_full"]
    w = torch.where(c > 0.5, c, torch.zeros_like(c))
    tot = float(w.sum())
    edges = [split, C - 100, C - 30, C + 1]
    names = [f"{int(split)}..{C - 101}", f"{C - 100}..{C - 31}", f"{C - 30}..{C}"]
    parts = []
    for (lo, hi), n in zip(zip(edges[:-1], edges[1:]), names):
        sel = (t0 >= lo) & (t0 < hi)
        parts.append(f"t0 {n}: {100 * float(w[sel].sum()) / max(tot, 1e-9):.0f} % of the contribution ({int(((c > 0.5) & sel).sum())} Gaussians)")
    # most frequent birth keyframes (by contribution)
    u, inv = torch.unique(t0[c > 0.5], return_inverse=True)
    s = torch.zeros(u.shape[0], device=DEV).scatter_add_(0, inv, c[c > 0.5])
    top = torch.argsort(s, descending=True)[:6]
    print(f"{req['tag']} frame {C} cand kf {cand} split {split:.0f}: new layer seen from the frame -> " + "; ".join(parts))
    print("   top birth keyframes:", [(int(u[i]), f"{100 * float(s[i]) / max(tot, 1e-9):.0f} %") for i in top])
