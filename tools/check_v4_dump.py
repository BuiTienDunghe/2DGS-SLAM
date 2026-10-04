"""CPU-only sanity check of a plan-v4 dump: raw parameters present and activations reproduce exactly, anchor
frame kept at its PGO pose by the online write-back, pose shift of the deformation vs PGO.
usage: CUDA_VISIBLE_DEVICES= python tools/check_v4_dump.py DUMP.pt [DUMP.pt ...]
"""
import sys

import numpy as np
import torch

for p in sys.argv[1:]:
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["gauss_pre"]
    anchors = d.get("anchor_uids")
    print(p)
    print("  raw params:", "scale_raw" in g, "anchor_uids:", anchors, "mode:", d["meta"].get("mode"), "accepted:", (d.get("deform_log") or {}).get("accepted"))
    if "scale_raw" in g:
        print("  exp(scale_raw)==scale:", bool(torch.equal(torch.exp(g["scale_raw"]), g["scale"])),
              "sigmoid(opacity_raw)==opacity:", bool(torch.equal(torch.sigmoid(g["opacity_raw"]), g["opacity"])),
              "normalize(rot_raw)==rot:", bool(torch.equal(torch.nn.functional.normalize(g["rot_raw"]), g["rot"])))
    if anchors:
        a = anchors[0]
        print("  anchor %d: |poses_final - poses_pgo| = %.3e m" % (a, float((d["poses_final"][a][:3, 3] - d["poses_pgo"][a][:3, 3]).norm())))
    dd = [float((d["poses_final"][u][:3, 3] - d["poses_pgo"][u][:3, 3]).norm()) for u in d["poses_pgo"] if u in d["poses_final"]]
    print("  pose shift deform vs PGO: median %.2f mm, max %.2f mm (n=%d)" % (1e3 * np.median(dd), 1e3 * np.max(dd), len(dd)))
