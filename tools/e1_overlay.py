"""T9: overlay CAPE-lite planes on real frames.
usage: python tools/e1_overlay.py CONFIG OUT_DIR FRAME [FRAME ...]
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "gaussian_splatting")
from munch import munchify  # noqa: E402

from utils.config_utils import load_config  # noqa: E402
from utils.dataset import load_dataset  # noqa: E402
from utils.plane_anchor import PlaneExtractor  # noqa: E402

cfg = load_config(sys.argv[1])
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
ds = load_dataset(munchify(cfg["model_params"]), "", config=cfg)
ds.device = "cpu"   # CPU only: never compete with a SLAM run for the GPU
ext = PlaneExtractor(cfg["PlaneTracking"])
cs = ext.cell
rng = np.random.default_rng(0)
colors = (rng.uniform(60, 255, (64, 3))).astype(np.uint8)
seq = cfg["Dataset"]["sequence_name"]
lines = []
for f in [int(x) for x in sys.argv[3:]]:
    img, _, depth, _ = ds[f]
    planes = ext(depth, ds.K)
    rgb = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    over = rgb.copy()
    H, W = rgb.shape[:2]
    for k, p in enumerate(planes):
        m = np.kron(p.mask_cells.numpy().astype(np.uint8), np.ones((cs, cs), np.uint8)).astype(bool)
        mm = np.zeros((H, W), bool)
        mm[: m.shape[0], : m.shape[1]] = m
        over[mm] = (0.45 * over[mm] + 0.55 * colors[k]).astype(np.uint8)
        ys, xs = np.nonzero(mm)
        cv2.putText(over, f"{k}", (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        lines.append(f"{seq} frame {f} plane {k}: n=({p.n[0]:+.3f},{p.n[1]:+.3f},{p.n[2]:+.3f}) d={p.d:.3f} m "
                     f"n_px={p.n_px} rms={1000 * np.sqrt(p.mse):.1f} mm")
    dv = depth.cpu().numpy()
    dcol = cv2.applyColorMap((np.clip(dv / 4.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    dcol[dv <= 0] = 0
    panel = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dcol, cv2.cvtColor(over, cv2.COLOR_RGB2BGR)], 1)
    cv2.putText(panel, f"{seq} frame {f}: {len(planes)} planes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out, f"overlay_{seq}_{f:05d}.png"), panel)
    lines.append(f"{seq} frame {f}: {len(planes)} planes, strip-rejected so far {ext.n_strip_rejected}")
open(os.path.join(out, f"overlay_{seq}.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
