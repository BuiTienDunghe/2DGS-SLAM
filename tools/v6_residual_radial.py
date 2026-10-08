"""plan v6 A5(i) (Q5): signed depth residual of the final map against the input depth, as a function of image radius.

usage: python tools/v6_residual_radial.py --out DIR RUN_DIR [RUN_DIR ...]      (GPU; run only between SLAM runs)
For every keyframe of final_state.pt (map before refinement): render the map at the keyframe's final pose, take
r = (D_rendered - D_input) / D_input on pixels with alpha > 0.95 and a valid input depth, where D_input is the depth
exactly as that run's loader produced it (the run's resolved config is stored in final_state.pt). The pixels of all
keyframes are pooled into 10 radial bins of equal pixel count (radius from the principal point). Reported per bin:
mean of r over |r| < 0.1 (gross outliers at depth edges dropped) and median of r. amplitude = outermost bin - innermost.
Only the view-inconsistent part of a depth bias can show up here: what all views agree on is absorbed by the map.
Writes DIR/residual_radial.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from munch import munchify  # noqa: E402

from deform.render_utils import DEV, make_cam, render_subset  # noqa: E402
from e_v4_durability import state_from_gauss  # noqa: E402
from utils.dataset import load_dataset  # noqa: E402

NB = 10


@torch.no_grad()
def profile(rd):
    final = torch.load(os.path.join(rd, "final_state.pt"), map_location="cpu", weights_only=False)
    cfg = final["config"]
    cal = cfg["Dataset"]["Calibration"]
    intr = {"fx": cal["fx"], "fy": cal["fy"], "cx": cal["cx"], "cy": cal["cy"], "W": cal["width"], "H": cal["height"]}
    ds = load_dataset(munchify(cfg["model_params"]), cfg["model_params"]["source_path"], config=cfg)
    G = state_from_gauss(final["gaussians"])
    key = "rend_depth_expected" if cfg["Training"]["depth_type"] == "expected" else "rend_depth_median"
    dmin, dmax = cfg["Training"]["depth_min_threshold"], cfg["Training"]["depth_max_threshold"]
    H, W = int(intr["H"]), int(intr["W"])
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    rad = torch.hypot(xs - intr["cx"], ys - intr["cy"])
    edges = torch.quantile(rad.flatten(), torch.linspace(0, 1, NB + 1, device=DEV))
    bin_id = torch.bucketize(rad, edges[1:-1])
    s_mean = torch.zeros(NB, dtype=torch.float64, device=DEV)
    n_mean = torch.zeros(NB, dtype=torch.float64, device=DEV)
    vals = [[] for _ in range(NB)]
    kfs = sorted(int(u) for u in final["keyframe_uids"])
    for u in kfs:
        _, _, depth, _ = ds[u]
        pkg = render_subset(make_cam(u, final["poses"][u], intr), G)
        d_hat, alpha = pkg[key][0], pkg["rend_alpha"][0]
        m = (alpha > 0.95) & (depth > dmin) & (depth < dmax)
        r = (d_hat - depth) / depth.clamp_min(1e-6)
        inl = m & (r.abs() < 0.1)
        s_mean += torch.bincount(bin_id[inl], weights=r[inl].double(), minlength=NB)
        n_mean += torch.bincount(bin_id[inl], minlength=NB).double()
        for b in range(NB):
            x = r[m & (bin_id == b)]
            if x.numel() > 400:  # subsample: the median over all keyframes does not need every pixel
                x = x[torch.randint(0, x.numel(), (400,), device=DEV, generator=None)]
            vals[b].append(x.float().cpu())
    mean = (s_mean / n_mean.clamp_min(1)).cpu().numpy()
    med = np.array([float(torch.cat(v).median()) if v else float("nan") for v in vals])
    return {"n_keyframes": len(kfs), "bin_edges_px": edges.cpu().tolist(), "mean": mean.tolist(), "median": med.tolist(),
            "n_px": n_mean.cpu().tolist(), "amplitude_mean": float(mean[-1] - mean[0]), "amplitude_median": float(med[-1] - med[0]),
            "loader": {"distorted": cal["distorted"], "depth_undistort": cal.get("depth_undistort", False), "fx": cal["fx"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    out = {}
    for rd in a.runs:
        rd = rd.rstrip("/")
        p = profile(rd)
        out[os.path.basename(rd)] = p
        print("%-34s kf %3d  amplitude mean %+.2e  median %+.2e | mean per bin (x1e-3): %s" % (
            os.path.basename(rd), p["n_keyframes"], p["amplitude_mean"], p["amplitude_median"],
            " ".join(f"{1e3 * v:+.2f}" for v in p["mean"])))
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(a.out, "residual_radial.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
