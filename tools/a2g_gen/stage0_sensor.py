"""PLAN A2G-P1 gen §3.2 (A3): is the segment measurement noise what R_s predicts? Uses GT poses (no pose error).

CAPE-lite + build_segments (weighted per-cell fit of P1 core) on every tracked frame, sigma_0 = 0. Segments are
taken to the world with the GT pose; two segments of frames <= 10 tracked frames apart are labelled "same surface"
when the world angle <= 3 deg, both offsets <= 3 cm and they share >= 1 cell of the 5 cm grid (labels only, tool
side). NIS_3 of r(s_a | s_b) with S = J_a R_a J_a^T + J_b R_b J_b^T.
usage: python tools/a2g_gen/stage0_sensor.py [--seqs snt_near snt_far st_far room] [--undistort-depth]
                                               [--tag raw] [--max-frames N]
Writes results_exp/a2g_p1gen/stage0/a3_<seq>_<tag>.npz (one row per pair) and prints a summary.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT0, G, frame_ids, load_cfg, load_ds, read_frame  # noqa: E402

from utils.plane_anchor import PlaneExtractor  # noqa: E402
from utils.pw.segments import build_segments  # noqa: E402
from utils.pw.uncertainty import SensorNoise, jac_a, jac_b, residual  # noqa: E402

GAP, ANG, OFF, H = 10, 3.0, 0.03, 0.05          # labelling thresholds of PLAN §3.2 (tool only)


def seg_stats(sg, sensor, K, Tcw):
    """World plane + diagnostics of one segment: centre pixel distance to the image centre, centroid depth,
    fit excess chi2_fit/dof of the cell centres w.r.t. the fitted plane (sigma_c as in segment_cov)."""
    n_w, p_w, B_w = G.to_world(Tcw, sg.n, sg.p, sg.B)
    n_w = n_w / np.linalg.norm(n_w)
    B_w = G.gram_schmidt(B_w, n_w)
    uvw = K @ sg.p
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    rad = float(np.hypot(u - K[0, 2], v - K[1, 2]))
    mu = sg.mu
    sig = np.maximum(sensor.sigma_z(mu[:, 2]) * np.abs((mu / mu[:, 2:3]) @ sg.n), 1e-9)
    dof = len(mu) - 3
    fitx = float(np.sum(((mu - sg.p) @ sg.n / sig) ** 2) / dof) if dof > 0 else np.nan
    cells = G.cells_from_segment(sg.cells_ij, 20, K, sg.n, sg.d, Tcw, p_w, B_w, H)
    return dict(n=n_w, p=p_w, B=B_w, R=sg.R, rad=rad, z=float(sg.p[2]), fitx=fitx, cells=cells, Tcw=Tcw, sg=sg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["snt_near", "snt_far", "st_far", "room"])
    ap.add_argument("--undistort-depth", action="store_true")
    ap.add_argument("--tag", default="raw")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT0, exist_ok=True)
    for seq in a.seqs:
        t0 = time.time()
        cfg = load_cfg(seq)
        ds = load_ds(cfg)
        K = np.array(ds.K, dtype=np.float64)
        se = cfg["PlaneWorld"]["sensor"]
        sensor = SensorNoise(se.get("a", 0.0), se.get("b", 1.425e-3), 0.0, 0.0)      # sigma_0 = 0 (PLAN §3.2.1)
        ext = PlaneExtractor(cfg["PlaneTracking"])
        fids = frame_ids(cfg, ds)
        if a.max_frames:
            fids = fids[:a.max_frames]
        hist = []                                   # list (per tracked frame) of segment dicts
        rows = []
        n_seg = 0
        for i, f in enumerate(fids):
            _, depth, Tcw = read_frame(ds, f, a.undistort_depth)
            planes = ext(torch.from_numpy(depth).float(), ds.K)
            segs = [seg_stats(sg, sensor, K, Tcw) for sg in build_segments(planes, sensor)]
            n_seg += len(segs)
            for j in range(max(0, i - GAP), i):
                for sb in hist[j]:
                    for sa in segs:
                        c = float(sa["n"] @ sb["n"])
                        if c < np.cos(np.deg2rad(ANG)):
                            continue
                        if abs(sb["n"] @ (sa["p"] - sb["p"])) > OFF or abs(sa["n"] @ (sb["p"] - sa["p"])) > OFF:
                            continue
                        ca = G.cells_from_segment(sa["sg"].cells_ij, 20, K, sa["sg"].n, sa["sg"].d, sa["Tcw"],
                                                  sb["p"], sb["B"], H)
                        if not (ca & sb["cells"]):
                            continue
                        r = residual(sa["n"], sa["p"], sb["n"], sb["p"], sb["B"])
                        Ja = jac_a(sa["n"], sa["B"], sb["n"], sb["B"])
                        Jb = jac_b(sa["n"], sa["p"], sb["n"], sb["p"], sb["B"])
                        SA, SB = Ja @ sa["R"] @ Ja.T, Jb @ sb["R"] @ Jb.T
                        rows.append(dict(r=r, SA=SA, SB=SB, fa=sa["fitx"], fb=sb["fitx"], rad=max(sa["rad"], sb["rad"]),
                                         z=0.5 * (sa["z"] + sb["z"]), gap=i - j, ia=i, ib=j, nc=min(sa["sg"].N_s, sb["sg"].N_s),
                                         ang=float(np.degrees(np.arccos(min(1.0, c))))))
            hist.append(segs)
            if i >= GAP:
                hist[i - GAP] = []                  # frames older than GAP are never read again
            if (i + 1) % 100 == 0:
                print(f"  {seq}: {i + 1}/{len(fids)} frames, {len(rows)} pairs, {time.time() - t0:.0f} s", flush=True)
        if not rows:
            print(f"{seq}: no pairs")
            continue
        r = np.array([x["r"] for x in rows])
        SA = np.array([x["SA"] for x in rows])
        SB = np.array([x["SB"] for x in rows])
        S = SA + SB
        nisv = np.einsum("ni,ni->n", r, np.linalg.solve(S, r[..., None])[..., 0])
        out = os.path.join(OUT0, f"a3_{seq}_{a.tag}.npz")
        np.savez(out, r=r, SA=SA, SB=SB, nis=nisv, fa=np.array([x["fa"] for x in rows]),
                 fb=np.array([x["fb"] for x in rows]), rad=np.array([x["rad"] for x in rows]),
                 z=np.array([x["z"] for x in rows]), gap=np.array([x["gap"] for x in rows]),
                 ia=np.array([x["ia"] for x in rows]), ib=np.array([x["ib"] for x in rows]),
                 ang=np.array([x["ang"] for x in rows]), nc=np.array([x["nc"] for x in rows]), n_frames=len(fids), n_seg=n_seg,
                 undistort_depth=a.undistort_depth)
        print(f"{seq} [{a.tag}]: {len(fids)} frames, {n_seg} segments, {len(rows)} pairs, NIS mean "
              f"{nisv.mean():.2f} median {np.median(nisv):.2f}, {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
