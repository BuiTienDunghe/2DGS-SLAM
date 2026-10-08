"""plan v6 A4 (Q5): do the RGB edges and the depth discontinuities line up under each loader mode? CPU only.

usage: python tools/v6_loader_chamfer.py --out DIR [--n 50] [--overlay 0 300 600] [--config configs/tum/fr1_room.yaml]
Three image pairs per frame (B and R load the same raw images and differ only in K, so they share one result):
  A    RGB undistorted (bilinear), depth raw                        = the released loader
  raw  RGB raw, depth raw                                           = modes B and R
  D    RGB undistorted (bilinear), depth undistorted (nearest)      = plan v6 X3
Measure: for every depth-discontinuity pixel, the distance to the nearest Canny edge of the RGB image (one-sided
chamfer, pixels; capped at 20 px). Reported as the per-frame median over the centre (radius < 150 px from the
principal point) and over the rim. Writes DIR/loader_chamfer.json and overlay PNGs for the requested frames.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from utils.config_utils import load_config  # noqa: E402
from utils.dataset import TUMParser  # noqa: E402

CAP = 20.0


def depth_edges(depth_m, rel_jump=0.05):
    """Pixels where the depth jumps by more than rel_jump of the nearer depth to a 4-neighbour; both must be valid."""
    d = depth_m
    v = d > 0
    e = np.zeros(d.shape, bool)
    for ax in (0, 1):
        a = np.roll(d, -1, axis=ax)
        va = np.roll(v, -1, axis=ax)
        jump = np.abs(a - d) > rel_jump * np.minimum(a, d)
        m = jump & v & va
        if ax == 0:
            m[-1, :] = False
        else:
            m[:, -1] = False
        # mark the nearer (foreground) side: the occluding contour, which is where the colour edge sits
        near_here = d <= a
        e |= m & near_here
        e |= np.roll(m & ~near_here, 1, axis=ax)
    return e


def chamfer(rgb, depth_m, cx, cy):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    can = cv2.Canny(gray, 40, 100) > 0
    dist = cv2.distanceTransform((~can).astype(np.uint8), cv2.DIST_L2, 5)
    de = depth_edges(depth_m)
    yy, xx = np.nonzero(de)
    if yy.size == 0:
        return None, None, can, de
    r = np.hypot(xx - cx, yy - cy)
    dd = np.minimum(dist[yy, xx], CAP)
    c, o = dd[r < 150], dd[r >= 150]
    return (float(np.median(c)) if c.size > 50 else None, float(np.median(o)) if o.size > 50 else None, can, de)


RINGS = [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 420)]
SHIFTS = np.arange(-14, 15)


def fg_edges(depth_m, domain, rel_jump=0.05):
    """Valid pixels on the near side of a depth jump or next to an invalid pixel (the Kinect shadow band hides most
    occlusion boundaries from a both-sides-valid test). Holes inside surfaces add unbiased noise only."""
    v = depth_m > 0
    e = np.zeros(v.shape, bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = np.roll(depth_m, (-dy, -dx), axis=(0, 1))
        nv = nb > 0
        e |= v & nv & ((nb - depth_m) > rel_jump * depth_m)
        e |= v & ~nv
    return e & domain


def shift_sums(rgb, depth_m, domain, cx, cy, rgb_domain=None):
    """Radial misalignment probe: sum over foreground depth-edge pixels of the (capped) distance to the nearest strong
    RGB edge after moving the pixel s px along the radius (s > 0: outwards), per ring and per shift."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 1.0)
    can = cv2.Canny(gray, 60, 150) > 0
    if rgb_domain is not None:
        can &= rgb_domain  # drop the artificial edge between the undistorted image and its black border
    dist = np.minimum(cv2.distanceTransform((~can).astype(np.uint8), cv2.DIST_L2, 5), 10.0)
    H, W = dist.shape
    yy, xx = np.nonzero(fg_edges(depth_m, domain))
    r = np.hypot(xx - cx, yy - cy)
    ux, uy = (xx - cx) / np.maximum(r, 1e-6), (yy - cy) / np.maximum(r, 1e-6)
    sums, cnt = np.zeros((len(RINGS), len(SHIFTS))), np.zeros(len(RINGS))
    masks = [(r >= lo) & (r < hi) for lo, hi in RINGS]
    for i, m in enumerate(masks):
        cnt[i] = m.sum()
    for j, s in enumerate(SHIFTS):
        x2 = np.clip(np.rint(xx + s * ux).astype(int), 0, W - 1)
        y2 = np.clip(np.rint(yy + s * uy).astype(int), 0, H - 1)
        dd = dist[y2, x2]
        for i, m in enumerate(masks):
            sums[i, j] = dd[m].sum()
    return sums, cnt


def overlay(rgb, can, de):
    img = (0.45 * rgb).astype(np.uint8)
    img[can] = (255, 255, 255)
    img[cv2.dilate(de.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (255, 40, 40)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/tum/fr1_room.yaml")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--overlay", type=int, nargs="*", default=[0, 300, 600])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    cal = cfg["Dataset"]["Calibration"]
    K = np.array([[cal["fx"], 0, cal["cx"]], [0, cal["fy"], cal["cy"]], [0, 0, 1.0]])
    dist = np.array([cal["k1"], cal["k2"], cal["p1"], cal["p2"], cal["k3"]])
    m1x, m1y = cv2.initUndistortRectifyMap(K, dist, np.eye(3), K, (cal["width"], cal["height"]), cv2.CV_32FC1)
    P = TUMParser(cfg["Dataset"]["dataset_path"])
    n = len(P.color_paths)
    frames = sorted(set(np.linspace(0, n - 1, a.n).round().astype(int).tolist()))
    rows = []
    H, W = cal["height"], cal["width"]
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    inner = np.zeros((H, W), bool)
    inner[16:-16, 16:-16] = True
    # pixels of the undistorted image whose source lies inside the raw image (minus a margin)
    dom_u = cv2.erode(((m1x >= 0) & (m1x <= W - 1) & (m1y >= 0) & (m1y <= H - 1)).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    domains = {"A": inner & dom_u, "raw": inner, "D": inner & dom_u}
    src_in = ((m1x >= 0) & (m1x <= W - 1) & (m1y >= 0) & (m1y <= H - 1)).astype(np.uint8)
    rgb_dom = {"A": cv2.erode(src_in, np.ones((9, 9), np.uint8)) > 0, "raw": None, "D": cv2.erode(src_in, np.ones((9, 9), np.uint8)) > 0}
    disp = np.hypot(m1x - xs, m1y - ys)
    rr = np.hypot(xs - cal["cx"], ys - cal["cy"])
    acc = {m: [np.zeros((len(RINGS), len(SHIFTS))), np.zeros(len(RINGS))] for m in ("A", "raw", "D")}
    for i in sorted(set(frames) | set(a.overlay)):
        rgb = np.array(Image.open(P.color_paths[i]))
        raw = np.array(Image.open(P.depth_paths[i]))
        raw = raw if raw.dtype == np.uint16 else raw.astype(np.float32)
        rgb_u = cv2.remap(rgb, m1x, m1y, cv2.INTER_LINEAR)
        raw_u = cv2.remap(raw, m1x, m1y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        d, d_u = raw / cal["depth_scale"], raw_u / cal["depth_scale"]
        res = {}
        for mode, (im, dm) in {"A": (rgb_u, d), "raw": (rgb, d), "D": (rgb_u, d_u)}.items():
            c, o, can, de = chamfer(im, dm, cal["cx"], cal["cy"])
            res[mode] = {"centre": c, "rim": o}
            if i in a.overlay:
                cv2.imwrite(os.path.join(a.out, f"overlay_f{i:04d}_{mode}.png"), cv2.cvtColor(overlay(im, can, de), cv2.COLOR_RGB2BGR))
            if i in frames:
                s_, c_ = shift_sums(im, dm, domains[mode], cal["cx"], cal["cy"], rgb_dom[mode])
                acc[mode][0] += s_
                acc[mode][1] += c_
        if i in frames:
            rows.append({"frame": int(i), **res})
    out = {"frames": [r["frame"] for r in rows], "per_frame": rows, "summary": {}, "radial_shift": {"shifts": SHIFTS.tolist(), "rings": RINGS}}
    print("radial offset of the depth edges that best matches the RGB edges (px, + = outwards); expected for A = -|undistortion shift|")
    for k, (lo, hi) in enumerate(RINGS):
        exp = float(np.median(disp[(rr >= lo) & (rr < hi)]))
        line = f"  r in [{lo:3d}, {hi:3d}): undistortion shift {exp:5.2f} px |"
        for mode in ("A", "raw", "D"):
            curve = acc[mode][0][k] / max(acc[mode][1][k], 1)
            j = int(np.argmin(curve))
            out["radial_shift"].setdefault(mode, []).append({"ring": [lo, hi], "undistortion_shift_px": exp, "best_shift_px": int(SHIFTS[j]),
                                                             "curve": curve.tolist(), "n_edge_px": int(acc[mode][1][k])})
            line += f" {mode}: best {int(SHIFTS[j]):+3d} (dist {curve[j]:.2f} vs {curve[list(SHIFTS).index(0)]:.2f} at 0) |"
        print(line)
    for mode in ("A", "raw", "D"):
        for reg in ("centre", "rim"):
            v = np.array([r[mode][reg] for r in rows if r[mode][reg] is not None and r["A"][reg] is not None])
            out["summary"][f"{mode}_{reg}_median_px"] = float(np.median(v)) if v.size else None
    for mode in ("raw", "D"):
        for reg in ("centre", "rim"):
            pairs = [(r[mode][reg], r["A"][reg]) for r in rows if r[mode][reg] is not None and r["A"][reg] is not None]
            out["summary"][f"{mode}_{reg}_lower_than_A"] = [int(sum(x < y for x, y in pairs)), len(pairs)]
    s = out["summary"]
    for reg in ("centre", "rim"):
        print(f"{reg:6s}: A {s[f'A_{reg}_median_px']:.2f} px | raw {s[f'raw_{reg}_median_px']:.2f} px "
              f"(lower than A in {s[f'raw_{reg}_lower_than_A'][0]}/{s[f'raw_{reg}_lower_than_A'][1]} frames) | "
              f"D {s[f'D_{reg}_median_px']:.2f} px (lower in {s[f'D_{reg}_lower_than_A'][0]}/{s[f'D_{reg}_lower_than_A'][1]})")
    json.dump(out, open(os.path.join(a.out, "loader_chamfer.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
