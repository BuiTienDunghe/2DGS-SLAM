"""plan v6 quick, E4: the merge rule on synthetic layers.

usage: python tests/test_v6_e4.py
  M1 two coincident layers 3 mm apart  -> about 100 % of the pairs merge (tau_d 5 and 10 mm)
  M2 two layers 20 mm apart            -> no pair
  M3 normals 40 deg apart / tangential distance above the scale -> no pair; one old Gaussian is used once
  M4 mode B: opacity-weighted position / scales, max opacity, proper rotation whose normal is the merged normal
  M5 (needs CUDA, skipped without) GaussianModel rebuilt from a state: prune_points == plain indexing
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from e4_merge import R_to_quat, match_pairs, merge_b, oriented_normals, quat_to_R  # noqa: E402


def layer(dz, rng, n=40, spacing=0.01, jitter=0.5e-3):
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2) * spacing
    x = np.concatenate([g + rng.normal(0, jitter, g.shape), np.full((g.shape[0], 1), dz)], 1)
    nrm = np.tile(np.array([0.0, 0.0, 1.0]), (x.shape[0], 1))
    return x, nrm, np.full(x.shape[0], 0.01)


def main():
    rng = np.random.default_rng(0)
    xo, no, so = layer(0.0, rng)
    for tau in (0.005, 0.010):
        xn, nn_, sn = layer(0.003, rng)
        i, j, dn = match_pairs(xo, no, so, xn, nn_, sn, tau)
        frac = len(i) / len(xn)
        assert frac > 0.97 and len(np.unique(j)) == len(j) and len(np.unique(i)) == len(i), (tau, frac)
        xn, nn_, sn = layer(0.020, rng)
        i2, _, _ = match_pairs(xo, no, so, xn, nn_, sn, tau)
        assert len(i2) == 0, (tau, len(i2))
        print(f"M1 M2 ok (tau_d {1e3 * tau:.0f} mm): 3 mm apart -> {100 * frac:.1f} % merged, 20 mm apart -> 0")
    xn, nn_, sn = layer(0.003, rng)
    tilt = np.array([0.0, np.sin(np.radians(40)), np.cos(np.radians(40))])
    assert len(match_pairs(xo, no, so, xn, np.tile(tilt, (len(xn), 1)), sn, 0.005)[0]) == 0
    assert len(match_pairs(xo, no, np.full(len(xo), 1e-4), xn + np.array([0.004, 0, 0]), nn_, np.full(len(xn), 1e-4), 0.005)[0]) == 0
    i, j, _ = match_pairs(xo[:1], no[:1], so[:1], np.repeat(xo[:1], 5, 0) + np.array([0, 0, 0.001]) * np.arange(1, 6)[:, None], np.repeat(no[:1], 5, 0), np.repeat(so[:1], 5), 0.01)
    assert len(i) == 1 and i[0] == 0, (i, j)
    print("M3 ok: normal / tangential gates, one old Gaussian merges once (with the closest new one)")
    # ---- mode B
    xyz = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.004]])
    Rk = quat_to_R(np.array([[1.0, 0, 0, 0]]))[0]
    tl = np.radians(20)
    Rd = np.array([[1, 0, 0], [0, np.cos(tl), -np.sin(tl)], [0, np.sin(tl), np.cos(tl)]])
    quat = R_to_quat(np.stack([Rk, Rd]))
    scale = np.array([[0.01, 0.02], [0.03, 0.02]])
    opac = np.array([0.9, 0.3])
    n_or = oriented_normals(xyz, quat, np.array([10, 20]), {10: np.array([0.0, 0.0, 2.0]), 20: np.array([0.0, 0.0, 2.0])})
    x, q, s, o = merge_b(xyz, quat, scale, opac, n_or, np.array([0]), np.array([1]))
    assert np.allclose(x[0], (0.9 * xyz[0] + 0.3 * xyz[1]) / 1.2) and np.allclose(s[0], (0.9 * scale[0] + 0.3 * scale[1]) / 1.2) and o[0] == 0.9
    R = quat_to_R(q)[0]
    n_exp = 0.9 * n_or[0] + 0.3 * n_or[1]
    n_exp /= np.linalg.norm(n_exp)
    assert np.allclose(R[:, 2], n_exp, atol=1e-9) and np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0.999
    assert abs(R[:, 0] @ np.array([1.0, 0, 0])) > 0.999  # tangent axis of the more reliable member kept
    print("M4 ok: mode B position / scales / opacity / rotation")
    # ---- upstream prune on a rebuilt model
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")
    except Exception as e:
        print(f"M5 skipped ({e})")
        print("ALL PASS (CPU part)")
        return
    import deform  # noqa: F401
    from e4_merge import model_from_state
    from utils import loop_dump

    N = 500
    g = {"xyz": torch.randn(N, 3), "f_dc": torch.rand(N, 1, 3), "opacity_raw": torch.randn(N, 1), "scale_raw": torch.randn(N, 2) - 4,
         "rot_raw": torch.nn.functional.normalize(torch.randn(N, 4), dim=-1), "t0": torch.arange(N) * 2, "tc": torch.arange(N) * 2 + 4,
         "tl": torch.arange(N) * 2 + 8, "dc": torch.rand(N), "active": torch.rand(N) > 0.5}
    opt = {"position_lr": 3.2e-5, "feature_lr": 2.5e-3, "opacity_lr": 0.05, "scaling_lr": 5e-4, "rotation_lr": 1e-3, "percent_dense": 0.01}
    m = model_from_state(g, opt)
    mask = torch.rand(N) < 0.3
    m.prune_points(mask.cuda())
    m.check_bookkeeping()
    s = loop_dump.gaussian_state(m)
    keep = ~mask
    for a, b in (("xyz", "xyz"), ("f_dc", "f_dc"), ("opacity_raw", "opacity_raw"), ("scale_raw", "scale_raw"), ("rot_raw", "rot_raw")):
        assert torch.equal(s[a], g[b][keep].float()), a
    for a in ("t0", "tc", "tl"):
        assert torch.equal(s[a].long(), g[a][keep].long()), a
    assert torch.equal(s["active"], g["active"][keep]) and torch.allclose(s["dc"], g["dc"][keep])
    print(f"M5 ok: prune_points on the rebuilt GaussianModel == indexing ({int(mask.sum())} of {N} removed, every per-Gaussian tensor cut)")
    print("ALL PASS")


if __name__ == "__main__":
    main()
