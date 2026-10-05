"""plan v6 quick, E3: the structure compensation must recover known per-keyframe offsets (CPU only).

usage: python tests/test_v6_e3.py
A plane sampled by 30 "keyframes" (2 000 points each, all over the plane), every keyframe shifted along the
normal by a known offset in [-3, 3] mm, plus 0,3 mm noise. keyframe_offsets must return the offsets with an
error < 0,5 mm, and one compensation pass must bring every remaining offset below 0,5 mm.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import numpy as np  # noqa: E402

from e3_floor_event import compensate, keyframe_offsets  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    K, M = 30, 2000
    true = np.linspace(-3e-3, 3e-3, K)
    rng.shuffle(true)
    xs, ts = [], []
    for a in range(K):
        p = np.zeros((M, 3))
        p[:, :2] = rng.uniform(0, 1.0, (M, 2))
        p[:, 2] = true[a] + rng.normal(0, 0.3e-3, M)
        xs.append(p)
        ts.append(np.full(M, 300 + 2 * a))
    x, t = np.concatenate(xs), np.concatenate(ts)
    n = np.tile(np.array([0.0, 0.0, 1.0]), (x.shape[0], 1))
    off, cnt = keyframe_offsets(x, n, t)
    est = np.array([off[300 + 2 * a] for a in range(K)])
    err = np.abs(est - (true - true.mean()))
    print(f"offsets recovered: max error {1e3 * err.max():.3f} mm, median {1e3 * np.median(err):.3f} mm, pairs per keyframe {min(cnt.values())}..{max(cnt.values())}")
    assert err.max() < 0.5e-3, err.max()
    x2, total, _ = compensate(x, n, t, iters=1)
    off2, _ = keyframe_offsets(x2, n, t)
    rest = np.abs(np.array(list(off2.values())))
    print(f"after one compensation pass: largest remaining offset {1e3 * rest.max():.3f} mm (before {1e3 * np.abs(est).max():.3f} mm)")
    assert rest.max() < 0.5e-3
    print("ALL PASS")


if __name__ == "__main__":
    main()
