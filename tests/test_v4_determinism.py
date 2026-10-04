"""plan v4 step A unit tests (no GPU data needed; the GPU is used only if available).

A1  splitmix64 test vectors; hash cap: order independence, CPU == GPU, remove one pair -> at most one node
    changes, tie-break on equal keys, random mode unchanged against the v4 backup implementation.
A3  det rasterizer: two renders of the same random scene give bit-identical contributions, and agree with the
    float build to the fixed-point resolution (skipped without CUDA / without the det build).
usage: python tests/test_v4_determinism.py
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deform.correspondences import HASH_SALT, cap_pairs, pair_hash_order, pair_keys, splitmix64  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def synth(P, Kn, W, seed, dev):
    g = torch.Generator().manual_seed(seed)
    uid = torch.randint(0, 300, (P,), generator=g)
    pix = torch.randint(0, W * 480, (P,), generator=g)
    node = torch.randint(0, Kn, (P,), generator=g)
    pairs = {"src_uid": uid.to(dev), "src_pix": pix.to(dev), "x": torch.randn(P, 3, generator=g).to(dev), "P": P}
    return pairs, node.to(dev)


def kept_ids(out):
    return set(zip(out["src_uid"].cpu().tolist(), out["src_pix"].cpu().tolist()))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # ---- A1.1 splitmix64 vectors (first outputs of the seed-0 stream, salt = increment)
    want = [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    got = [int(v) for v in splitmix64(np.array([0, HASH_SALT, (2 * HASH_SALT) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64))]
    check("A1.1 splitmix64 test vectors", got == want, [hex(v) for v in got])
    # ---- A1.2 keys are unique per (uid, pixel) and decode back
    W = 640
    uid = torch.tensor([0, 1, 299, 5]); pix = torch.tensor([0, 639, 640 * 479 + 639, 12345])
    k = pair_keys(uid, pix, W)
    back_u, back_v, back_uid = k & ((1 << 20) - 1), (k >> 20) & ((1 << 20) - 1), k >> 40
    check("A1.2 pair key round trip", bool(((back_v * W + back_u) == pix).all() and (back_uid == uid).all()))
    # ---- A1.3 hash cap: order independence and CPU == GPU
    P, Kn = 20000, 150
    pairs, node = synth(P, Kn, W, 1, dev)
    a = cap_pairs(pairs, node, 64, 50000, 0, select="hash", img_w=W)
    perm = torch.randperm(P, generator=torch.Generator().manual_seed(7)).to(dev)
    pp = {kk: (v[perm] if torch.is_tensor(v) else v) for kk, v in pairs.items()}
    b = cap_pairs(pp, node[perm], 64, 50000, 0, select="hash", img_w=W)
    check("A1.3a hash cap is order independent", kept_ids(a) == kept_ids(b), f"{len(kept_ids(a))} kept")
    pc = {kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in pairs.items()}
    c = cap_pairs(pc, node.cpu(), 64, 50000, 0, select="hash", img_w=W)
    check("A1.3b hash cap CPU == GPU", kept_ids(a) == kept_ids(c))
    # ---- A1.4 remove one pair: at most one node changes; removing an unselected pair changes nothing
    node_of = {(int(u), int(p)): int(n) for u, p, n in zip(pairs["src_uid"].cpu(), pairs["src_pix"].cpu(), node.cpu())}
    ka = kept_ids(a)
    worst, n_sel_changed, n_unsel_changed = 0, 0, 0
    rng = np.random.default_rng(0)
    for i in rng.choice(P, 200, replace=False):
        keep = torch.ones(P, dtype=torch.bool, device=dev); keep[int(i)] = False
        q = {kk: (v[keep] if torch.is_tensor(v) else v) for kk, v in pairs.items()}
        kb = kept_ids(cap_pairs(q, node[keep], 64, 50000, 0, select="hash", img_w=W))
        changed = {node_of[t] for t in ka ^ kb}
        worst = max(worst, len(changed))
        removed = (int(pairs["src_uid"][i]), int(pairs["src_pix"][i]))
        if removed in ka:
            n_sel_changed += len(changed) > 0
        else:
            n_unsel_changed += len(changed) > 0
    check("A1.4 remove one pair -> <= 1 node changes", worst <= 1 and n_unsel_changed == 0,
          f"max nodes changed {worst}, unselected removals that changed something {n_unsel_changed}")
    # ---- A1.5 equal keys: tie broken by index, deterministic
    pairs2 = {"src_uid": torch.tensor([3, 3, 3], device=dev), "src_pix": torch.tensor([10, 10, 11], device=dev),
              "v": torch.tensor([0.0, 1.0, 2.0], device=dev), "P": 3}
    t1 = cap_pairs(pairs2, torch.zeros(3, dtype=torch.long, device=dev), 2, 50000, 0, select="hash", img_w=W)
    t2 = cap_pairs(pairs2, torch.zeros(3, dtype=torch.long, device=dev), 2, 50000, 5, select="hash", img_w=W)
    check("A1.5 equal keys deterministic", t1["v"].tolist() == t2["v"].tolist() and t1["v"].numel() == 2, t1["v"].tolist())
    # ---- A1.6 max_total keeps the smallest hashes
    m = cap_pairs(pairs, node, 64, 1000, 0, select="hash", img_w=W)
    hs = pair_hash_order(pair_keys(m["src_uid"], m["src_pix"], W))
    h_all = pair_hash_order(pair_keys(a["src_uid"], a["src_pix"], W))
    check("A1.6 max_total keeps smallest hashes", m["src_uid"].numel() == 1000 and hs.max() <= torch.sort(h_all).values[999])
    # ---- A1.7 random mode unchanged against the v4 backup implementation
    spec = importlib.util.spec_from_file_location("corr_v4", os.path.join(ROOT, "runs", "v4_backup", "deform", "correspondences.py"))
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    r_new = cap_pairs(pairs, node, 64, 3000, 11, select="random")
    r_old = old.cap_pairs(pairs, node, 64, 3000, 11)
    check("A1.7 random mode == backup", kept_ids(r_new) == kept_ids(r_old) and r_new["x"].shape == r_old["x"].shape)
    # ---- A3 det rasterizer on a random scene
    try:
        import diff_surfel_rasterization_det  # noqa: F401
        have_det = torch.cuda.is_available()
    except ImportError:
        have_det = False
    if have_det:
        from deform.render_utils import det_scope, make_cam, render_subset
        g = torch.Generator().manual_seed(3)
        N = 60000
        G = {"xyz": (torch.rand(N, 3, generator=g) * 4 - 2).cuda(), "rot": torch.nn.functional.normalize(torch.randn(N, 4, generator=g), dim=-1).cuda(),
             "scale": (torch.rand(N, 2, generator=g) * 0.05 + 0.01).cuda(), "opacity": torch.rand(N, 1, generator=g).cuda(),
             "f_dc": torch.rand(N, 1, 3, generator=g).cuda()}
        G["xyz"][:, 2] += 3.0
        cam = make_cam(0, torch.eye(4, dtype=torch.float64), {"fx": 520.0, "fy": 520.0, "cx": 320.0, "cy": 240.0, "W": 640, "H": 480})
        err = torch.rand(480, 640, generator=g).cuda()
        with det_scope(True):
            c1 = render_subset(cam, G, error_img=err)["contrib_full"]
            c2 = render_subset(cam, G, error_img=err)["contrib_full"]
            d1 = render_subset(cam, G)
        c_float = [render_subset(cam, G, error_img=err)["contrib_full"] for _ in range(3)]
        d0 = render_subset(cam, G)
        check("A3.1 det renders bit-identical", bool(torch.equal(c1, c2)), f"n>0 {int((c1 > 0).sum())}")
        diff_float = max(float((c_float[0] - c_float[i]).abs().max()) for i in (1, 2))
        gap = float((c1 - c_float[0]).abs().max())
        rel = float(((c1 - c_float[0]).abs() / c_float[0].abs().clamp_min(1.0)).max())
        # the float build carries ~1e-6 relative accumulation error on sums of ~1e3 (its own run-to-run spread
        # is of the same order); the det sum is exact to 2^-34 per add
        check("A3.2 det vs float agree to float32 accumulation error", rel < 1e-5,
              f"max |det-float| {gap:.2e} (rel {rel:.1e}), float-vs-float {diff_float:.2e}")
        same_img = all(torch.equal(d0[k], d1[k]) for k in ("render", "rend_alpha", "rend_depth_median", "rend_normal"))
        check("A3.3 det build leaves images unchanged", same_img)
    else:
        print("[SKIP] A3 det rasterizer (no CUDA or det build missing)")
    print("V4_UNIT_PASS" if not FAILS else f"V4_UNIT_FAIL {FAILS}")
    return not FAILS


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
