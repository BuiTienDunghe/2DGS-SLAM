"""plan v6 quick, E3: noise floor of the two-layer gap AT EVENT TIME on a single-visit region, and the part of it
that per-keyframe depth offsets (scale / shift of each keyframe's depth) explain.

usage: python tools/e3_floor_event.py --out DIR [--min-cur 600] [--revisit] [--final] RUN_DIR [RUN_DIR ...]
States per run:
  event  the map right after the correction of every loop event with cur frame > min-cur (gauss_post, poses_final)
  revisit (--revisit) the revisit-burst dumps of an E5 run (revisit_dumps/, map as dumped)
  final  (--final) final_state.pt, with the same region / views (reference for the 10-11 mm and 6,8-7,2 mm floors)
Region: Gaussians born at keyframes lo..hi (300..600, visited once). Two halves of the region, split by
  rank  parity of the birth-keyframe rank (keyframe-wise; t0 values are all even)
  hash  parity bit of splitmix64(Gaussian index) (per Gaussian)
are rendered as two layers from every 5th keyframe of lo..hi and measured with the e_dl of tools/h0_noise_floor.py
(eval_pixel_set gating, gap_maps, edl_paired).
Structure compensation: for every Gaussian i of the region (normal n_i turned towards its birth keyframe), up to 8
nearest neighbours born at another keyframe within 2 cm; offset of keyframe a = median of n_i . (x_i - x_j) over
the Gaussians born at a; every Gaussian is moved by -offset(t0) along n_i, then the splits are measured again.
Writes DIR/e3.jsonl (one row per state).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402


def keyframe_offsets(x, n, t, r=0.02, k=8, k_query=48, min_pairs=30):
    """x [N,3], n [N,3] oriented unit normals, t [N] birth keyframe (numpy). Returns ({kf: offset}, {kf: n_pairs}).
    offset(a) = median over Gaussians i born at a and their <= k nearest neighbours j born elsewhere (within r) of
    n_i . (x_i - x_j): how far keyframe a's surface sits in front of (+) / behind (-) its neighbours' surface."""
    from scipy.spatial import cKDTree

    N = x.shape[0]
    tree = cKDTree(x)
    kq = min(k_query, N)
    dist, idx = tree.query(x, k=kq, distance_upper_bound=r)
    ok = idx < N
    idx_c = np.where(ok, idx, 0)
    ok &= t[idx_c] != t[:, None]
    ok &= np.cumsum(ok, axis=1) <= k
    d = np.einsum("nd,nkd->nk", n, x[:, None, :] - x[idx_c])
    off, cnt = {}, {}
    ti = np.broadcast_to(t[:, None], d.shape)[ok]
    dv = d[ok]
    order = np.argsort(ti, kind="stable")
    ti, dv = ti[order], dv[order]
    uniq, start = np.unique(ti, return_index=True)
    for a, s, e in zip(uniq, start, list(start[1:]) + [len(ti)]):
        cnt[int(a)] = int(e - s)
        if e - s >= min_pairs:
            off[int(a)] = float(np.median(dv[s:e]))
    return off, cnt


def compensate(x, n, t, iters=1, **kw):
    """Shift every Gaussian by -offset(t0) along its normal, `iters` times. Returns (x_new, total offset per kf, pairs)."""
    x = x.copy()
    total, cnt = {}, {}
    for _ in range(iters):
        off, cnt = keyframe_offsets(x, n, t, **kw)
        o = np.array([off.get(int(a), 0.0) for a in t])
        x = x - o[:, None] * n
        for a, v in off.items():
            total[a] = total.get(a, 0.0) + v
    return x, total, cnt


def main():
    import deform  # noqa: F401
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from deform import config as dcfg
    from deform import metrics as M
    from deform.correspondences import splitmix64
    from deform.dump import EventState, list_dumps, load_dump
    from deform.field import quat_to_rotmat
    from deform.pipeline import with_pos
    from deform.render_utils import DEV, det_scope
    from e_v4_durability import state_from_gauss
    from h0_noise_floor import FinalState
    from utils.loop_dump import load_deform_config

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=int, default=300)
    ap.add_argument("--hi", type=int, default=600)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-cur", type=int, default=600)
    ap.add_argument("--revisit", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--comp-iters", type=int, default=1)
    ap.add_argument("--config", default=os.path.join(root, "configs", "deform", "selected_v6.yaml"))
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    user = load_deform_config(a.config)
    out_path = os.path.join(a.out, "e3.jsonl")
    open(out_path, "w").close()

    def states(rd):
        dumps = [load_dump(p) for p in list_dumps(rd)]
        for d in dumps:
            if int(d["meta"]["cur_uid"]) > a.min_cur:
                st = EventState(d)
                G = state_from_gauss(st.gpre, d["gauss_post"]["xyz"], d["gauss_post"]["rot"])
                poses = {int(u): torch.as_tensor(v).double() for u, v in d["poses_final"].items()}
                yield f'event {d["meta"]["cur_uid"]}<->{d["meta"]["loop_uid"]}', st, G, st.gpre["t0"], poses, d["config"], sorted(st.kf_uids)
        if a.revisit:
            for p in sorted(glob.glob(os.path.join(rd, "revisit_dumps", "rv_*.pt"))):
                d = load_dump(p)
                st = EventState(d)
                G = state_from_gauss(st.gpre)
                poses = {int(u): torch.as_tensor(v).double() for u, v in d["poses_pre"].items()}
                yield f'revisit {d["request"]["tag"]} f{d["request"]["uid"]}', st, G, st.gpre["t0"], poses, d["config"], sorted(st.kf_uids)
        if a.final:
            final = torch.load(os.path.join(rd, "final_state.pt"), map_location="cpu", weights_only=False)
            intr = dumps[0]["intrinsics"] if dumps else load_dump(sorted(glob.glob(os.path.join(rd, "revisit_dumps", "rv_*.pt")))[0])["intrinsics"]
            st = FinalState(final, intr)
            yield "final", st, state_from_gauss(final["gaussians"]), final["gaussians"]["t0"], st.poses, final["config"], sorted(st.kf_uids)

    def floor(st, G, poses, views, la, lb, cfg):
        J = [(u, None) for u in views]
        Pi = M.eval_pixel_set(st, G, poses, J, la, lb, cfg)
        maps = M.gap_maps(st, G, poses, Pi, la, lb)
        e, n = M.edl_paired(Pi, {"floor": maps})
        f = e["floor"]
        return {"median_mm": None if f["median"] is None else 1e3 * f["median"], "p90_mm": None if f["p90"] is None else 1e3 * f["p90"], "n_pix": n}

    for rd in a.runs:
        for label, st, G, t0, poses, config, kfs in states(rd):
            cfg = dcfg.resolve(user, config)
            t0 = t0.reshape(-1).long()
            N = t0.shape[0]
            region = ((t0 >= a.lo) & (t0 <= a.hi)).to(DEV)
            views = [u for u in kfs if a.lo <= u <= a.hi][:: a.every]
            row = {"run": os.path.basename(rd), "state": label, "n_gauss": N, "n_region": int(region.sum()), "views": views}
            if len(views) == 0 or int(region.sum()) < 1000:
                row["note"] = "region not mapped yet"
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                continue
            _, inv = torch.unique(t0, return_inverse=True)
            rank_a = (inv % 2 == 0).to(DEV)
            h = splitmix64(np.arange(N, dtype=np.uint64))
            hash_a = torch.from_numpy(((h >> np.uint64(32)) & np.uint64(1)) == 0).to(DEV)
            splits = {"rank": (rank_a & region, ~rank_a & region), "hash": (hash_a & region, ~hash_a & region)}
            with torch.no_grad(), det_scope(True):
                row["floor"] = {k: floor(st, G, poses, views, la, lb, cfg) for k, (la, lb) in splits.items()}
                # ---- structure compensation on the region
                ridx = torch.nonzero(region).reshape(-1)
                x = G["xyz"][ridx].double().cpu().numpy()
                nrm = quat_to_rotmat(G["rot"][ridx].double())[:, :, 2].cpu().numpy()
                tr = t0[ridx.cpu()].numpy()
                cen = np.stack([np.asarray(poses[int(u)], dtype=np.float64)[:3, 3] if int(u) in poses else np.full(3, np.nan) for u in tr])
                s = np.sign(np.einsum("nd,nd->n", nrm, cen - x))
                s[~np.isfinite(s) | (s == 0)] = 1.0
                nrm = nrm * s[:, None]
                x_new, off, cnt = compensate(x, nrm, tr, iters=a.comp_iters)
                ov = np.array(sorted(off.values()))
                row["offsets"] = {"n_kf_region": int(len(np.unique(tr))), "n_kf_with_offset": int(len(ov)), "pairs_total": int(sum(cnt.values())),
                                  "median_abs_mm": None if len(ov) == 0 else 1e3 * float(np.median(np.abs(ov))),
                                  "p90_abs_mm": None if len(ov) == 0 else 1e3 * float(np.quantile(np.abs(ov), 0.9)),
                                  "max_abs_mm": None if len(ov) == 0 else 1e3 * float(np.abs(ov).max()),
                                  "mean_mm": None if len(ov) == 0 else 1e3 * float(ov.mean()),
                                  "per_kf_mm": {int(k): 1e3 * v for k, v in sorted(off.items())}}
                xyz2 = G["xyz"].clone()
                xyz2[ridx] = torch.from_numpy(x_new).float().to(DEV)
                G2 = with_pos(G, xyz2, G["rot"])
                row["floor_comp"] = {k: floor(st, G2, poses, views, la, lb, cfg) for k, (la, lb) in splits.items()}
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            fl, fc, of = row["floor"], row["floor_comp"], row["offsets"]
            print(f"{row['run'][-10:]:10s} {label:22s} region {row['n_region']:6d} views {len(views):2d}  rank {fl['rank']['median_mm']:.1f} (p90 {fl['rank']['p90_mm']:.1f}, {fl['rank']['n_pix']} px)  "
                  f"hash {fl['hash']['median_mm']:.1f} (p90 {fl['hash']['p90_mm']:.1f})  | offsets kf {of['n_kf_with_offset']}/{of['n_kf_region']} med|o| {of['median_abs_mm']:.2f} p90 {of['p90_abs_mm']:.2f} max {of['max_abs_mm']:.2f} mm"
                  f"  | after comp: rank {fc['rank']['median_mm']:.1f} (p90 {fc['rank']['p90_mm']:.1f})  hash {fc['hash']['median_mm']:.1f}", flush=True)
            del G
            torch.cuda.empty_cache()
    print("[e3] done ->", out_path)


if __name__ == "__main__":
    main()
