"""plan v6 quick, E1: what does the final map refinement make up for -- the inconsistency left by loops, or online
mapping that was simply not optimised enough?

usage: python tools/e1_refine_split.py --out DIR RUN_DIR [RUN_DIR ...]
Per run: final_state.pt (map before refinement) and final_state_refined.pt (after), same poses (checked).
Two pixel regions:
  loop    union over the loop events of the run of Pi*(k) ∩ (both t0 layers still rendered on the final map),
          built exactly like tools/e_v4_durability.py (rigid replay of dump k, t0 layers split at s_k, J_eval(k));
          only on the J_eval keyframes
  single  every valid pixel of every 5th keyframe with uid in lo..hi (300..600, visited once)
PSNR (upstream evaluation mask: ground truth > 0 and render > 0, no exposure correction) and depth L1 against the
sensor depth (depth in [min, max], alpha > 0,95) per region, before and after the refinement, on the pixels valid in
both maps. Pooled over the region (per-keyframe means are reported too).
Reading (initial thresholds of the plan): d_loop - d_single >= 0,5 dB, or the depth-L1 reduction in the loop region
>= 1,5 x the one of the single-visit region -> the refinement makes up for loop inconsistency; otherwise for
under-optimised online mapping.
Writes DIR/e1.jsonl.
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

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.dump import EventState, list_dumps, load_dump, load_frame  # noqa: E402
from deform.pipeline import delta_T_dict, inp_from_dump, rigid_result, with_pos  # noqa: E402
from deform.render_utils import DEV, det_scope, make_cam, render_subset  # noqa: E402
from e_v4_durability import measure, state_from_gauss, t0_masks  # noqa: E402
from h0_noise_floor import FinalState  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def psnr_of(se, n):
    return None if n == 0 else float(10.0 * np.log10(1.0 / max(se / n, 1e-12)))


class Acc:
    """Pooled sums of one region for the two maps."""

    def __init__(self):
        self.se = {"pre": 0.0, "post": 0.0}
        self.n_col = 0
        self.l1 = {"pre": 0.0, "post": 0.0}
        self.n_dep = 0
        self.n_pix = 0
        self.kf_psnr = {"pre": [], "post": []}
        self.kf_l1 = {"pre": [], "post": []}
        self.n_kf = 0

    def add(self, region, img, depth, rend, tr):
        """region [H,W] bool; rend = {name: (rgb [3,H,W], depth [H,W], alpha [H,W])}."""
        if not bool(region.any()):
            return
        self.n_kf += 1
        self.n_pix += int(region.sum())
        cm = (img > 0) & (rend["pre"][0] > 0) & (rend["post"][0] > 0) & region[None]
        dm = (depth > tr["depth_min_threshold"]) & (depth < tr["depth_max_threshold"]) & region
        dm &= (rend["pre"][2] > M.ALPHA_THR) & (rend["post"][2] > M.ALPHA_THR)
        nc, nd = int(cm.sum()), int(dm.sum())
        self.n_col += nc
        self.n_dep += nd
        for k in ("pre", "post"):
            if nc:
                se = float(((rend[k][0] - img)[cm] ** 2).sum())
                self.se[k] += se
                self.kf_psnr[k].append(psnr_of(se, nc))
            if nd:
                l1 = float((rend[k][1] - depth).abs()[dm].sum())
                self.l1[k] += l1
                self.kf_l1[k].append(l1 / nd)

    def out(self):
        r = {"n_kf": self.n_kf, "n_pix": self.n_pix, "n_col": self.n_col, "n_dep": self.n_dep}
        for k in ("pre", "post"):
            r[f"psnr_{k}"] = psnr_of(self.se[k], self.n_col)
            r[f"l1_{k}_mm"] = None if self.n_dep == 0 else 1e3 * self.l1[k] / self.n_dep
            r[f"psnr_kfmean_{k}"] = float(np.mean(self.kf_psnr[k])) if self.kf_psnr[k] else None
            r[f"l1_kfmean_{k}_mm"] = 1e3 * float(np.mean(self.kf_l1[k])) if self.kf_l1[k] else None
        if r["psnr_pre"] is not None:
            r["d_psnr"] = r["psnr_post"] - r["psnr_pre"]
            r["d_psnr_kfmean"] = r["psnr_kfmean_post"] - r["psnr_kfmean_pre"]
            d = np.array(self.kf_psnr["post"]) - np.array(self.kf_psnr["pre"])
            r["d_psnr_kf_median"], r["d_psnr_kf_q25"], r["d_psnr_kf_q75"] = float(np.median(d)), float(np.quantile(d, 0.25)), float(np.quantile(d, 0.75))
        if r["l1_pre_mm"] is not None:
            r["l1_drop_mm"] = r["l1_pre_mm"] - r["l1_post_mm"]
            r["l1_drop_rel"] = r["l1_drop_mm"] / r["l1_pre_mm"] if r["l1_pre_mm"] > 0 else None
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=int, default=300)
    ap.add_argument("--hi", type=int, default=600)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "deform", "selected_v6.yaml"))
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    user = load_deform_config(a.config)
    out_path = os.path.join(a.out, "e1.jsonl")
    open(out_path, "w").close()
    for rd in a.runs:
        rd = rd.rstrip("/")
        row = {"run": os.path.basename(rd)}
        p_pre, p_post = os.path.join(rd, "final_state.pt"), os.path.join(rd, "final_state_refined.pt")
        if not (os.path.exists(p_pre) and os.path.exists(p_post)):
            row["note"] = "map before or after the refinement is missing: run skipped"
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            continue
        pre = torch.load(p_pre, map_location="cpu", weights_only=False)
        post = torch.load(p_post, map_location="cpu", weights_only=False)
        dumps = [load_dump(p) for p in list_dumps(rd)]
        if not dumps:
            row["note"] = "no loop dump (intrinsics / loop region unavailable): run skipped"
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            continue
        intr = dumps[0]["intrinsics"]
        config = pre["config"]
        cfg = dcfg.resolve(user, config)
        tr = config["Training"]
        Hh, Wd = int(intr["H"]), int(intr["W"])
        st_f = FinalState(pre, intr)
        G = {"pre": state_from_gauss(pre["gaussians"]), "post": state_from_gauss(post["gaussians"])}
        poses = {"pre": {int(u): v for u, v in pre["poses"].items()}, "post": {int(u): v for u, v in post["poses"].items()}}
        kfs = sorted(int(u) for u in pre["keyframe_uids"])
        row["pose_diff_max_m"] = max(float((poses["pre"][u][:3, 3] - poses["post"][u][:3, 3]).norm()) for u in kfs)
        row["n_gauss"] = {k: int(G[k]["xyz"].shape[0]) for k in G}
        cam_meta = pre.get("cam_meta") or {}
        dkey = "rend_depth_expected" if tr.get("depth_type") == "expected" else "rend_depth_median"
        # ---- loop region: union of Pi*(k) ∩ valid on the final (pre-refinement) map
        loop_mask, events = {}, []
        with torch.no_grad(), det_scope(True):
            for d in dumps:
                st = EventState(d)
                split = 0.5 * (float(st.meta["loop_uid"]) + float(st.meta["cur_uid"]))
                inp = inp_from_dump(d, st.frame)
                dT = delta_T_dict(sorted(inp["kf_uids"]), inp["all_cam_ids"], inp["poses_pre"], inp["poses_pgo"])
                xyz_rig, rot_rig, poses_rig = rigid_result(inp, dT)
                G_rig = with_pos(state_from_gauss(st.gpre), xyz_rig, rot_rig)
                mk = t0_masks(st.gpre["t0"], split)
                JL, _ = M.select_loop_kfs(st, G_rig, poses_rig, mk[0], mk[1], cfg)
                ev = {"event": f'{st.meta["cur_uid"]}<->{st.meta["loop_uid"]}', "n_loop_kfs": len(JL)}
                if JL:
                    _, J_eval = M.split_opt_eval(JL, Hh, Wd, cfg["corr"]["checker"])
                    Pi, _ = M.pi_star(st, G_rig, poses_rig, [u for u, _ in J_eval], mk[0], mk[1], cfg)
                    r, valid = measure(st_f, G["pre"], poses["pre"], Pi, t0_masks(pre["gaussians"]["t0"], split))
                    ev.update({"n_pi": int(sum(int(m.sum()) for m, _ in Pi.values())), "n_pi_valid_final": int(sum(int(v.sum()) for v in valid.values())),
                               "gap_final_mm": r["median_mm"], "J_eval": sorted(int(u) for u in Pi)})
                    for u, v in valid.items():
                        loop_mask[int(u)] = v if int(u) not in loop_mask else (loop_mask[int(u)] | v)
                events.append(ev)
                del G_rig, inp
                torch.cuda.empty_cache()
            row["events"] = events
            single = [u for u in kfs if a.lo <= u <= a.hi][:: a.every]
            full = torch.ones((Hh, Wd), dtype=torch.bool, device=DEV)
            acc = {"loop": Acc(), "single": Acc(), "single_gated": Acc(), "loop_kf_full": Acc()}
            cell_b = M.stride_grid(Hh, Wd, cfg["corr"]["stride"]) & M.checker(Hh, Wd, cfg["corr"]["checker"], 1)
            for u in sorted(set(loop_mask) | set(single)):
                img, depth = load_frame(config, u, cam_meta)
                rend = {}
                for k in ("pre", "post"):
                    p = render_subset(make_cam(u, poses[k][u], intr), G[k])
                    rend[k] = (p["render"].clamp(0, 1), p[dkey][0], p["rend_alpha"][0])
                if u in loop_mask:
                    acc["loop"].add(loop_mask[u], img, depth, rend, tr)
                    acc["loop_kf_full"].add(full, img, depth, rend, tr)  # whole image of the loop keyframes (context)
                if u in single:
                    acc["single"].add(full, img, depth, rend, tr)
                    # control with the pixel selection of Pi* minus the two-layer requirement: cells B of the stride
                    # grid, opaque on the map before the refinement, away from depth edges
                    sel = cell_b & (rend["pre"][2] > M.ALPHA_THR) & ~M.edge_mask(rend["pre"][1], cfg["corr"]["g_max"])
                    acc["single_gated"].add(sel, img, depth, rend, tr)
        row["regions"] = {k: v.out() for k, v in acc.items()}
        L, S = row["regions"]["loop"], row["regions"]["single"]
        if L.get("d_psnr") is not None and S.get("d_psnr") is not None:
            row["d_psnr_loop_minus_single"] = L["d_psnr"] - S["d_psnr"]
            row["l1_drop_ratio_loop_over_single"] = (L["l1_drop_mm"] / S["l1_drop_mm"]) if S.get("l1_drop_mm") else None
            row["l1_drop_rel_ratio"] = (L["l1_drop_rel"] / S["l1_drop_rel"]) if S.get("l1_drop_rel") else None
            SG = row["regions"]["single_gated"]
            row["alt"] = {"kfmean_loop_minus_single": L["d_psnr_kfmean"] - S["d_psnr_kfmean"],
                          "kfmedian_loop_minus_single": L["d_psnr_kf_median"] - S["d_psnr_kf_median"],
                          "pooled_loop_minus_gated": L["d_psnr"] - SG["d_psnr"],
                          "kfmean_loop_minus_gated": L["d_psnr_kfmean"] - SG["d_psnr_kfmean"],
                          "kfmedian_loop_minus_gated": L["d_psnr_kf_median"] - SG["d_psnr_kf_median"],
                          "l1_drop_ratio_loop_over_gated": (L["l1_drop_mm"] / SG["l1_drop_mm"]) if SG.get("l1_drop_mm") else None}
            c1 = row["d_psnr_loop_minus_single"] >= 0.5
            c2 = row["l1_drop_ratio_loop_over_single"] is not None and S["l1_drop_mm"] > 0 and row["l1_drop_ratio_loop_over_single"] >= 1.5
            row["rule"] = {"psnr_gap_ge_0p5dB": bool(c1), "l1_drop_ge_1p5x": bool(c2), "verdict": "loop" if (c1 or c2) else "online"}
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        fm = lambda x, nd=2: "—" if x is None else f"{x:.{nd}f}"  # noqa: E731
        print(f"== {row['run']}  poses pre/post differ by {row['pose_diff_max_m']:.1e} m  Gaussians {row['n_gauss']['pre']} -> {row['n_gauss']['post']}  events {[(e['event'], e.get('n_pi_valid_final')) for e in events]}")
        for k, r in row["regions"].items():
            print(f"   {k:12s} kf {r['n_kf']:3d} px {r['n_pix']:8d}  PSNR {fm(r.get('psnr_pre'))} -> {fm(r.get('psnr_post'))} (d {fm(r.get('d_psnr'))}; per-kf mean d {fm(r.get('d_psnr_kfmean'))}, median d {fm(r.get('d_psnr_kf_median'))})  "
                  f"depth L1 {fm(r.get('l1_pre_mm'), 1)} -> {fm(r.get('l1_post_mm'), 1)} mm (drop {fm(r.get('l1_drop_mm'), 1)} mm, {fm(None if r.get('l1_drop_rel') is None else 100 * r['l1_drop_rel'], 0)} %)")
        print(f"   d_loop - d_single = {fm(row.get('d_psnr_loop_minus_single'))} dB; L1 drop loop / single = {fm(row.get('l1_drop_ratio_loop_over_single'))} -> {row.get('rule')}", flush=True)
        print("   other aggregations:", {k: round(v, 2) for k, v in (row.get("alt") or {}).items() if v is not None}, flush=True)
        del G, pre, post
        torch.cuda.empty_cache()
    print("[e1] done ->", out_path)


if __name__ == "__main__":
    main()
