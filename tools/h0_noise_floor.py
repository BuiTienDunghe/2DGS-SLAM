"""plan v5 step B, H0: noise floor of the two-layer gap metric on a single-visit region of the final map.

usage: python tools/h0_noise_floor.py --out DIR RUN_DIR [RUN_DIR ...] [--lo 260 --hi 600]
On final_state.pt (before refinement) of each run: keyframes with uid in [lo, hi] (fr1/room: visited once; the short
loop 634<->702 is excluded) are the evaluation views. The Gaussians are split into two interleaved halves by the
parity of the birth-rank of t0 (t0 values are all even: frame step 2), the halves are rendered as the two layers,
gated like eval_pixel_set (both alpha > 0.95, |dD| < eps_eval, normals, edges, stride) and the median |gap| is the
floor: two renderings of the same surface. Writes DIR/h0.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.dump import list_dumps, load_dump  # noqa: E402
from deform.render_utils import DEV, make_cam  # noqa: E402
from e_v4_durability import state_from_gauss  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FinalState:
    def __init__(self, final, intr):
        self.intr = intr
        self.poses = {int(u): v for u, v in final["poses"].items()}
        self.kf_uids = [int(u) for u in final["keyframe_uids"]]

    def cam(self, uid, which=None, poses=None):
        P = poses if poses is not None else self.poses
        return make_cam(uid, P[uid], self.intr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=int, default=260)
    ap.add_argument("--hi", type=int, default=600)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "deform", "selected_v5.yaml"))
    ap.add_argument("--split", default="rank", choices=["rank", "random"], help="rank: birth-rank parity (keyframe-wise); random: 50/50 per Gaussian")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    user = load_deform_config(a.config)
    out = {}
    for rd in a.runs:
        final = torch.load(os.path.join(rd, "final_state.pt"), map_location="cpu", weights_only=False)
        d0 = load_dump(list_dumps(rd)[0])
        cfg = dcfg.resolve(user, d0["config"])
        st = FinalState(final, d0["intrinsics"])
        gf = final["gaussians"]
        G = state_from_gauss(gf)
        t0 = gf["t0"].reshape(-1).long()
        uniq, inv = torch.unique(t0, return_inverse=True)
        if a.split == "rank":
            half_a = (inv % 2 == 0).to(DEV)
        else:  # per-Gaussian random split: the same density halving, but no keyframe-wise depth bias
            half_a = (torch.rand(t0.shape[0], generator=torch.Generator().manual_seed(0)) < 0.5).to(DEV)
        half_b = ~half_a
        views = [u for u in st.kf_uids if a.lo <= u <= a.hi]
        H, W = int(st.intr["H"]), int(st.intr["W"])
        J = [(u, None) for u in views]
        Pi = M.eval_pixel_set(st, G, st.poses, J, half_a, half_b, cfg)
        maps = M.gap_maps(st, G, st.poses, Pi, half_a, half_b)
        e, n = M.edl_paired(Pi, {"floor": maps})
        per_view = {}
        for u in views:
            m = Pi[u][0] & maps[u]["valid"]
            x = maps[u]["gap"][m]
            per_view[u] = None if x.numel() == 0 else 1e3 * float(x.median())
        out[os.path.basename(rd)] = {"n_views": len(views), "views": views, "n_pix": n, "floor_median_mm": None if e["floor"]["median"] is None else 1e3 * e["floor"]["median"],
                                     "floor_p90_mm": None if e["floor"]["p90"] is None else 1e3 * e["floor"]["p90"], "per_view_mm": per_view,
                                     "n_half_a": int(half_a.sum()), "n_half_b": int(half_b.sum())}
        print(os.path.basename(rd), f"views {len(views)} pix {n} floor median {out[os.path.basename(rd)]['floor_median_mm']} mm p90 {out[os.path.basename(rd)]['floor_p90_mm']} mm")
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(a.out, f"h0_{a.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
