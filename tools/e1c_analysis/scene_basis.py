"""PLAN E1c P4: V_scene per tracked frame = obs_basis of ALL planes CAPE-lite measures on the dataset depth.
Same for every method (depends on depth only). CPU only.
usage: python tools/e1c_analysis/scene_basis.py [OUT_NPZ]
"""
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "gaussian_splatting")
from munch import munchify  # noqa: E402

from utils.config_utils import load_config  # noqa: E402
from utils.dataset import load_dataset  # noqa: E402
from utils.plane_anchor import PlaneExtractor  # noqa: E402
import importlib.util  # noqa: E402

# obs_decouple.py sits next to this tool tree (../../utils); loaded by path so the script also runs from a
# staging copy before the E1c branch exists
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils", "obs_decouple.py")
_spec = importlib.util.spec_from_file_location("obs_decouple", _p)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
obs_basis = _mod.obs_basis

out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/2DGS-SLAM/results_exp/e1c/scene_basis_fr3_snt_far.npz")
cfg = load_config("configs/tum/e1/fr3_snt_far.yaml")
pt = cfg["PlaneTracking"]
thr = float(pt.get("obs_threshold", pt["min_singular_value"]))
ds = load_dataset(munchify(cfg["model_params"]), "", config=cfg)
ds.device = "cpu"
ext = PlaneExtractor(pt)
b, s = int(cfg["Dataset"]["frame_begin"]), int(cfg["Dataset"]["frame_step"])
fids = list(range(b, len(ds), s))
Vt, St, Vr, Sr, NP = [], [], [], [], []
for f in fids:
    _, _, depth, _ = ds[f]
    planes = ext(depth, ds.K)
    o = obs_basis([p.n.numpy() for p in planes], thr)
    Vt.append(o["Vt"]); St.append(o["sig_t"]); Vr.append(o["Vr"]); Sr.append(o["sig_r"]); NP.append(len(planes))
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, frame_ids=np.array(fids), Vt=np.array(Vt), sig_t=np.array(St), Vr=np.array(Vr), sig_r=np.array(Sr),
         n_planes=np.array(NP), threshold=thr)
n_obs = (np.array(St) >= thr).sum(1)
print(f"{out}: {len(fids)} frames, n_obs_t counts {dict(zip(*np.unique(n_obs, return_counts=True)))}")
