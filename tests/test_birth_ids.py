"""birth_kfIDs (t^0) bookkeeping through add -> densify (clone + split) -> prune.

Run: python tests/test_birth_ids.py   (small GPU model, < 100 MB VRAM; pytest also works)
"""
import os
import sys

import torch
from munch import munchify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gaussian_splatting"))

from utils.config_utils import load_config  # noqa: E402
from utils.seed_utils import seed_everything  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}")


def _model():
    cfg = load_config(os.path.join(ROOT, "configs/tum/fr1_room.yaml"))
    g = GaussianModel(cfg["model_params"]["sh_degree"], cfg["model_params"]["initial_opacity"])
    g.training_setup(munchify(cfg["opt_params"]))
    return g


def _add(g, n, cam_id, offset):
    dev = "cuda"
    xyz = torch.rand(n, 3, device=dev) + offset
    depth = torch.rand(n, 1, device=dev) + 1.0
    dist = torch.full((n,), 0.01, device=dev)
    normals = torch.nn.functional.normalize(torch.randn(n, 3, device=dev), dim=-1)
    colors = torch.rand(n, 3, device=dev)
    g.add_gaussians(depth, xyz, dist, normals, colors, cam_id)


def test_birth_ids_survive_add_densify_prune():
    torch.cuda.set_per_process_memory_fraction(0.06)  # ~1 GB on a 16 GB card
    seed_everything(0)
    g = _model()
    _add(g, 500, cam_id=0, offset=0.0)
    _add(g, 400, cam_id=4, offset=2.0)
    assert g.birth_kfIDs.shape[0] == 900
    assert g.check_bookkeeping()
    assert int((g.birth_kfIDs == 0).sum()) == 500 and int((g.birth_kfIDs == 4).sum()) == 400

    # later observation overwrites t^c and t^l but must not touch t^0 (mimics update_state)
    later = (g.birth_kfIDs == 0).squeeze(1)
    g.unique_kfIDs[later] = 10
    g.last_observe_ids[later] = 12
    assert g.check_bookkeeping()

    # densify: half clone (small scale), half split (large scale)
    with torch.no_grad():
        g._scaling[:450] = torch.log(torch.tensor(1e-4, device="cuda"))
        g._scaling[450:] = torch.log(torch.tensor(0.5, device="cuda"))
    sel = torch.zeros(900, dtype=torch.bool, device="cuda")
    sel[::3] = True
    parent_birth = g.birth_kfIDs.clone()
    g.densify(sel)
    assert g.check_bookkeeping()
    n_clone = int((sel[:450]).sum())
    n_split_parents = int((sel[450:]).sum())
    assert g.get_xyz.shape[0] == 900 + n_clone + n_split_parents  # split: +2 children -1 parent
    # every Gaussian still carries a birth id that existed before densification
    assert set(torch.unique(g.birth_kfIDs).tolist()) <= set(torch.unique(parent_birth).tolist())
    # children of birth-4 parents keep birth 4 (all split parents were born at 4 or 0)
    assert int((g.birth_kfIDs == 4).sum()) >= 400

    # prune
    keep_n = g.get_xyz.shape[0]
    pm = torch.zeros(keep_n, dtype=torch.bool, device="cuda")
    pm[::5] = True
    before = g.birth_kfIDs[~pm].clone()
    g.prune_points(pm)
    assert g.check_bookkeeping()
    assert torch.equal(g.birth_kfIDs, before)


def test_check_bookkeeping_detects_errors():
    g = _model()
    _add(g, 50, cam_id=6, offset=0.0)
    g.unique_kfIDs[0] = 2  # t^c earlier than t^0 -> invalid
    _raises(AssertionError, g.check_bookkeeping)
    g.unique_kfIDs[0] = 6
    g.birth_kfIDs = g.birth_kfIDs[:-1]
    _raises(AssertionError, g.check_bookkeeping)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
