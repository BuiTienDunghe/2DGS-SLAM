"""plan v5 T3: birth_kfIDs follow each Gaussian through densify_and_clone, densify_and_split, prune_points and
update_after_pgo of GaussianModel (the layer split by t0 relies on it).

usage: python tests/test_v5_birth_ids.py DUMP.pt   (the dump only provides the model / optimizer config)
Each Gaussian gets birth id = its own index (unique_kfIDs = last_observe_ids = index + 1000, so t0 <= tc holds), then
the four operations run and the expected id sequence is checked exactly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import torch  # noqa: E402
from munch import munchify  # noqa: E402
from torch import nn  # noqa: E402

from scene.gaussian_model import GaussianModel  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def make_model(cfg, N, dev="cuda"):
    g = GaussianModel(cfg["model_params"]["sh_degree"], cfg["model_params"]["initial_opacity"])
    gen = torch.Generator().manual_seed(0)
    g._xyz = nn.Parameter(torch.randn(N, 3, generator=gen).to(dev))
    g._rotation = nn.Parameter(torch.nn.functional.normalize(torch.randn(N, 4, generator=gen), dim=-1).to(dev))
    g._scaling = nn.Parameter(torch.log(torch.rand(N, 2, generator=gen) * 0.05 + 0.01).to(dev))
    g._opacity = nn.Parameter(torch.zeros(N, 1).to(dev))
    g._features_dc = nn.Parameter(torch.rand(N, 1, 3, generator=gen).to(dev))
    g._features_rest = nn.Parameter(torch.zeros((N, 0, 3), device=dev))
    idx = torch.arange(N, device=dev, dtype=torch.int)[:, None]
    g.unique_kfIDs = idx + 1000
    g.last_observe_ids = idx + 1000
    g.birth_kfIDs = idx.clone()
    g.min_observed_depth = torch.ones(N, 1, device=dev)
    g.active_mask = torch.ones(N, 1, dtype=torch.bool, device=dev)
    g.training_setup(munchify(cfg["opt_params"]))
    return g


def main(dump_path):
    d = torch.load(dump_path, map_location="cpu", weights_only=False)
    cfg = d["config"]
    N = 50
    g = make_model(cfg, N)
    dev = g._xyz.device
    birth = lambda: g.birth_kfIDs.reshape(-1).tolist()  # noqa: E731
    # 1. clone: ids of the appended copies == ids of the cloned ones, in order
    mask = torch.zeros(N, dtype=torch.bool, device=dev); mask[[3, 7, 11, 20]] = True
    g.densify_and_clone(mask)
    exp = list(range(N)) + [3, 7, 11, 20]
    check("T3.1 clone keeps birth ids", birth() == exp, f"{birth()[-6:]}")
    g.check_bookkeeping()
    # 2. split: N copies appended per selected point, originals pruned
    M = g._xyz.shape[0]
    sel = torch.zeros(M, dtype=torch.bool, device=dev); sel[[0, 5, M - 1]] = True
    before = birth()
    g.densify_and_split(sel, N=2)
    kept = [b for i, b in enumerate(before) if not sel[i]]
    appended = [before[i] for i in (0, 5, M - 1)] * 2  # repeat(N, 1) -> [a, b, c, a, b, c]
    check("T3.2 split keeps birth ids", birth() == kept + appended, f"n {len(birth())} tail {birth()[-6:]}")
    g.check_bookkeeping()
    # 3. prune
    M = g._xyz.shape[0]
    pm = torch.zeros(M, dtype=torch.bool, device=dev); pm[1::7] = True
    before = birth()
    g.prune_points(pm)
    exp = [b for i, b in enumerate(before) if not pm[i]]
    check("T3.3 prune keeps birth ids", birth() == exp, f"n {len(birth())}")
    g.check_bookkeeping()
    # 4. update_after_pgo: ids untouched, xyz moved by the per-Gaussian transform
    M = g._xyz.shape[0]
    T = torch.eye(4, device=dev).repeat(M, 1, 1); T[:, :3, 3] = 0.1
    before = birth(); x0 = g.get_xyz.detach().clone()
    g.update_after_pgo(T)
    check("T3.4 update_after_pgo keeps birth ids", birth() == before and bool(torch.allclose(g.get_xyz.detach(), x0 + 0.1, atol=1e-6)))
    g.check_bookkeeping()
    # 5. birth <= unique and <= last_observe for every Gaussian (bookkeeping invariant)
    check("T3.5 invariants", bool((g.birth_kfIDs <= g.unique_kfIDs).all() and (g.birth_kfIDs <= g.last_observe_ids).all()))
    print("T3_PASS" if not FAILS else f"T3_FAIL {FAILS}")
    return not FAILS


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
