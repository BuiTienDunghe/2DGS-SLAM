"""G4 diagnosis: online hook vs offline, against the offline-vs-offline spread of the same config.

usage: python tools/g4_noise.py DEFORM_YAML DUMP.pt [DUMP.pt ...]
For each dump: two offline correct_map runs (fresh caches) and one online backend_correct run (fake backend
as in tests/test_online_hook.py). Prints max / p99.9 / count(> 1e-4 m) of |d mu| for off1-off2 and on-off1,
the acceptance of each, and pair counts. Diagnostic only; the G4 threshold is not changed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
import deform  # noqa: F401,E402
import torch  # noqa: E402
import yaml  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform.dump import EventState, load_dump  # noqa: E402
from deform.online import backend_correct  # noqa: E402
from deform.pipeline import correct_map, inp_from_dump  # noqa: E402
import test_online_hook as T  # noqa: E402


def stats(a, b):
    d = (a - b).norm(dim=1)
    return (f"max {float(d.max()):.2e} m, p99.9 {float(torch.quantile(d.float()[torch.randperm(d.numel(), device=d.device)[:200000]], 0.999)):.2e} m, "
            f"n>1e-4 {int((d > 1e-4).sum())}/{d.numel()}")


def main():
    path_cfg = sys.argv[1]
    user = yaml.safe_load(open(path_cfg))
    for p in sys.argv[2:]:
        torch.manual_seed(0)
        d = load_dump(p)
        st = EventState(d)
        cfg = dcfg.resolve(user, d["config"])
        cfg["seed"] = int(d["meta"].get("seed", 0))
        r1 = correct_map(inp_from_dump(d, st.frame), cfg, cfg["variant"])
        r2 = correct_map(inp_from_dump(d, st.frame), cfg, cfg["variant"])
        be = T.fake_backend(d, st)
        cur = be.key_cameras[int(d["meta"]["cur_uid"])]
        loop = be.key_cameras[int(d["meta"]["loop_uid"])]
        applied, log = backend_correct(be, cur, loop, cfg)
        on = be.gaussians.get_xyz.detach()
        keys = ("corr_raw", "corr_gated", "corr_capped", "n_nodes", "n_edges", "lbfgs_iters")
        print(os.path.basename(p), os.path.basename(path_cfg), f"accepted off1 {r1['accepted']} off2 {r2['accepted']} online {applied}")
        for lab, lg in (("off1", r1["log"]), ("off2", r2["log"]), ("online", log)):
            print(f"   {lab:6s}", {k: lg.get(k) for k in keys})
        print("   offline vs offline:", stats(r1["xyz"], r2["xyz"]))
        print("   online  vs offline:", stats(on, r1["xyz"]))
        del be, r1, r2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
