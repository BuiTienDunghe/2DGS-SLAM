"""plan v5 T7: with every new switch off (selected_v5), the pipeline reproduces the v4 K1 snapshot bit for bit.

usage: python tests/test_v5_regression.py DUMP.pt SNAPSHOT_run0.pt [DEFORM_YAML=configs/deform/selected_v5.yaml]
The snapshot is one of results_exp/reports/v4/probe/K1_*_run0.pt (cand_v5 == selected_v5, tools/freeze_v5.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import torch  # noqa: E402
import yaml  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from deform import config as dcfg  # noqa: E402
from deform.dump import EventState, load_dump  # noqa: E402
from deform.pipeline import correct_map, inp_from_dump  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(dump_path, snap_path, yaml_path):
    torch.manual_seed(0)
    d = load_dump(dump_path)
    st = EventState(d)
    cfg = dcfg.resolve(yaml.safe_load(open(yaml_path)), d["config"])
    cfg["seed"] = int(d["meta"].get("seed", 0))
    res = correct_map(inp_from_dump(d, st.frame), cfg, cfg["variant"])
    snap = torch.load(snap_path, weights_only=False)
    dx = float((res["xyz"].cpu().double() - snap["xyz"].double()).abs().max())
    dt = None if res.get("node_t") is None or snap.get("node_t") is None else float((res["node_t"].double() - snap["node_t"].double()).abs().max())
    same_iters = res["log"].get("lbfgs_iters") == snap.get("lbfgs_iters")
    same_edl = (res["log"].get("edl_opt_init_mm"), res["log"].get("edl_opt_final_mm")) == tuple(snap.get("edl_opt"))
    print(f"xyz max diff {dx:.2e} m, node_t max diff {dt}, iters {res['log'].get('lbfgs_iters')} vs {snap.get('lbfgs_iters')}, "
          f"e_dl opt {(res['log'].get('edl_opt_init_mm'), res['log'].get('edl_opt_final_mm'))} vs {snap.get('edl_opt')}, "
          f"accepted {res['accepted']} vs {snap.get('accepted')}, layers {res['log'].get('layers')}")
    ok = dx == 0.0 and (dt is None or dt == 0.0) and same_iters and same_edl and res["accepted"] == snap.get("accepted")
    print("T7_PASS" if ok else "T7_FAIL")
    return ok


if __name__ == "__main__":
    y = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "configs", "deform", "selected_v5.yaml")
    sys.exit(0 if main(sys.argv[1], sys.argv[2], y) else 1)
