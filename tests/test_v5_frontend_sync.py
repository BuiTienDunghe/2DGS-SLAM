"""plan v5 T6: after the hook's write-back, the frontend receives the written poses through the upstream path
(BackEnd.push_to_frontend_after_pgo -> FrontEnd.sync_backend) for every frame: keyframes of the sliding window
and the current (tracking) frame.

usage: python tests/test_v5_frontend_sync.py DUMP.pt DEFORM_YAML
The fake backend (tests/test_online_hook_v4.py, real gtsam graph) runs the hook; the message is built exactly like
push_to_frontend_after_pgo; a namespace frontend with stale cameras runs the real FrontEnd.sync_backend.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import torch  # noqa: E402
import yaml  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform.dump import EventState, load_dump  # noqa: E402
from deform.online import backend_correct  # noqa: E402
from utils.io_utils import clone_obj  # noqa: E402
from utils.slam_frontend import FrontEnd  # noqa: E402
import test_online_hook_v4 as T4  # noqa: E402


def main(dump_path, yaml_path):
    torch.manual_seed(0)
    d = load_dump(dump_path)
    st = EventState(d)
    cfg = dcfg.resolve(yaml.safe_load(open(yaml_path)), d["config"])
    cfg["seed"] = int(d["meta"].get("seed", 0))
    be = T4.fake_backend_real_pgo(d, st)
    cur, loop = be.key_cameras[int(d["meta"]["cur_uid"])], be.key_cameras[int(d["meta"]["loop_uid"])]
    # stale frontend: cameras hold the PRE poses (what the frontend had before PGO)
    fe = types.SimpleNamespace(cameras={u: clone_obj(c) for u, c in be.all_cameras.items()}, gaussians=None, loop_uid_pairs=[])
    applied, log = backend_correct(be, cur, loop, cfg)
    assert applied, f"hook not applied: {log.get('fallback_reason')}"
    # message exactly as BackEnd.push_to_frontend_after_pgo (pgo_with_all_frames=True path)
    cameras = [clone_obj(be.all_cameras[u]) for u in be.all_cam_ids]
    msg = ["pgo", None, cameras, []]
    FrontEnd.sync_backend(fe, msg)
    window = [int(u) for u in d["cam_sliding_window"]] + [int(d["meta"]["cur_uid"])]
    worst_bits = 0.0
    moved = 0
    for u in window:
        T_be, T_fe = be.all_cameras[u].T, fe.cameras[u].T
        worst_bits = max(worst_bits, float((T_be - T_fe).abs().max()))
        # the written pose must differ from the stale (pre-PGO) one for most frames, else the test proves nothing
        P_pre = d["poses_pre"][u].double()
        moved += int(float((torch.linalg.inv(T_fe.double()).cpu()[:3, 3] - P_pre[:3, 3]).norm()) > 1e-6)
    n_all = sum(1 for u in be.all_cam_ids if torch.equal(be.all_cameras[u].T, fe.cameras[u].T))
    print(f"window+current ({len(window)} frames): max |T_fe - T_be| = {worst_bits:.2e}; frames that changed vs pre-PGO: {moved}/{len(window)}; "
          f"all frames identical to the backend: {n_all}/{len(be.all_cam_ids)}")
    ok = worst_bits == 0.0 and n_all == len(be.all_cam_ids) and moved > 0
    print("T6_PASS" if ok else "T6_FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
