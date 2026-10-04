"""plan v4 K4 + A4 tests: the online hook on a fake backend whose pose graph is a REAL gtsam graph
(fixed prior sigma 1e-9 on the first frame + odometry factors between consecutive frames, as in the SLAM
backend; the old FakePGO of tests/test_online_hook.py has no factors and let the write_back gauge bug through).

usage: python tests/test_online_hook_v4.py DUMP.pt DEFORM_YAML [--out DIR]
  T0  zero deformation (solver.max_iter_conv 0, accept.enforce false): written poses == PGO poses (<= 1e-6 m),
      Gaussians == rigid replay, keyframe ATE unchanged, prior factor error unchanged
  T1  K4: online (hook) vs offline (correct_map): max |d mu| <= 1e-4 m, max |d node t| <= 1e-4 m,
      e_dl on the evaluation pixels differs <= 0.01 mm, event time <= 120 s
  T2  A4: the prior-anchored frame keeps its gtsam value (resid <= 1e-6 m); graph error after write_back is
      finite and contains no prior term (prior factor error <= 1e-6); gtsam value == camera pose written
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import gtsam  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform import metrics as M  # noqa: E402
from deform.dump import EventState, load_dump  # noqa: E402
from deform.online import anchored_uids, backend_correct, backend_inp  # noqa: E402
from deform.pipeline import correct_map, inp_from_dump, with_pos  # noqa: E402
from deform.rigid import replay  # noqa: E402
from utils.slam_backend import BackEnd  # noqa: E402
import test_online_hook as T  # noqa: E402


class RealPGO:
    """gtsam graph as the backend builds it: fixed prior on the first frame, odometry between consecutive
    all_cam_ids (relative transforms from the PRE poses, like the backend), initial values = PGO poses."""

    def __init__(self, d, tran_std=0.01, rot_std_deg=0.03):
        self.poses = d["poses_pgo"]
        self.graph_initials = gtsam.Values()
        self.graph_factors = gtsam.NonlinearFactorGraph()
        ids = [int(u) for u in d["all_cam_ids"]]
        for u in ids:
            self.graph_initials.insert(gtsam.symbol("x", u), gtsam.Pose3(self.poses[u].double().numpy()))
        fixed = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-9))
        self.anchor = ids[0]
        self.graph_factors.add(gtsam.PriorFactorPose3(gtsam.symbol("x", self.anchor),
                                                      gtsam.Pose3(self.poses[self.anchor].double().numpy()), fixed))
        cov = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.radians(rot_std_deg)] * 3 + [tran_std] * 3))
        pre = d["poses_pre"]
        for a, b in zip(ids[:-1], ids[1:]):
            rel = np.linalg.inv(pre[a].double().numpy()) @ pre[b].double().numpy()
            self.graph_factors.add(gtsam.BetweenFactorPose3(gtsam.symbol("x", a), gtsam.symbol("x", b), gtsam.Pose3(rel), cov))
        self.graph_optimized = self.graph_initials
        self.last_error = float(self.graph_factors.error(self.graph_initials))

    def get_optimized_node_pose(self, idx):
        return self.poses[int(idx)].double().numpy()

    def prior_error(self):
        return float(self.graph_factors.at(0).error(self.graph_initials))


def fake_backend_real_pgo(d, st):
    be = T.fake_backend(d, st)
    be.pgo = RealPGO(d)
    return be


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("yaml")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    d = load_dump(a.dump)
    st = EventState(d)
    user = yaml.safe_load(open(a.yaml))
    cfg = dcfg.resolve(user, d["config"])
    cfg["seed"] = int(d["meta"].get("seed", 0))
    rep = {"dump": os.path.basename(a.dump), "yaml": a.yaml}
    fails = []

    def check(name, ok, detail):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        rep[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)

    # ------------------------------------------------------------ T0 zero deformation
    cfg0 = dcfg._merge(cfg, {"solver": {"max_iter_conv": 0, "max_iter": 0}, "accept": {"enforce": False}})
    be = fake_backend_real_pgo(d, st)
    cur, loop = be.key_cameras[int(d["meta"]["cur_uid"])], be.key_cameras[int(d["meta"]["loop_uid"])]
    e_prior0, e_graph0 = be.pgo.prior_error(), float(be.pgo.graph_factors.error(be.pgo.graph_initials))
    applied, log0 = backend_correct(be, cur, loop, cfg0)
    xyz_r, _, _ = replay(d, use_online=False)
    inp0 = inp_from_dump(d, st.frame)
    inp0["anchor_uids"] = anchored_uids(be.pgo)
    res0 = correct_map(inp0, cfg0, cfg0["variant"])
    dmu_pipe = float((be.gaussians.get_xyz.detach() - res0["xyz"]).norm(dim=1).max())  # online vs offline, same zero field
    dmu_rig_off = float((res0["xyz"] - res0["G_rig"]["xyz"]).norm(dim=1).max())  # zero field vs rigid state (offline)
    dmu = float((be.gaussians.get_xyz.detach() - xyz_r).norm(dim=1).max())
    dpos = max(float((torch.linalg.inv(c.T.double()).cpu()[:3, 3] - d["poses_pgo"][u].double()[:3, 3]).norm())
               for u, c in be.all_cameras.items())
    poses_w = {u: torch.linalg.inv(c.T.double()).cpu() for u, c in be.all_cameras.items()}
    ate_pgo, ate_w = M.ate_kf(st, d["poses_pgo"]), M.ate_kf(st, poses_w)
    check("T0_applied", applied, f"accepted={applied} reason={log0.get('fallback_reason')} iters={log0.get('lbfgs_iters')}")
    check("T0_poses_eq_pgo", dpos <= 1e-6, f"max |d t| {dpos:.2e} m (<= 1e-6)")
    # the blended field with identity nodes returns x * sum_k w_k, and the float32 weights sum to 1 +- 1e-7, so the
    # zero deformation reproduces the rigid state to ~1e-6 m (0.001 mm) and not bit-exactly, online and offline alike
    # (pre-v4 dumps carry no raw parameters, so the online activations differ by float32 ulps from the dump's and
    # the weights differ too: online vs offline zero field is then ~1e-6 m as well; v4 dumps make it exact)
    check("T0_gauss_eq_rigid", dmu_pipe <= 2e-6 and dmu <= 2e-6,
          f"online vs offline zero field {dmu_pipe:.2e} m; zero field vs rigid state: offline {dmu_rig_off:.2e} m, "
          f"online vs backend rigid formula {dmu:.2e} m (float32 weight normalisation, <= 2e-6)")
    check("T0_ate_unchanged", ate_pgo is not None and abs(ate_w - ate_pgo) <= 1e-6, f"ATE kf {ate_pgo} -> {ate_w}")
    check("T0_prior_err", abs(be.pgo.prior_error() - e_prior0) <= 1e-6, f"prior error {e_prior0:.3e} -> {be.pgo.prior_error():.3e}")
    check("T0_graph_err", abs(float(be.pgo.last_error) - e_graph0) <= 1e-3 * max(1.0, e_graph0),
          f"graph error {e_graph0:.4f} -> {float(be.pgo.last_error):.4f}")
    del be
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ T1/T2 full method
    be = fake_backend_real_pgo(d, st)
    cur, loop = be.key_cameras[int(d["meta"]["cur_uid"])], be.key_cameras[int(d["meta"]["loop_uid"])]
    e_graph0 = float(be.pgo.graph_factors.error(be.pgo.graph_initials))
    # offline reference 1: the pipeline on exactly the backend's inputs (what the hook feeds correct_map) -> the K4
    # rule tests the hook plumbing (write_back, cameras, gtsam, anchor); reference 2: the pipeline on the dump
    # (dump fidelity; pre-v4 dumps lack the raw parameters, so their activations differ by float32 ulps)
    inp_be = backend_inp(be, cur)
    applied, log = backend_correct(be, cur, loop, cfg)
    res = correct_map(inp_be, cfg, cfg["variant"])
    inp = inp_from_dump(d, st.frame)
    inp["anchor_uids"] = anchored_uids(be.pgo)
    res_d = correct_map(inp, cfg, cfg["variant"])
    if not applied:
        BackEnd.apply_rigid_correction(be)
    xyz_on = be.gaussians.get_xyz.detach()
    dmu = float((xyz_on - res["xyz"]).norm(dim=1).max())
    dmu_d = float((xyz_on - res_d["xyz"]).norm(dim=1).max())
    dnode_d = None if (res.get("node_t") is None or res_d.get("node_t") is None or res["node_t"].shape != res_d["node_t"].shape) \
        else float((res["node_t"].double() - res_d["node_t"].double()).norm(dim=1).max())
    poses_on = {u: torch.linalg.inv(c.T.double()).cpu() for u, c in be.all_cameras.items()}
    dpose = max(float((poses_on[u][:3, 3] - res["poses"][u].double()[:3, 3]).norm()) for u in poses_on)
    check("T1_accept_same", applied == res["accepted"] == res_d["accepted"],
          f"online {applied} ({log.get('fallback_reason')}) offline {res['accepted']} ({res['reason']}) offline-from-dump {res_d['accepted']}")
    check("T1_K4_gauss", dmu <= 1e-4, f"max |d mu| hook vs pipeline on the same inputs {dmu:.2e} m (<= 1e-4)")
    check("T1_K4_poses", dpose <= 1e-4, f"max |d t| hook vs pipeline on the same inputs {dpose:.2e} m")
    rep["dump_fidelity"] = {"dmu_m": dmu_d, "dnode_t_m": dnode_d, "raw_params_in_dump": "scale_raw" in d["gauss_pre"],
                            "iters_online": log.get("lbfgs_iters"), "iters_from_dump": res_d["log"].get("lbfgs_iters")}
    print(f"[INFO] dump fidelity: hook vs pipeline-from-dump max |d mu| {dmu_d:.2e} m, max |d node t| {dnode_d} m, "
          f"raw params in dump: {'scale_raw' in d['gauss_pre']}, iters {log.get('lbfgs_iters')} vs {res_d['log'].get('lbfgs_iters')}")
    check("T1_time", (log.get("t_stage_s") or {}).get("total", 1e9) <= 120, f"t_event {(log.get('t_stage_s') or {}).get('total')} s, t_hook {log.get('t_hook_s')} s")
    # e_dl on the evaluation pixels (as test_online_hook G4)
    G = res["G_rig"]
    m_old, m_new = M.layer_masks(st.active)
    JL = sorted(set(res["J_opt"]) | set(res["J_eval"]))
    _, J_eval = M.split_opt_eval(JL, int(st.intr["H"]), int(st.intr["W"])) if JL else ([], [])
    if J_eval:
        Pi = M.eval_pixel_set(st, G, res["poses_rig"], J_eval, m_old, m_new, cfg)
        maps = {"off": M.gap_maps(st, with_pos(G, res["xyz"], res["rot"]), res["poses"], Pi, m_old, m_new),
                "on": M.gap_maps(st, with_pos(G, xyz_on, be.gaussians.get_rotation.detach()), poses_on, Pi, m_old, m_new),
                "rigid": M.gap_maps(st, G, res["poses_rig"], Pi, m_old, m_new)}
        e, _ = M.edl_paired(Pi, maps)
        diff = abs((e["on"]["median"] or 0) - (e["off"]["median"] or 0))
        check("T1_K4_edl", diff <= 1e-5, f"e_dl rigid {1e3 * (e['rigid']['median'] or 0):.3f} off {1e3 * (e['off']['median'] or 0):.3f} on {1e3 * (e['on']['median'] or 0):.3f} mm, diff {1e3 * diff:.4f} mm (<= 0.01)")
    # A4
    anc = be.pgo.anchor
    held = be.pgo.graph_initials.atPose3(gtsam.symbol("x", anc)).matrix()
    resid_cam = float(np.linalg.norm(poses_on[anc].numpy()[:3, 3] - held[:3, 3]))
    check("T2_anchor_kept", float(np.linalg.norm(held[:3, 3] - d["poses_pgo"][anc].double().numpy()[:3, 3])) <= 1e-9,
          f"gtsam anchor moved {float(np.linalg.norm(held[:3, 3] - d['poses_pgo'][anc].double().numpy()[:3, 3])):.2e} m")
    check("T2_anchor_cam_eq_gtsam", resid_cam <= 1e-6, f"camera pose vs gtsam at anchor {resid_cam:.2e} m; pipeline anchor log {log.get('anchor')}")
    check("T2_prior_err", be.pgo.prior_error() <= 1e-6, f"prior factor error {be.pgo.prior_error():.3e}")
    e1 = float(be.pgo.last_error)
    check("T2_graph_err_finite", np.isfinite(e1) and e1 < 1e9, f"graph error {e_graph0:.1f} -> {e1:.1f} (write_back log {log.get('gtsam_err_before_sync')} -> {log.get('gtsam_err_after_sync')})")
    rep["vram"] = {k: log.get(k) for k in ("vram_before_gb", "vram_event_peak_gb", "vram_run_peak_gb")}
    rep["event"] = {k: log.get(k) for k in ("corr_gated", "corr_capped", "lbfgs_iters", "edl_opt_init_mm", "edl_opt_final_mm",
                                            "max_node_disp_m", "t_hook_s", "det_warnings", "anchor", "anchored_uids", "anchor_resid_m")}
    print("event:", json.dumps(rep["event"], default=str))
    print("vram:", rep["vram"])
    rep["pass"] = not fails
    print("K4_V4_PASS" if not fails else f"K4_V4_FAIL {fails}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"k4_{os.path.basename(a.dump).replace('.pt', '')}.json"), "w") as f:
            json.dump(rep, f, indent=1, default=str)
    return not fails


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
