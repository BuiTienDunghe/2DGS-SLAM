"""Deformation config (Appendix A). Dataset-dependent defaults are picked from Dataset.type."""
import copy
import math

import yaml

DEFAULTS = {
    "mode": "rigid",
    "variant": "B",
    "seed_from_run": True,
    "seed": 0,
    "nodes": {"r_node": 0.30, "K": 4, "K_cand": 32, "beta": 10.0, "min_alpha": 0.1},
    "reliability": {"tau_alpha": 0.5, "n_min": 2, "n0": 3, "sigma_D": None, "sigma_n": 0.2,
                    "tau_S": 0.3, "s_min": 0.001, "s_max": "from_config", "contrib_min": 0.5,
                    "kf_step": 1, "time_budget_s": 30.0},
    "corr": {"eps_d": 0.05, "eps_eval": 0.10, "theta_n_deg": 30.0, "g_max": 0.03, "stride": None,
             "max_per_node": 64, "max_total": 50000, "max_loop_kfs": 20,
             "cov_layer": 0.10, "cov_both": 0.05, "alpha_thr": 0.95, "checker": 32,
             # plan v2 switches; the defaults reproduce the frozen A-1 path exactly
             # mode: pixel (A-1) | registration (D3, ICP per keyframe) | oracle (D1, pairs on the eval cells)
             "mode": "pixel", "residual": "p2l",
             # S gate: hard (min(S_old, S_new) >= tau, tau None -> reliability.tau_S) | weight (F1w) | off
             "s_gate": {"mode": "hard", "tau": None, "floor": 0.05, "layers": "both"},
             # keyframe: pairs only from J_opt | checkerboard (F3): + cells A (parity 0) of J_eval
             "split": "keyframe", "reg_use_eval": False,
             # plan v4 A1: per-node cap by a seeded random permutation (random, frozen A-1..v4) or by a fixed
             # integer hash of (keyframe, pixel) (hash); hash_salt is part of the frozen config
             "select": "random", "hash_salt": 0x9E3779B97F4A7C15,
             # K3 spread probe only: added to the run seed of the random cap (default 0 = frozen behaviour)
             "seed_offset": 0},
    # D3 registration (plan v2 §3.3); per-dataset Q2/Q3 thresholds below
    "reg": {"levels": [4, 2, 1], "max_iter": 20, "d_max": 0.10, "theta_deg": 30.0, "huber": 1.0,
            "q1_frac": 0.20, "q1_min": 3000, "q2_rms": None, "q3_t": None, "q3_r_deg": None,
            "q4_t": 0.05, "q4_r_deg": 2.0, "eps_conv": 1.0e-8},
    "energy": {"w_con": 1.0, "w_reg": 1.0, "w_p": 10.0, "sigma_c": None, "sigma_r": 0.01,
               "sigma_t": 0.02, "sigma_w_deg": 1.0},
    # tol_mode: legacy (frozen A-1) | energy (v2 diagnostics) | grad (plan v3: normalised energy,
    # stop at ||g|| <= grad_rtol ||g0||, at most max_iter_conv iterations)
    "solver": {"max_iter": 200, "rel_tol": 1.0e-7, "history": 20, "tol_mode": "legacy",
               "grad_rtol": 1.0e-6, "max_iter_conv": 3000, "history_conv": 20},
    "accept": {"min_corr": 500, "min_gain": 0.10, "max_disp": 0.10, "max_rot_deg": 5.0,
               "time_budget_s": 120.0, "enforce": True},
    # anchor (plan v4 A4): none (frozen A-1..v4) | prior -> after the pose sync the whole result (Gaussians and
    # poses) is moved by one rigid transform so that the prior-fixed frame keeps exactly its PGO pose
    # regen_odom (plan v5 A2): odometry factors touching a moved pose are re-measured from the written poses
    "pose_sync": {"method": "kabsch", "min_gauss": 200, "anchor": "none", "regen_odom": False},
    "diagnostics": {"stretch_samples": 10000, "stretch_h": 0.01},
    # plan v4 A3: render -> deformation renders use diff_surfel_rasterization_det (int64 fixed-point
    # contributions); torch -> torch.use_deterministic_algorithms inside the pipeline (restored after)
    "det": {"render": False, "torch": False},
    # plan v5 A1: two layers by the active mask (A-1..v5) or by birth keyframe t0 with the split
    # s_k = (loop_uid + cur_uid) / 2 (old: t0 < s_k, new: t0 >= s_k); used for pairs, node assignment and Pi*
    "layers": {"mode": "active"},
}

# values that differ between TUM and Replica (Appendix A)
PER_DATASET = {
    "tum": {"reliability.sigma_D": 0.010, "corr.stride": 4, "energy.sigma_c": 0.005,
            "reg.q2_rms": 0.010, "reg.q3_t": 0.002, "reg.q3_r_deg": 0.1},
    "replica": {"reliability.sigma_D": 0.005, "corr.stride": 8, "energy.sigma_c": 0.003,
                "reg.q2_rms": 0.003, "reg.q3_t": 0.001, "reg.q3_r_deg": 0.05},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(user_cfg, slam_config):
    """Full config for this dataset: DEFAULTS <- per-dataset <- user YAML (explicit values win)."""
    cfg = copy.deepcopy(DEFAULTS)
    dtype = slam_config["Dataset"]["type"]
    for dotted, v in PER_DATASET.get(dtype, PER_DATASET["tum"]).items():
        a, b = dotted.split(".")
        cfg[a][b] = v
    cfg = _merge(cfg, {k: v for k, v in (user_cfg or {}).items() if not k.startswith("_")})
    # per-dataset overrides written in the YAML as {tum: {...}, replica: {...}}
    if dtype in (user_cfg or {}):
        cfg = _merge(cfg, user_cfg[dtype])
    if cfg["reliability"]["s_max"] == "from_config":
        cfg["reliability"]["s_max"] = float(slam_config["Training"]["prune_size_threshold"])
    cfg["dataset_type"] = dtype
    cfg["corr"]["cos_theta_n"] = math.cos(math.radians(cfg["corr"]["theta_n_deg"]))
    return cfg


def load(path, slam_config):
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    return resolve(user, slam_config)
