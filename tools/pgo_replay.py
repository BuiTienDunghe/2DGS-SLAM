"""Offline reconstruction of the backend's pose graph from loop dumps (plan v5 step B, H1).

Odometry factors join consecutive frames of all_cam_ids with measurement inv(T_last) T_cur taken from the TRACKED
poses; a frame's tracked pose is poses_pre of the earliest dump whose cur_uid >= frame (poses only change at PGO
events). The prior is the fixed one on the first frame.

Loop measurements are not stored before v6 dumps. They are RECOVERED from the stationarity of the real PGO solution:
at the optimum x* of event e, the gradient of the whole graph vanishes at the current frame c_e, whose only factors are
the odometry to its predecessor and the new loop factor (l_e, c_e); so the loop's whitened residual w solves
A_c^T w = -grad_rest(c_e), where A_c is the loop factor's whitened Jacobian w.r.t. c_e, and the measurement follows from
m = hx Exp(-r), r = sigma * w, hx = T_l^-1 T_c (gtsam BetweenFactor: r = Log(m^-1 hx)). Two or three fixed-point
iterations refine the Jacobian. Loops are recovered event by event, each with the previous ones already in the graph.
Verification per loop: gradient norm at x* with the recovered factor, and the displacement of an LM run started at x*.
When a dump carries loop_transform (v6 dumps) that value is used instead and the recovery is only a check.
"""
import json
import os

import gtsam
import numpy as np
import torch


def sym(u):
    return gtsam.symbol("x", int(u))


def mat(P):
    return P.double().numpy() if torch.is_tensor(P) else np.asarray(P, dtype=np.float64)


def rel(Pa, Pb):
    return np.linalg.inv(mat(Pa)) @ mat(Pb)


def load_attempts(rd):
    p = os.path.join(rd, "loop_attempts.jsonl")
    rows = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
    out = {}
    for r in rows:
        kept = r.get("loop_factor_in_graph")
        if kept is None:  # pre-A6 log: the factor stayed iff the graph error with it was >= 50
            kept = (r.get("pgo_err_before") or 0.0) >= 50.0
        out[(int(r["cur_uid"]), int(r["loop_uid"]))] = {"kept": bool(kept), "pgo_ran": bool(r.get("pgo_ran"))}
    return out


class Replay:
    def __init__(self, dumps, attempts, cfg_tr):
        self.D = dumps
        self.att = attempts
        self.tran, self.rot = float(cfg_tr["pgo_tran_std"]), float(np.radians(cfg_tr["pgo_rot_std"]))
        self.sig = np.array([self.rot] * 3 + [self.tran] * 3)
        self.cov = gtsam.noiseModel.Diagonal.Sigmas(self.sig)
        self.fixed = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-9))
        self.max_iter = int(cfg_tr["pgo_max_iter"])
        self.thr_frame = float(cfg_tr["pgo_error_thre_frame"])
        self.cur_uids = [int(d["meta"]["cur_uid"]) for d in dumps]
        self.first = int(dumps[0]["all_cam_ids"][0])
        # the fixed prior holds the first frame at its initial pose; the real graph keeps x0 there to ~1e-12, so the
        # PGO value of frame 0 is the prior up to double rounding (poses_pre[0] of the first dump is the float32
        # camera matrix inverted in double: 1e-7 m off, which the 1e-9 sigma turns into ~3e3 error units)
        self.prior_pose = mat(dumps[0]["poses_pgo"][self.first])
        self.loops = {}      # (l, c) -> 4x4 measurement, in event order
        self.loop_order = []
        self.loop_checks = {}

    # ---------------------------------------------------------------- graph pieces
    def dump_of(self, f):
        """The dump whose poses_pre still hold the poses as they were when frame f was inserted into the graph:
        the earliest dump with cur_uid >= f (poses only change at PGO events)."""
        for d, c in zip(self.D, self.cur_uids):
            if f <= c:
                return d
        return self.D[-1]

    def odom_tracked(self, ids):
        """Odometry measurement of (a, b) = inv(T_a) T_b with BOTH poses as they were when b was added
        (add_odom_node_to_graph uses all_cameras[a].T at that moment, i.e. after any earlier PGO / write-back)."""
        out = {}
        for a, b in zip(ids[:-1], ids[1:]):
            d = self.dump_of(b)
            out[(a, b)] = rel(d["poses_pre"][a], d["poses_pre"][b])
        return out

    def build(self, ids, odom, loops):
        g = gtsam.NonlinearFactorGraph()
        g.add(gtsam.PriorFactorPose3(sym(ids[0]), gtsam.Pose3(self.prior_pose), self.fixed))
        for a, b in zip(ids[:-1], ids[1:]):
            g.add(gtsam.BetweenFactorPose3(sym(a), sym(b), gtsam.Pose3(odom[(a, b)]), self.cov))
        for (l, c), m in loops:
            g.add(gtsam.BetweenFactorPose3(sym(l), sym(c), gtsam.Pose3(m), self.cov))
        return g

    def values(self, ids, poses):
        """Initial values; the first frame is pinned to the exact prior pose (the dumps hold float32 camera matrices,
        whose ~1e-7 rounding the 1e-9 prior sigma would turn into 10^3-10^4 artificial error units)."""
        v = gtsam.Values()
        for u in ids:
            v.insert(sym(u), gtsam.Pose3(self.prior_pose if u == self.first else mat(poses[u])))
        return v

    def optimize(self, g, v):
        p = gtsam.LevenbergMarquardtParams()
        p.setMaxIterations(self.max_iter)
        return gtsam.LevenbergMarquardtOptimizer(g, v, p).optimizeSafely()

    # ---------------------------------------------------------------- loop recovery
    def recover_loop(self, g_rest, x_star, l, c, n_iter=3):
        Tl, Tc = x_star.atPose3(sym(l)), x_star.atPose3(sym(c))
        hx = Tl.between(Tc)
        grad = g_rest.linearize(x_star).gradientAtZero()
        gc = np.asarray(grad.at(sym(c))).reshape(-1)
        m = hx
        for _ in range(n_iter):
            f = gtsam.BetweenFactorPose3(sym(l), sym(c), m, self.cov)
            jf = f.linearize(x_star)
            keys = list(jf.keys())
            A = jf.getA()
            col = keys.index(sym(c)) * 6
            Ac = A[:, col:col + 6]
            w = np.linalg.solve(Ac.T, -gc)
            r = self.sig * w
            m = hx.compose(gtsam.Pose3.Expmap(-r))
        return m.matrix(), r

    def recover_all(self):
        """Event by event; returns {(l, c): (meas, check)} with check = gradient norms and LM drift at x*."""
        for d in self.D:
            c, l = int(d["meta"]["cur_uid"]), int(d["meta"]["loop_uid"])
            if not self.att.get((c, l), {}).get("kept", True):
                continue
            ids = [int(u) for u in d["all_cam_ids"]]
            odom = self.odom_tracked(ids)
            older = [(k, self.loops[k]) for k in self.loop_order]
            g_rest = self.build(ids, odom, older)
            x_star = self.values(ids, d["poses_pgo"])
            if d.get("loop_transform") is not None:
                m = mat(d["loop_transform"])
                src = "stored"
                _, r = None, None
            else:
                m, r = self.recover_loop(g_rest, x_star, l, c)
                src = "recovered"
            g_full = self.build(ids, odom, older + [((l, c), m)])
            gr0 = g_rest.linearize(x_star).gradientAtZero()
            gr1 = g_full.linearize(x_star).gradientAtZero()
            n0 = float(np.linalg.norm(np.concatenate([np.asarray(gr0.at(sym(u))).reshape(-1) for u in ids])))
            n1 = float(np.linalg.norm(np.concatenate([np.asarray(gr1.at(sym(u))).reshape(-1) for u in ids])))
            x1 = self.optimize(g_full, x_star)
            drift = max(float(np.linalg.norm(x1.atPose3(sym(u)).translation() - x_star.atPose3(sym(u)).translation())) for u in ids)
            e_star = float(g_full.error(x_star))
            chk = {"source": src, "grad_norm_rest": n0, "grad_norm_full": n1, "lm_drift_from_xstar_m": drift, "err_at_xstar": e_star,
                   "err_after_lm": float(g_full.error(x1)), "logged_pgo_err_after": d["meta"].get("pgo_err_after"),
                   "loop_residual_rot_rad": None if r is None else float(np.linalg.norm(r[:3])), "loop_residual_t_m": None if r is None else float(np.linalg.norm(r[3:]))}
            self.loops[(l, c)] = m
            self.loop_order.append((l, c))
            self.loop_checks[(l, c)] = chk
        return self.loops, self.loop_checks

    def loops_upto(self, k_event_index):
        """Recovered loops of the dumps 0..k (inclusive), in order."""
        keys = []
        for d in self.D[: k_event_index + 1]:
            key = (int(d["meta"]["loop_uid"]), int(d["meta"]["cur_uid"]))
            if key in self.loops:
                keys.append(key)
        return [(k, self.loops[k]) for k in keys]

    def loop_of(self, d):
        key = (int(d["meta"]["loop_uid"]), int(d["meta"]["cur_uid"]))
        return (key, self.loops[key]) if key in self.loops else None
