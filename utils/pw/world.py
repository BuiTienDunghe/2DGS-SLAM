"""PLAN A2G-P1 v2: plane world with a MEASURED anchor-map offset instead of an assumed drift model.

* Anchor parameters (n, o, B, P) never change after creation (no EKF, no re-centring, no freezing); only the support
  grid grows.
* b_hat_j (OffsetEstimator) is estimated from the residuals of recent matches and used for decisions only
  (IC, ambiguity, JC, merging) through y_j = x_j (+) b_hat_j. The pose constraint (plane_ctx, GN) uses x_j.
* Gate pose T_gate = constant-velocity prediction; Q_pred = full 6x6 covariance of its own error.
* Gamma (growth of V when an anchor is not seen) is learnt online from re-matches after gaps and revisit events.
* Merging only between anchors that both have fresh data (tau >= k - W_b); the earlier anchor is kept.
Never reads ground truth (I12).
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from . import geometry as G
from .association import ORIG, ambiguity, build_jc, ic_and_assign, init_from_jc, jc_stats, pair_terms
from .model import (CHI2_2, CHI2_3, OffsetEstimator, PredNoise, SensorNoise, ang_nis, chi2, jac_a, jac_b, nis,
                    residual)
from .segments import build_segments, merge_in_frame

ALLOWED = {("candidate", "active"), ("conflict", "candidate"), ("conflict", "deleted"), ("candidate", "deleted")}


def np_pose(T):
    """float64 pose with the rotation projected onto SO(3) (frontend poses drift off SO(3), see v1 report)."""
    T = T.detach().cpu().double().numpy() if isinstance(T, torch.Tensor) else np.array(T, dtype=np.float64)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    out = T.copy()
    out[:3, :3] = R
    return out


@dataclass
class AnchorP1:                     # world frame; n, o, B, P fixed at creation
    id: int
    primary_id: int
    n: np.ndarray
    o: np.ndarray
    B: np.ndarray
    P: np.ndarray
    grid_o: np.ndarray
    grid_B: np.ndarray
    cells: set
    state: str                      # candidate | active | conflict | merged
    conflict_with: Optional[int]
    T_ref: np.ndarray
    t_ref: int
    created_kf: int
    est: OffsetEstimator = None
    hash0: str = ""
    matched_kf: int = 0
    inview_unmatched_kf: int = 0
    conflict_quiet: int = 0
    conflict_diff_seen: int = 0


@dataclass
class FrameOut:
    init_pose: torch.Tensor
    init_source: str
    ctx: Optional[dict]
    plane_info: dict


def _j(x):
    if isinstance(x, np.ndarray):
        return [_j(v) for v in x.tolist()] if x.ndim else _j(float(x))
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _j(v) for k, v in x.items()}
    return x


def _hash(*arrs):
    h = hashlib.sha1()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


class PlaneWorld:
    def __init__(self, cfg, K, H, W, legacy, out_dir=None):
        self.cfg = cfg
        self.K = np.asarray(K.detach().cpu().double().numpy() if isinstance(K, torch.Tensor) else K, dtype=np.float64)
        self.H, self.W = int(H), int(W)
        self.legacy = legacy
        se = cfg.get("sensor", {}) or {}
        self.sensor = SensorNoise(se.get("a", 0.0), se.get("b", 1.425e-3), se.get("sigma_theta0_deg", 0.5),
                                  se.get("sigma_d0_m", 0.005))
        pi = cfg.get("pred_init", {}) or {}
        self.pred = PredNoise(cfg.get("pred_window", 30), pi.get("sigma_rho_m", 0.01), pi.get("sigma_theta_deg", 0.5))
        self.Wb = int(cfg.get("offset_window", 30))
        g0 = cfg.get("gamma0", {}) or {}
        th = np.deg2rad(float(g0.get("theta_deg", 0.15)))
        self.Gamma0 = np.array([th ** 2, th ** 2, float(g0.get("d_m", 0.005)) ** 2])
        self.gamma_sum = self.Gamma0.copy()          # prior counted as one sample
        self.gamma_n = 1
        self.gamma_min_gap = int(cfg.get("gamma_min_gap", 5))
        self.h = float(cfg.get("grid_h", 0.05))
        self.L = int(cfg.get("L_kf", 10))
        self.m_diff = int(cfg.get("m_diff", 2))
        self.inview_min = int(cfg.get("inview_min_cells", 3))
        self.obs_thr = float(cfg.get("obs_threshold", 0.3))
        self.gn_iters = int(cfg.get("gn_iters", 3))
        self.debug = bool(cfg.get("debug_checks", False))
        self.cs = int(getattr(legacy, "cell", 20)) if legacy is not None else 20
        self.sig_n = getattr(legacy, "sigma_n", 0.02) if legacy is not None else 0.02
        self.sig_d = getattr(legacy, "sigma_d", 0.01) if legacy is not None else 0.01
        self.anchors = {}
        self.next_id = 0
        self.diff = {}
        self.T_hist = []                             # last two optimised poses (projected)
        self.kt = 0                                  # tracked-frame counter (tau, Gamma, freshness use it)
        self.cur = None
        self.ghash_kf = None
        self.n_invariant = 0
        self.violation_msgs = []
        self.skip_merge_once = False                 # test hook (Q11 duplicate injection)
        self.counters = dict(merges=0, conflicts=0, revisits=0, cand_deleted=0, jc_jump=0, jc_shape_drop=0,
                             rank6=0, frames=0, plane_init=0, bias_accept=0, bias_reject=0, gamma_samples=0)
        self.ms_pw = []
        self.f_frames = self.f_kf = self.f_ev = None
        self.out_dir = out_dir
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            self.f_frames = open(os.path.join(out_dir, "pw_frames.jsonl"), "w")
            self.f_kf = open(os.path.join(out_dir, "pw_kf.jsonl"), "w")
            self.f_ev = open(os.path.join(out_dir, "pw_events.jsonl"), "w")

    # ------------------------------------------------------------------ helpers
    @property
    def Gamma(self):
        """Running mean of the (unclipped) per-sample estimates, prior Gamma0 counted as one sample, clipped at 0.
        PLAN v2 §3.4 clips every sample at 0 before averaging; that biases Gamma upwards (E[max(0, x)] > 0 for
        zero-mean noise) - see the report."""
        return np.diag(np.clip(self.gamma_sum / self.gamma_n, 0, None))

    def primaries(self):
        return [a for a in self.anchors.values() if a.state != "merged"]

    def geom_y(self, a):
        return a.est.y(a.n, a.o, a.B, G.oplus)

    def Vk_fn(self, k):
        Gm = self.Gamma
        return lambda a: a.est.cov_at(k, Gm)

    def _event(self, d):
        if self.f_ev is not None:
            self.f_ev.write(json.dumps(_j(d)) + "\n")
            self.f_ev.flush()

    def _violation(self, iid, msg):
        self.n_invariant += 1
        if len(self.violation_msgs) < 50:
            self.violation_msgs.append(f"{iid} k={self.cur['k'] if self.cur else None}: {msg}")
        self._event(dict(type="invariant", id=iid, msg=msg, k=self.cur["k"] if self.cur else None))
        if self.debug:
            raise AssertionError(f"PlaneWorld invariant {iid}: {msg}")

    def ghash(self):
        h = hashlib.sha1()
        for a in sorted(self.anchors.values(), key=lambda x: x.id):
            h.update(np.concatenate([a.n, a.o, a.P.ravel()]).tobytes())
            h.update(a.state.encode())
        return h.hexdigest()[:16]

    def _segs(self, planes):
        segs = build_segments(planes, self.sensor)
        if self.skip_merge_once:
            self.skip_merge_once = False
            return segs
        return merge_in_frame(segs, planes, self.sensor)

    def _new_anchor(self, seg, T, k, state, conflict_with=None):
        n, o, B = G.to_world(T, seg.n, seg.p, seg.B)
        n = n / np.linalg.norm(n)
        B = G.gram_schmidt(B, n)
        aid = self.next_id
        self.next_id += 1
        a = AnchorP1(id=aid, primary_id=aid, n=n, o=o, B=B, P=seg.R.copy(), grid_o=o.copy(), grid_B=B.copy(),
                     cells=set(), state=state, conflict_with=conflict_with, T_ref=T.copy(), t_ref=int(k),
                     created_kf=int(k))
        a.est = OffsetEstimator(a.P, self.kt, self.Wb)
        a.hash0 = _hash(a.n, a.o, a.B, a.P)
        a.cells = G.cells_from_segment(seg.cells_ij, self.cs, self.K, seg.n, seg.d, T, a.grid_o, a.grid_B, self.h)
        self.anchors[aid] = a
        return a

    def _sigma_min(self, segs, pose_pairs):
        if not pose_pairs:
            return 0.0
        sv = np.linalg.svd(np.stack([segs[s].n for s, _ in pose_pairs]), compute_uv=False)
        return float(sv[2]) if sv.shape[0] >= 3 else 0.0

    def _ctx(self, segs, pose_pairs, device, dtype):
        """plane_ctx in the E1c format, from the ORIGINAL anchors (I15)."""
        if not pose_pairs:
            return None, None
        n_w = np.stack([self.anchors[a].n for _, a in pose_pairs])
        d_w = np.array([-float(self.anchors[a].n @ self.anchors[a].o) for _, a in pose_pairs])
        w = torch.tensor([float(segs[s].meas.n_px) for s, _ in pose_pairs], dtype=torch.float64)
        ctx = dict(
            m=torch.tensor(np.stack([segs[s].n for s, _ in pose_pairs]), dtype=dtype, device=device),
            n_w=torch.tensor(n_w, dtype=dtype, device=device),
            d_meas=torch.tensor([segs[s].d for s, _ in pose_pairs], dtype=dtype, device=device),
            d_w=torch.tensor(d_w, dtype=dtype, device=device),
            w=(w / w.sum()).to(device=device, dtype=dtype),
            inv_sn2=1.0 / self.sig_n ** 2,
            inv_sd2=1.0 / self.sig_d ** 2,
        )
        return ctx, _hash(n_w, d_w)

    def T_gate(self, T_pred):
        if len(self.T_hist) >= 2:
            T1, T2 = self.T_hist[-1], self.T_hist[-2]
            return T1 @ G.inv_T(T2) @ T1
        if self.T_hist:
            return self.T_hist[-1].copy()
        return T_pred

    # ------------------------------------------------------------------ frame 0
    def on_first_frame(self, k, planes, T0):
        t0 = time.perf_counter()
        T = np_pose(T0)
        segs = self._segs(planes)
        for sg in segs:
            self._new_anchor(sg, T, k, "active")
        self.T_hist = [T]
        self.ghash_kf = self.ghash()
        self._write_kf(dict(k=int(k), created=[dict(id=a.id, state="active", reason="frame0")
                                               for a in self.anchors.values()], transitions=[], Gamma=np.diag(self.Gamma),
                            n_primary=len(self.primaries())))
        return dict(planes=[s.meas for s in segs], pairs=[], init_source="prev", sigma_min=float("nan"),
                    ms_extract=0.0, ms_match=1000 * (time.perf_counter() - t0), ms_solve=0.0, T_init=T0, active_ids=[])

    # ------------------------------------------------------------------ every frame
    def on_frame(self, k, planes, T_pred):
        t0 = time.perf_counter()
        self.kt += 1
        if self.ghash_kf is not None and self.ghash() != self.ghash_kf:
            self._violation("I6", "anchor geometry changed outside a keyframe")
        Tp_np = np_pose(T_pred)
        Tg = self.T_gate(Tp_np)
        segs = self._segs(planes)
        Q = self.pred.Q()
        prim = self.primaries()
        Vk = self.Vk_fn(self.kt)
        cand, assign, info = ic_and_assign(segs, prim, Tg, self.geom_y, Vk, Q)
        t1 = time.perf_counter()
        ambig = ambiguity(segs, assign, prim, Tg, self.geom_y, Vk, self.h, Q)
        pose = [(s, j) for s, j in assign if self.anchors[j].state == "active"]
        res = init_from_jc(pose, segs, self.anchors, Tg, Q, self.geom_y, Vk, self.obs_thr, self.gn_iters)
        if res["removed"]:
            assign = [p for p in assign if p not in res["removed"]]
            self.counters["jc_shape_drop"] += len(res["removed"])
        source, T_init, pose_pairs = res["source"], res["T_init"], res["pose_pairs"]
        added, dT = [], None
        if source == "plane":
            self.counters["plane_init"] += 1
            dT = float(np.linalg.norm(G.se3_log(T_init @ G.inv_T(Tg))))
            used_s, used_a = {s for s, _ in assign}, {j for _, j in assign}
            free_s = [i for i in range(len(segs)) if i not in used_s]
            free_a = {a.id for a in prim if a.id not in used_a}
            if free_s and free_a:
                c2, a2, info2 = ic_and_assign(segs, prim, T_init, self.geom_y, Vk, res["Ip"], restrict_s=free_s,
                                              restrict_a=free_a)
                if a2:
                    new_pose = [(s, j) for s, j in a2 if self.anchors[j].state == "active"]
                    ok = True
                    if new_pose:
                        st = jc_stats(*build_jc(pose_pairs + new_pose, segs, self.anchors, Tg, Q, self.geom_y, Vk))
                        ok = st["dof_shape"] <= 0 or st["shape"] <= chi2(st["dof_shape"])
                    if ok:
                        added = a2
                        assign = sorted(assign + a2)
                        pose_pairs = pose_pairs + new_pose
                        cand += [dict(c, at="T_p") for c in c2]
                        info.update(info2)
        elif source == "jc_jump":
            self.counters["jc_jump"] += 1
            self._event(dict(type="jc_jump", k=int(k), jc=res["jc"]))
        if res["jc"] is not None and res["jc"].get("rank") == 6:
            self.counters["rank6"] += 1
        self.counters["frames"] += 1
        ctx, ctx_hash = self._ctx(segs, pose_pairs, T_pred.device, T_pred.dtype)
        if pose_pairs:                                       # I15: ctx built from the original anchors
            ref = _hash(np.stack([self.anchors[a].n for _, a in pose_pairs]),
                        np.array([-float(self.anchors[a].n @ self.anchors[a].o) for _, a in pose_pairs]))
            if ref != ctx_hash:
                self._violation("I15", "plane_ctx not built from the original anchors")
        T_init_t = (torch.from_numpy(T_init).to(device=T_pred.device, dtype=T_pred.dtype) if source == "plane"
                    else T_pred)
        bhat = [float(np.linalg.norm(self.anchors[j].est.b)) for _, j in pose_pairs]
        self.cur = dict(k=int(k), kt=self.kt, segs=segs, T_gate=Tg, assign=assign, pose_pairs=pose_pairs, ambig=ambig, cand=cand,
                        info=info, jc=res["jc"], source=source, dxi=res["dxi"], Q=Q, added=added, dT=dT,
                        bhat_mean=float(np.mean(bhat)) if bhat else None, gn_start=res.get("gn_start"))
        self._check_frame()
        t2 = time.perf_counter()
        ms = 1000 * (t2 - t0)
        self.ms_pw.append(ms)
        plane_info = dict(planes=[s.meas for s in segs], pairs=assign, init_source=source,
                          sigma_min=self._sigma_min(segs, pose_pairs), ms_extract=0.0, ms_match=1000 * (t1 - t0),
                          ms_solve=1000 * (t2 - t1), T_init=T_init_t, active_ids=[j for _, j in pose_pairs], ms_pw=ms)
        return FrameOut(init_pose=T_init_t, init_source=source, ctx=ctx, plane_info=plane_info)

    def _check_frame(self):
        c = self.cur
        ss, aa = [s for s, _ in c["assign"]], [j for _, j in c["assign"]]
        if len(set(ss)) != len(ss) or len(set(aa)) != len(aa):
            self._violation("I1", "one-to-one assignment broken")
        for s, j in c["assign"]:
            v = c["info"].get((s, j), (None,))[0]
            if v is not None and v > CHI2_3 + 1e-9:
                self._violation("I2", f"assigned pair ({s},{j}) NIS {v:.2f}")
            a = self.anchors[j]
            if a.state == "conflict" and c["info"].get((s, a.conflict_with), (np.inf,))[0] <= CHI2_3:
                self._violation("I4", f"conflict {j} got segment {s} that {a.conflict_with} passes")
        for _, j in c["pose_pairs"]:
            if self.anchors[j].state != "active":
                self._violation("I3", f"pose pair with {self.anchors[j].state} anchor {j}")
        if c["source"] == "jc_jump" and c["pose_pairs"]:
            self._violation("I3", "jump failed but pose pairs kept")
        if c["source"] == "plane":
            jc = c["jc"]
            if not (jc["rank"] == 6 and jc["jump"] <= chi2(6)):
                self._violation("I10", f"plane init with rank {jc['rank']} jump {jc['jump']:.2f}")

    # ------------------------------------------------------------------ after tracking: Q_pred, b_hat samples
    def _bias_sample(self, seg, a, T, k):
        r, Ja, Jb, _ = pair_terms(seg, ORIG(a), T)
        Jbi = np.linalg.inv(Jb)
        z = -Jbi @ r
        Rz = Jbi @ (Ja @ seg.R @ Ja.T) @ Jbi.T
        Rz = 0.5 * (Rz + Rz.T)
        fresh = (k - a.est.tau) < self.Wb
        ok, m, gamma_obs = a.est.sample(z, Rz, k, self.Gamma, self.gamma_min_gap)
        if ok:
            self.counters["bias_accept"] += 1
            if gamma_obs is not None:
                self.gamma_sum += gamma_obs
                self.gamma_n += 1
                self.counters["gamma_samples"] += 1
        else:
            self.counters["bias_reject"] += 1
        if np.min(np.linalg.eigvalsh(a.est.V)) <= 0:
            self._violation("I8", f"V of {a.id} not positive definite")
        return dict(z=z, accepted=bool(ok), m=m, fresh=bool(fresh))

    def after_tracking(self, k, T_opt):
        c = self.cur
        T = np_pose(T_opt)
        self.pred.push(G.se3_log(T @ G.inv_T(c["T_gate"])))
        pairs_log = []
        for s, j in c["assign"]:
            a = self.anchors[j]
            fresh = (self.kt - a.est.tau) < self.Wb
            bs = self._bias_sample(c["segs"][s], a, T, self.kt)
            pairs_log.append(dict(s=s, j=j, nis=c["info"].get((s, j), (None,))[0], fresh=bool(fresh), bias_sample=bs))
        self.T_hist = (self.T_hist + [T])[-2:]
        ids = sorted(self.anchors[j].primary_id for _, j in c["assign"])
        for i in range(len(ids)):
            for l in range(i + 1, len(ids)):
                self.diff[(ids[i], ids[l])] = self.diff.get((ids[i], ids[l]), 0) + 1
        if self.f_frames is not None:
            row = dict(k=c["k"], kt=self.kt, segs=[dict(src=s.src, N_s=s.N_s, n=s.n, p=s.p) for s in c["segs"]],
                       cand=[dict(s=x["s"], j=x["j"], state=x["state"], nis=x["nis"], ic=x["ic"],
                                  blocked=x.get("blocked", False), at=x.get("at", "T_gate")) for x in c["cand"]],
                       assign=c["assign"], pose=c["pose_pairs"], ambig=c["ambig"], added=c["added"], pairs=pairs_log,
                       jc=c["jc"], init=dict(source=c["source"], dxi=c["dxi"], dT_gate=c["dT"], gn_start=c["gn_start"],
                                             bhat_mean=c["bhat_mean"]),
                       Qpred=c["Q"], T_gate=c["T_gate"][:3, :], T_opt=T[:3, :], ghash=self.ghash())
            self.f_frames.write(json.dumps(_j(row)) + "\n")
            self.f_frames.flush()

    # ------------------------------------------------------------------ keyframe
    def _diff(self, a, b):
        a, b = self.anchors[a].primary_id, self.anchors[b].primary_id
        return self.diff.get((min(a, b), max(a, b)), 0)

    def on_keyframe(self, k, T_opt, depth):
        c = self.cur
        T = np_pose(T_opt)
        Z = depth.detach().cpu().float().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
        segs = c["segs"]
        prim = self.primaries()
        kt = self.kt
        Vk = self.Vk_fn(kt)
        cand, assign, info = ic_and_assign(segs, prim, T, self.geom_y, Vk, None)
        ambig = ambiguity(segs, assign, prim, T, self.geom_y, Vk, self.h, None)
        kf = dict(k=int(k), kt=kt, reassign=[dict(s=s, j=j, nis=info[(s, j)][0],
                                           fresh=bool((kt - self.anchors[j].est.tau) < self.Wb)) for s, j in assign],
                  ambig=ambig, created=[], transitions=[], merges=[])
        for s, j in assign:                                    # support region only (geometry fixed)
            a = self.anchors[j]
            sg = segs[s]
            a.cells |= G.cells_from_segment(sg.cells_ij, self.cs, self.K, sg.n, sg.d, T, a.grid_o, a.grid_B, self.h)
        used = {s for s, _ in assign}
        assigned_anchors = {j for _, j in assign}
        amb = {}
        for s, j in ambig:
            amb.setdefault(s, []).append(j)
        new_ids = set()
        for s, sg in enumerate(segs):
            if s in used:
                continue
            js = amb.get(s, [])
            free = [j for j in js if j not in assigned_anchors]
            if free:
                a = self._new_anchor(sg, T, k, "conflict", conflict_with=free[0])
                reason = f"ambiguous_with_{free[0]}"
                self.counters["conflicts"] += 1
            else:
                a = self._new_anchor(sg, T, k, "candidate")
                reason = "new"
                for j in js:
                    pj = self.anchors[j].primary_id
                    key = (min(a.id, pj), max(a.id, pj))
                    self.diff[key] = max(self.diff.get(key, 0), 1)
                if js:
                    reason = f"ambiguous_with_assigned_{js[0]}"
            a.conflict_diff_seen = self._diff(a.id, a.conflict_with) if a.conflict_with is not None else 0
            new_ids.add(a.id)
            kf["created"].append(dict(id=a.id, state=a.state, reason=reason))
        Gm = self.Gamma
        for a in self.primaries():
            if a.id in new_ids:
                continue
            if a.id in assigned_anchors:
                a.matched_kf += 1
                a.inview_unmatched_kf = 0
                continue
            Vd = a.est.cov_at(kt, Gm)[2, 2]
            nvis = G.n_visible_cells(a.cells, a.grid_o, a.grid_B, self.h, T, self.K, Z, self.sensor.sigma_z,
                                     float(np.sqrt(Vd)), self.H, self.W)
            inview = nvis >= self.inview_min
            if inview:
                a.inview_unmatched_kf += 1
            if a.state == "conflict" and inview:
                d = self._diff(a.id, a.conflict_with)
                if d <= a.conflict_diff_seen:
                    a.conflict_quiet += 1
                a.conflict_diff_seen = d
        for a in list(self.primaries()):
            if a.id in new_ids:
                continue
            frm, to = a.state, None
            if a.state == "conflict":
                js = self.anchors.get(a.conflict_with)
                if js is None:
                    to = "candidate"
                else:
                    if js.state == "merged":
                        a.conflict_with = js.primary_id
                        js = self.anchors[a.conflict_with]
                    if self._diff(a.id, a.conflict_with) >= self.m_diff:
                        to = "candidate"
                    elif a.conflict_quiet >= self.L:
                        to = "deleted"
                        self._revisit(a, js, T, k)
            elif a.state == "candidate":
                if a.matched_kf >= 1:
                    to = "active"
                elif a.inview_unmatched_kf >= self.L:
                    to = "deleted"
                    self.counters["cand_deleted"] += 1
            if to is not None:
                if (frm, to) not in ALLOWED:
                    self._violation("I11", f"transition {frm}->{to}")
                kf["transitions"].append([a.id, frm, to])
                if to == "deleted":
                    self._event(dict(type="delete", k=int(k), id=a.id, state=frm))
                    del self.anchors[a.id]
                    for x in self.anchors.values():
                        if x.conflict_with == a.id:
                            x.conflict_with = None
                else:
                    a.state = to
        while True:
            best = self._best_merge(kt)
            if best is None:
                break
            kf["merges"].append(self._merge(*best, k, kt))
        for a in self.anchors.values():
            if _hash(a.n, a.o, a.B, a.P) != a.hash0:
                self._violation("I5", f"anchor {a.id} parameters changed")
            if a.state == "merged" and (a.primary_id not in self.anchors or self.anchors[a.primary_id].state == "merged"):
                self._violation("I9", f"secondary {a.id} -> {a.primary_id}")
        if np.any(np.diag(self.Gamma) < 0):
            self._violation("I8", "Gamma negative")
        kf["n_primary"] = len(self.primaries())
        kf["states"] = {st: sum(1 for a in self.primaries() if a.state == st) for st in ("candidate", "active", "conflict")}
        kf["Gamma"] = np.diag(self.Gamma)
        kf["bias_reject"] = self.counters["bias_reject"]
        kf["bias_accept"] = self.counters["bias_accept"]
        self._write_kf(kf)
        self.ghash_kf = self.ghash()

    def _revisit(self, c, js, T, k):
        """Conflict c deleted after L quiet keyframes: revisit event and a Gamma sample (PLAN v2 §3.4 ii)."""
        yn, yo, yB = self.geom_y(js)
        an, ao, aB = G.to_camera(T, c.n, c.o, c.B)
        bn, bo, bB = G.to_camera(T, yn, yo, yB)
        r = residual(an, ao, bn, bo, bB)
        Jbi = np.linalg.inv(jac_b(an, ao, bn, bo, bB))
        z = -Jbi @ r
        gap = max(1, self.kt - js.est.tau)
        gam = np.diag(np.outer(z, z) - js.est.V - c.P) / gap
        self.gamma_sum += gam
        self.gamma_n += 1
        self.counters["revisits"] += 1
        self.counters["gamma_samples"] += 1
        self._event(dict(type="revisit", k=int(k), c=c.id, jstar=js.id, r=r, z=z, gap=gap, gamma=gam))

    def _aa(self, a, b):
        """r(y_a | y_b) in the world with S = J_a V_a J_a^T + J_b V_b J_b^T (no pose term, no Gamma)."""
        an, ao, aB = self.geom_y(a)
        bn, bo, bB = self.geom_y(b)
        r = residual(an, ao, bn, bo, bB)
        Ja, Jb = jac_a(an, aB, bn, bB), jac_b(an, ao, bn, bo, bB)
        S = Ja @ a.est.V @ Ja.T + Jb @ b.est.V @ Jb.T
        return r, 0.5 * (S + S.T)

    def _regions_touch(self, a, b):
        if not a.cells or not b.cells:
            return False
        X = G.grid_centres(b.cells, b.grid_o, b.grid_B, self.h)
        return G.overlap_or_adjacent(G.grid_index(X, a.grid_o, a.grid_B, self.h), a.cells)

    def _best_merge(self, k):
        prim = [a for a in self.primaries() if a.state != "conflict" and a.est.tau >= k - self.Wb]
        best = None
        for i in range(len(prim)):
            for l in range(i + 1, len(prim)):
                aj, ak = prim[i], prim[l]
                if self._diff(aj.id, ak.id) >= self.m_diff:
                    continue
                r1, S1 = self._aa(ak, aj)
                if ang_nis(r1, S1) > CHI2_2 or not self._regions_touch(aj, ak):
                    continue
                r2, S2 = self._aa(aj, ak)
                v1, v2 = nis(r1, S1), nis(r2, S2)
                if v1 <= CHI2_3 and v2 <= CHI2_3 and (best is None or max(v1, v2) < best[2]):
                    best = (aj.id, ak.id, max(v1, v2), (r1, S1, v1, v2))
        return best

    def _merge(self, jid, kid, key, extra, k, kt):
        aj, ak = self.anchors[jid], self.anchors[kid]
        if self._diff(jid, kid) >= self.m_diff or "conflict" in (aj.state, ak.state):
            self._violation("I7", f"merge of {jid},{kid} in E or conflict")
        if min(aj.est.tau, ak.est.tau) < kt - self.Wb:
            self._violation("I14", f"merge of {jid},{kid} without fresh data")
        if (aj.t_ref, -aj.matched_kf) <= (ak.t_ref, -ak.matched_kf):
            keep, drop = aj, ak
        else:
            keep, drop = ak, aj
        yk, yd = self.geom_y(keep), self.geom_y(drop)
        ev = dict(type="merge", k=int(k), kt=int(kt), keep=keep.id, drop=drop.id, nis=[extra[2], extra[3]], r=extra[0], S=extra[1],
                  tau=[keep.est.tau, drop.est.tau], bhat=[keep.est.b, drop.est.b], V=[keep.est.V, drop.est.V],
                  keep_par=dict(n=keep.n, o=keep.o, state=keep.state, n_cells=len(keep.cells), t_ref=keep.t_ref),
                  drop_par=dict(n=drop.n, o=drop.o, state=drop.state, n_cells=len(drop.cells), t_ref=drop.t_ref),
                  dist_at_o_drop=float(yk[0] @ (yd[1] - yk[1])),
                  angle_deg=float(np.degrees(np.arccos(np.clip(abs(yk[0] @ yd[0]), -1, 1)))))
        if keep.state == "candidate" and drop.state == "active":
            keep.state = "active"
        if drop.cells:
            X = G.grid_centres(drop.cells, drop.grid_o, drop.grid_B, self.h)
            keep.cells |= {tuple(x) for x in G.grid_index(X, keep.grid_o, keep.grid_B, self.h).tolist()}
        for (a, b), v in list(self.diff.items()):
            if drop.id in (a, b):
                o = b if a == drop.id else a
                if o != keep.id:
                    key2 = (min(keep.id, o), max(keep.id, o))
                    self.diff[key2] = self.diff.get(key2, 0) + v
        keep.matched_kf += drop.matched_kf
        drop.state = "merged"
        drop.primary_id = keep.id
        drop.est.buf.clear()
        for x in self.anchors.values():
            if x.primary_id == drop.id:
                x.primary_id = keep.id
            if x.conflict_with == drop.id:
                x.conflict_with = keep.id
        self.counters["merges"] += 1
        self._event(ev)
        return dict(keep=keep.id, drop=drop.id, nis=[extra[2], extra[3]])

    # ------------------------------------------------------------------ output
    def _write_kf(self, d):
        if self.f_kf is not None:
            self.f_kf.write(json.dumps(_j(d)) + "\n")
            self.f_kf.flush()

    def _row(self, a, cells=False):
        d = dict(id=a.id, n_w=a.n.tolist(), d_w=-float(a.n @ a.o), n_obs_kf=a.matched_kf + 1,
                 active=a.state == "active", created_kf=a.created_kf, n_samples=len(a.cells), primary_id=a.primary_id,
                 state=a.state, o=a.o.tolist(), P=a.P.tolist(), t_ref=a.t_ref, n_cells=len(a.cells),
                 conflict_with=a.conflict_with, bhat=a.est.b.tolist(), V=a.est.V.tolist(), tau=a.est.tau,
                 f=a.est.f, n_accept=a.est.n_accept, n_reject=a.est.n_reject)
        if cells:
            d["cells"] = sorted(list(c) for c in a.cells)
            d["grid_o"], d["grid_B"] = a.grid_o.tolist(), a.grid_B.tolist()
        return d

    def anchors_table(self):
        return [self._row(a) for a in sorted(self.anchors.values(), key=lambda x: x.id)]

    def n_primary(self):
        return len(self.primaries())

    def n_active(self):
        return sum(1 for a in self.primaries() if a.state == "active")

    def summary(self):
        d = dict(self.counters)
        d.update(n_invariant=self.n_invariant, n_primary_final=self.n_primary(), n_active_final=self.n_active(),
                 Gamma_final=np.diag(self.Gamma).tolist(),
                 ms_pw_mean=float(np.mean(self.ms_pw)) if self.ms_pw else None,
                 ms_pw_p95=float(np.percentile(self.ms_pw, 95)) if self.ms_pw else None)
        return d

    def close(self):
        if self.out_dir is not None:
            with open(os.path.join(self.out_dir, "anchors_final.json"), "w") as f:
                json.dump(_j(dict(anchors=[self._row(a, cells=True) for a in self.anchors.values()],
                                  Gamma=np.diag(self.Gamma), counters=self.counters)), f)
        for f in (self.f_frames, self.f_kf, self.f_ev):
            if f is not None:
                f.close()
