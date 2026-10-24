"""PLAN A2G-P1 core: plane world for tracking built on one uncertainty model (utils/pw/uncertainty.py).

* Anchor parameters (n, o, B, P) fixed at creation; only the support grid grows at keyframes.
* b_hat_j measured from unambiguous matches (stationary, no growth model); decisions use y_j = x_j (+) b_hat_j, the
  pose constraint (plane_ctx, GN) uses the original x_j.
* Gate pose = constant-velocity prediction, Q_pred = 6x6 covariance of its own error.
* Life cycle: candidate / active / merged. An unassigned segment that is ambiguous with an active anchor j is a new
  candidate when j took another segment this frame, otherwise it is deferred (event; revisit_candidate for the loop
  module when j has no fresh data).
* Every decision is nis <= chi2 with S from residual_cov. Never reads ground truth (I12).
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from . import geometry as G
from .association import ORIG, ambiguity, build_jc, ic_and_assign, init_from_jc, jc_stats, pair_S, pair_terms
from .segments import build_segments, merge_in_frame
from .uncertainty import (CHI2_2, CHI2_3, OffsetEstimator, PredNoise, SensorNoise, ang_nis, chi2, jac_a, jac_b, nis,
                          residual, residual_cov)

ALLOWED = {("candidate", "active"), ("candidate", "deleted")}


def np_pose(T):
    """float64 pose with the rotation projected onto SO(3) (frontend poses drift off SO(3))."""
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
class AnchorP1:
    id: int
    primary_id: int
    n: np.ndarray
    o: np.ndarray
    B: np.ndarray
    P: np.ndarray
    grid_o: np.ndarray
    grid_B: np.ndarray
    cells: set
    state: str                      # candidate | active | merged
    T_ref: np.ndarray
    t_ref: int
    created_kf: int
    est: OffsetEstimator = None
    hash0: str = ""
    matched_kf: int = 0
    inview_unmatched_kf: int = 0


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
        se = cfg.get("sensor", {}) or {}
        self.sensor = SensorNoise(se.get("a", 0.0), se.get("b", 1.425e-3), se.get("sigma_theta0_deg", 0.5),
                                  se.get("sigma_d0_m", 0.005))
        pi = cfg.get("pred_init", {}) or {}
        self.pred = PredNoise(cfg.get("pred_window", 30), pi.get("sigma_rho_m", 0.01), pi.get("sigma_theta_deg", 0.5))
        self.Wb = int(cfg.get("offset_window", 30))
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
        self.T_hist = []
        self.kt = 0                                  # tracked-frame counter (tau, freshness)
        self.cur = None
        self.ghash_kf = None
        self.n_invariant = 0
        self.violation_msgs = []
        self.skip_merge_once = False                 # test hook (duplicate injection)
        self.counters = dict(merges=0, deferred=0, revisit_candidates=0, cand_deleted=0, jc_jump=0, jc_shape_drop=0,
                             rank6=0, frames=0, plane_init=0, bias_accept=0, bias_reject=0, bias_ambiguous=0)
        self.ms_pw = []
        self.f_frames = self.f_kf = self.f_ev = None
        self.out_dir = out_dir
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            self.f_frames = open(os.path.join(out_dir, "pw_frames.jsonl"), "w")
            self.f_kf = open(os.path.join(out_dir, "pw_kf.jsonl"), "w")
            self.f_ev = open(os.path.join(out_dir, "pw_events.jsonl"), "w")

    # ------------------------------------------------------------------ helpers
    def primaries(self):
        return [a for a in self.anchors.values() if a.state != "merged"]

    def geom_y(self, a):
        return a.est.y(a.n, a.o, a.B, G.oplus)

    @staticmethod
    def V_of(a):
        return a.est.V

    def fresh(self, a):
        return (self.kt - a.est.tau) < self.Wb

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

    def _new_anchor(self, seg, T, k, state):
        n, o, B = G.to_world(T, seg.n, seg.p, seg.B)
        n = n / np.linalg.norm(n)
        B = G.gram_schmidt(B, n)
        aid = self.next_id
        self.next_id += 1
        a = AnchorP1(id=aid, primary_id=aid, n=n, o=o, B=B, P=seg.R.copy(), grid_o=o.copy(), grid_B=B.copy(),
                     cells=set(), state=state, T_ref=T.copy(), t_ref=int(k), created_kf=int(k))
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
        """plane_ctx in the E1c format from the ORIGINAL anchors (I8)."""
        if not pose_pairs:
            return None, None
        n_w = np.stack([self.anchors[a].n for _, a in pose_pairs])
        d_w = np.array([-float(self.anchors[a].n @ self.anchors[a].o) for _, a in pose_pairs])
        w = torch.tensor([float(segs[s].meas.n_px) for s, _ in pose_pairs], dtype=torch.float64)
        ctx = dict(m=torch.tensor(np.stack([segs[s].n for s, _ in pose_pairs]), dtype=dtype, device=device),
                   n_w=torch.tensor(n_w, dtype=dtype, device=device),
                   d_meas=torch.tensor([segs[s].d for s, _ in pose_pairs], dtype=dtype, device=device),
                   d_w=torch.tensor(d_w, dtype=dtype, device=device),
                   w=(w / w.sum()).to(device=device, dtype=dtype),
                   inv_sn2=1.0 / self.sig_n ** 2, inv_sd2=1.0 / self.sig_d ** 2)
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
        self._write_kf(dict(k=int(k), kt=0, created=[dict(id=a.id, state="active", reason="frame0")
                                                     for a in self.anchors.values()], transitions=[], deferred=[],
                            n_primary=len(self.primaries())))
        return dict(planes=[s.meas for s in segs], pairs=[], init_source="prev", sigma_min=float("nan"),
                    ms_extract=0.0, ms_match=1000 * (time.perf_counter() - t0), ms_solve=0.0, T_init=T0, active_ids=[])

    # ------------------------------------------------------------------ every frame
    def on_frame(self, k, planes, T_pred):
        t0 = time.perf_counter()
        self.kt += 1
        if self.ghash_kf is not None and self.ghash() != self.ghash_kf:
            self._violation("I4", "anchor geometry changed outside a keyframe")
        Tg = self.T_gate(np_pose(T_pred))
        segs = self._segs(planes)
        Q = self.pred.Q()
        prim = self.primaries()
        cand, assign, info = ic_and_assign(segs, prim, Tg, self.geom_y, self.V_of, Q, h=self.h)
        t1 = time.perf_counter()
        pose = [(s, j) for s, j in assign if self.anchors[j].state == "active"]
        res = init_from_jc(pose, segs, self.anchors, Tg, Q, self.geom_y, self.V_of, self.obs_thr, self.gn_iters)
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
                c2, a2, info2 = ic_and_assign(segs, prim, T_init, self.geom_y, self.V_of, res["Ip"], restrict_s=free_s,
                                              restrict_a=free_a, h=self.h)
                if a2:
                    new_pose = [(s, j) for s, j in a2 if self.anchors[j].state == "active"]
                    ok = True
                    if new_pose:
                        st = jc_stats(*build_jc(pose_pairs + new_pose, segs, self.anchors, Tg, Q, self.geom_y, self.V_of))
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
        if pose_pairs:
            ref = _hash(np.stack([self.anchors[a].n for _, a in pose_pairs]),
                        np.array([-float(self.anchors[a].n @ self.anchors[a].o) for _, a in pose_pairs]))
            if ref != ctx_hash:
                self._violation("I8", "plane_ctx not built from the original anchors")
        T_init_t = (torch.from_numpy(T_init).to(device=T_pred.device, dtype=T_pred.dtype) if source == "plane"
                    else T_pred)
        flags = {}
        for c in cand:                                   # flags of the candidate that produced each assignment
            key = (c["s"], c["j"])
            if c.get("at", "T_gate") == ("T_p" if key in added else "T_gate"):
                flags[key] = (bool(c.get("unambiguous", False)), int(c.get("region_level", -1)))
        bhat = [float(np.linalg.norm(self.anchors[j].est.b)) for _, j in pose_pairs]
        self.cur = dict(k=int(k), kt=self.kt, segs=segs, T_gate=Tg, assign=assign, pose_pairs=pose_pairs, cand=cand,
                        info=info, jc=res["jc"], source=source, dxi=res["dxi"], Q=Q, added=added, dT=dT, flags=flags,
                        bhat_mean=float(np.mean(bhat)) if bhat else None, gn_start=res.get("gn_start"))
        self._check_frame()
        ms = 1000 * (time.perf_counter() - t0)
        self.ms_pw.append(ms)
        plane_info = dict(planes=[s.meas for s in segs], pairs=assign, init_source=source,
                          sigma_min=self._sigma_min(segs, pose_pairs), ms_extract=0.0, ms_match=1000 * (t1 - t0),
                          ms_solve=ms - 1000 * (t1 - t0), T_init=T_init_t, active_ids=[j for _, j in pose_pairs], ms_pw=ms)
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
        for _, j in c["pose_pairs"]:
            if self.anchors[j].state != "active":
                self._violation("I3", f"pose pair with {self.anchors[j].state} anchor {j}")
        if c["source"] == "jc_jump" and c["pose_pairs"]:
            self._violation("I3", "jump failed but pose pairs kept")
        if c["source"] == "plane":
            jc = c["jc"]
            if not (jc["rank"] == 6 and jc["jump"] <= chi2(6)):
                self._violation("I10", f"plane init with rank {jc['rank']} jump {jc['jump']:.2f}")

    # ------------------------------------------------------------------ after tracking
    def _bias_sample(self, seg, a, T, unambiguous):
        if not unambiguous:                              # PLAN core §3.2: only unambiguous pairs are samples (I6)
            self.counters["bias_ambiguous"] += 1
            return dict(z=None, accepted=False, m=None, unambiguous=False)
        r, Ja, Jb, _ = pair_terms(seg, ORIG(a), T)
        Jbi = np.linalg.inv(Jb)
        z = -Jbi @ r
        Rz = Jbi @ (Ja @ seg.R @ Ja.T) @ Jbi.T
        Rz = 0.5 * (Rz + Rz.T)
        ok, m = a.est.sample(z, Rz, self.kt)
        self.counters["bias_accept" if ok else "bias_reject"] += 1
        if ok and m > CHI2_3 + 1e-9:
            self._violation("I6", f"sample accepted with m {m:.2f}")
        if np.min(np.linalg.eigvalsh(a.est.V)) <= 0:
            self._violation("I5", f"V of {a.id} not positive definite")
        return dict(z=z, accepted=bool(ok), m=m, unambiguous=True)

    def after_tracking(self, k, T_opt):
        c = self.cur
        T = np_pose(T_opt)
        self.pred.push(G.se3_log(T @ G.inv_T(c["T_gate"])))
        pairs_log = []
        for s, j in c["assign"]:
            a = self.anchors[j]
            fresh = self.fresh(a)
            unamb, level = c["flags"].get((s, j), (False, -1))
            bs = self._bias_sample(c["segs"][s], a, T, unamb)
            pairs_log.append(dict(s=s, j=j, nis=c["info"].get((s, j), (None,))[0], fresh=bool(fresh),
                                  unambiguous=unamb, region_level=level, bias_sample=bs))
        self.T_hist = (self.T_hist + [T])[-2:]
        ids = sorted(self.anchors[j].primary_id for _, j in c["assign"])
        for i in range(len(ids)):
            for l in range(i + 1, len(ids)):
                self.diff[(ids[i], ids[l])] = self.diff.get((ids[i], ids[l]), 0) + 1
        if self.f_frames is not None:
            row = dict(k=c["k"], kt=self.kt, segs=[dict(src=s.src, N_s=s.N_s, n=s.n, p=s.p) for s in c["segs"]],
                       cand=[dict(s=x["s"], j=x["j"], state=x["state"], nis=x["nis"], ic=x["ic"],
                                  region_level=x.get("region_level"), unambiguous=x.get("unambiguous"),
                                  at=x.get("at", "T_gate")) for x in c["cand"]],
                       assign=c["assign"], pose=c["pose_pairs"], added=c["added"], pairs=pairs_log,
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
        cand, assign, info = ic_and_assign(segs, prim, T, self.geom_y, self.V_of, None, h=self.h)
        ambig = ambiguity(segs, assign, prim, T, self.geom_y, self.V_of, self.h, None)

        def _parts(s, j):
            _, S_, r_ = info[(s, j)]
            return float(r_[:2] @ np.linalg.solve(S_[:2, :2], r_[:2])), float(r_[2] ** 2 / S_[2, 2])

        unamb = {(x["s"], x["j"]): bool(x.get("unambiguous", False)) for x in cand}
        kf = dict(k=int(k), kt=self.kt, reassign=[dict(s=s, j=j, nis=info[(s, j)][0], nis_ang=_parts(s, j)[0],
                                                     nis_d=_parts(s, j)[1], unambiguous=unamb.get((s, j), False),
                                                     fresh=self.fresh(self.anchors[j])) for s, j in assign],
                  created=[], transitions=[], merges=[], deferred=[])
        for s, j in assign:
            a = self.anchors[j]
            sg = segs[s]
            a.cells |= G.cells_from_segment(sg.cells_ij, self.cs, self.K, sg.n, sg.d, T, a.grid_o, a.grid_B, self.h)
        used = {s for s, _ in assign}
        assigned = {j for _, j in assign}
        amb = {}
        for s, j in ambig:
            amb.setdefault(s, []).append(j)
        new_ids = set()
        for s, sg in enumerate(segs):
            if s in used:
                continue
            js = amb.get(s, [])
            if js and all(j not in assigned for j in js):
                j = js[0]                                   # ambiguous, j not assigned: defer (PLAN core §3.6)
                a = self.anchors[j]
                d = dict(s=s, j=j, fresh_j=self.fresh(a))
                kf["deferred"].append(d)
                self.counters["deferred"] += 1
                self._event(dict(type="deferred", k=int(k), **d))
                if not self.fresh(a):
                    r, S, *_ = pair_S(sg, self.geom_y(a), a.est.V, T, None)
                    self.counters["revisit_candidates"] += 1
                    self._event(dict(type="revisit_candidate", k=int(k), s=s, j=j, r=r, S=S,
                                     gap=self.kt - a.est.tau))
                continue
            a = self._new_anchor(sg, T, k, "candidate")
            reason = "new"
            for j in js:                                    # ambiguous with an assigned j: certainly another surface
                pj = self.anchors[j].primary_id
                key = (min(a.id, pj), max(a.id, pj))
                self.diff[key] = max(self.diff.get(key, 0), 1)
                reason = f"ambiguous_with_assigned_{j}"
            new_ids.add(a.id)
            kf["created"].append(dict(id=a.id, state=a.state, reason=reason))
        for a in self.primaries():
            if a.id in new_ids:
                continue
            if a.id in assigned:
                a.matched_kf += 1
                a.inview_unmatched_kf = 0
                continue
            nvis = G.n_visible_cells(a.cells, a.grid_o, a.grid_B, self.h, T, self.K, Z, self.sensor.sigma_z, 0.0,
                                     self.H, self.W)
            if nvis >= self.inview_min:
                a.inview_unmatched_kf += 1
        for a in list(self.primaries()):
            if a.id in new_ids or a.state != "candidate":
                continue
            to = "active" if a.matched_kf >= 1 else ("deleted" if a.inview_unmatched_kf >= self.L else None)
            if to is None:
                continue
            if ("candidate", to) not in ALLOWED:
                self._violation("I7", f"transition candidate->{to}")
            kf["transitions"].append([a.id, "candidate", to])
            if to == "deleted":
                self.counters["cand_deleted"] += 1
                self._event(dict(type="delete", k=int(k), id=a.id))
                del self.anchors[a.id]
            else:
                a.state = "active"
        while True:
            best = self._best_merge()
            if best is None:
                break
            kf["merges"].append(self._merge(*best, k))
        for a in self.anchors.values():
            if _hash(a.n, a.o, a.B, a.P) != a.hash0:
                self._violation("I4", f"anchor {a.id} parameters changed")
            if a.state == "merged" and (a.primary_id not in self.anchors or self.anchors[a.primary_id].state == "merged"):
                self._violation("I9", f"secondary {a.id} -> {a.primary_id}")
        kf["n_primary"] = len(self.primaries())
        kf["states"] = {st: sum(1 for a in self.primaries() if a.state == st) for st in ("candidate", "active")}
        kf["bias"] = {x: self.counters[x] for x in ("bias_accept", "bias_reject", "bias_ambiguous")}
        self._write_kf(kf)
        self.ghash_kf = self.ghash()

    def _aa(self, a, b):
        """r(y_a | y_b) in the world with S = J_a V_a J_a^T + J_b V_b J_b^T."""
        an, ao, aB = self.geom_y(a)
        bn, bo, bB = self.geom_y(b)
        r = residual(an, ao, bn, bo, bB)
        return r, residual_cov(jac_a(an, aB, bn, bB), a.est.V, jac_b(an, ao, bn, bo, bB), b.est.V)

    def _common_cells(self, a, b):
        if not a.cells or not b.cells:
            return 0
        X = G.grid_centres(b.cells, b.grid_o, b.grid_B, self.h)
        return sum(1 for x in G.grid_index(X, a.grid_o, a.grid_B, self.h).tolist() if tuple(x) in a.cells)

    def _merge_ok(self, aj, ak):
        if self._diff(aj.id, ak.id) >= self.m_diff or not (self.fresh(aj) and self.fresh(ak)):
            return None
        if self._common_cells(aj, ak) < 1:
            return None
        r1, S1 = self._aa(ak, aj)
        if ang_nis(r1, S1) > CHI2_2:
            return None
        r2, S2 = self._aa(aj, ak)
        v1, v2 = nis(r1, S1), nis(r2, S2)
        if v1 <= CHI2_3 and v2 <= CHI2_3:
            return max(v1, v2), (r1, S1, v1, v2)
        return None

    def _best_merge(self):
        prim = self.primaries()
        best = None
        for i in range(len(prim)):
            for l in range(i + 1, len(prim)):
                res = self._merge_ok(prim[i], prim[l])
                if res is not None and (best is None or res[0] < best[2]):
                    best = (prim[i].id, prim[l].id, res[0], res[1])
        return best

    def _merge(self, jid, kid, key, extra, k):
        aj, ak = self.anchors[jid], self.anchors[kid]
        if self._merge_ok(aj, ak) is None:
            self._violation("I7", f"merge of {jid},{kid} without all conditions")
        keep, drop = (aj, ak) if (aj.t_ref, -aj.matched_kf, aj.id) <= (ak.t_ref, -ak.matched_kf, ak.id) else (ak, aj)
        yk, yd = self.geom_y(keep), self.geom_y(drop)
        ev = dict(type="merge", k=int(k), kt=self.kt, keep=keep.id, drop=drop.id, nis=[extra[2], extra[3]],
                  common_cells=self._common_cells(aj, ak), tau=[keep.est.tau, drop.est.tau],
                  bhat=[keep.est.b, drop.est.b], V=[keep.est.V, drop.est.V],
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
                 bhat=a.est.b.tolist(), V=a.est.V.tolist(), tau=a.est.tau, f=a.est.f, n_accept=a.est.n_accept,
                 n_reject=a.est.n_reject)
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
                 ms_pw_mean=float(np.mean(self.ms_pw)) if self.ms_pw else None,
                 ms_pw_p95=float(np.percentile(self.ms_pw, 95)) if self.ms_pw else None)
        return d

    def close(self):
        if self.out_dir is not None:
            with open(os.path.join(self.out_dir, "anchors_final.json"), "w") as f:
                json.dump(_j(dict(anchors=[self._row(a, cells=True) for a in self.anchors.values()],
                                  counters=self.counters)), f)
        for f in (self.f_frames, self.f_kf, self.f_ev):
            if f is not None:
                f.close()
