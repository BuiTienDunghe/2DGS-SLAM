"""PLAN A2G-P1 (rút gọn) §3.6–3.12: plane world — anchors, evidence ledger, life cycle, merging, API, JSONL logs.

Two modes only: off (frontend runs the untouched E1c path) and on (everything below). Never reads ground truth
(I12): the only pose it receives for frame 0 is the frontend's cam_start.T, as E1c does.
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
from .association import ambiguity, build_jc, ic_and_assign, init_from_jc, jc_stats, pair_S, pair_terms
from .model import (CHI2_2, CHI2_3, DriftModel, PredNoise, SensorNoise, ang_nis, chi2, jac_a, jac_b, jac_xi,
                    nis, residual, residual_cov, shift_matrix)
from .segments import build_segments, merge_in_frame

ALLOWED = {("candidate", "active"), ("active", "frozen"), ("conflict", "candidate"), ("conflict", "deleted"),
           ("candidate", "deleted"), ("candidate", "merged"), ("active", "merged"), ("frozen", "merged"),
           ("candidate", "active_by_merge")}


@dataclass
class AnchorP1:                     # world frame
    id: int
    primary_id: int
    n: np.ndarray
    o: np.ndarray
    B: np.ndarray
    P: np.ndarray
    grid_o: np.ndarray
    grid_B: np.ndarray
    cells: set
    state: str                      # candidate | active | frozen | conflict | merged
    conflict_with: Optional[int]
    T_ref: np.ndarray
    t_ref: int
    s_ref: float
    phi_ref: float
    created_kf: int
    frozen_kf: Optional[int] = None
    last_R_dd: float = 0.0
    last_p_w: np.ndarray = None
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


def np_pose(T):
    """float64 numpy pose with the rotation projected onto SO(3). The frontend keeps poses in float32 and composes
    left updates without re-orthonormalising (|R R^T - I| reaches 0.04-0.15 on fr3 far); planes taken through such a
    matrix get non-unit normals, so every pose entering the plane world is projected first."""
    T = T.detach().cpu().double().numpy() if isinstance(T, torch.Tensor) else np.array(T, dtype=np.float64)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    out = T.copy()
    out[:3, :3] = R
    return out


def _j(x):
    if isinstance(x, np.ndarray):
        return [_j(v) for v in x.tolist()] if x.ndim else float(x)
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): _j(v) for k, v in x.items()}
    return x


class PlaneWorld:
    def __init__(self, cfg, K, H, W, legacy, out_dir=None):
        self.cfg = cfg
        self.K = np.asarray(K.detach().cpu().double().numpy() if isinstance(K, torch.Tensor) else K, dtype=np.float64)
        self.H, self.W = int(H), int(W)
        self.legacy = legacy                               # E1c PlaneAnchorMap (sigmas of the loss, min_sv)
        se = cfg.get("sensor", {}) or {}
        self.sensor = SensorNoise(se.get("a", 0.0), se.get("b", 1.425e-3), se.get("sigma_theta0_deg", 0.5),
                                  se.get("sigma_d0_m", 0.005))
        dr = cfg.get("drift", {}) or {}
        if dr.get("kappa_t") is None or dr.get("kappa_r") is None:
            raise ValueError("PlaneWorld.drift.kappa_t / kappa_r are not set (run tools/a2g_p1/quick_calib.py)")
        self.drift = DriftModel("linear", dr["kappa_t"], dr["kappa_r"])
        pi = cfg.get("pred_init", {}) or {}
        self.pred = PredNoise(cfg.get("pred_window", 30), pi.get("sigma_rho_m", 0.01), pi.get("sigma_theta_deg", 0.5))
        self.h = float(cfg.get("grid_h", 0.05))
        self.L = int(cfg.get("L_kf", 10))
        self.m_diff = int(cfg.get("m_diff", 2))
        self.inview_min = int(cfg.get("inview_min_cells", 3))
        self.obs_thr = float(cfg.get("obs_threshold", 0.3))
        self.gn_iters = int(cfg.get("gn_iters", 3))
        self.debug = bool(cfg.get("debug_checks", False))
        self.cs = int(getattr(legacy, "cell", 20))
        self.anchors = {}
        self.next_id = 0
        self.diff = {}
        self.s, self.phi = 0.0, 0.0
        self.last_T = None
        self.cur = None
        self.ghash_kf = None
        self.n_invariant = 0
        self.counters = dict(merges=0, conflicts=0, revisits=0, cand_deleted=0, jc_jump=0, jc_shape_drop=0,
                             rank6=0, frames=0, plane_init=0)
        self.ms_pw = []
        self.f_frames = self.f_kf = self.f_ev = None
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            self.f_frames = open(os.path.join(out_dir, "pw_frames.jsonl"), "w")
            self.f_kf = open(os.path.join(out_dir, "pw_kf.jsonl"), "w")
            self.f_ev = open(os.path.join(out_dir, "pw_events.jsonl"), "w")

    # ------------------------------------------------------------------ helpers
    def primaries(self):
        return [a for a in self.anchors.values() if a.state != "merged"]

    def C_to(self, a, s=None, phi=None):
        return self.drift.C(a.s_ref, a.phi_ref, self.s if s is None else s, self.phi if phi is None else phi)

    def cov_eps(self, ai, al):
        A, Bq = self.anchors[ai], self.anchors[al]
        return self.drift.cov_eps(self.s, self.phi, A.s_ref, A.phi_ref, Bq.s_ref, Bq.phi_ref)

    def _event(self, d):
        if self.f_ev is not None:
            self.f_ev.write(json.dumps(_j(d)) + "\n")
            self.f_ev.flush()

    def _violation(self, iid, msg):
        self.n_invariant += 1
        self._event(dict(type="invariant", id=iid, msg=msg, k=self.cur["k"] if self.cur else None))
        if self.debug:
            raise AssertionError(f"PlaneWorld invariant {iid}: {msg}")

    def ghash(self):
        h = hashlib.sha1()
        for a in sorted(self.anchors.values(), key=lambda x: x.id):
            h.update(np.concatenate([a.n, a.o, a.P.ravel()]).tobytes())
            h.update(a.state.encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _hash_frozen(a):
        return hashlib.sha1(np.concatenate([a.n, a.o, a.P.ravel()]).tobytes()).hexdigest()[:16]

    def _segs(self, planes):
        return merge_in_frame(build_segments(planes, self.sensor), planes, self.sensor)

    def _new_anchor(self, seg, T, k, state, conflict_with=None):
        n, o, B = G.to_world(T, seg.n, seg.p, seg.B)
        n = n / np.linalg.norm(n)
        B = G.gram_schmidt(B, n)
        aid = self.next_id
        self.next_id += 1
        a = AnchorP1(id=aid, primary_id=aid, n=n, o=o, B=B, P=seg.R.copy(), grid_o=o.copy(), grid_B=B.copy(),
                     cells=set(), state=state, conflict_with=conflict_with, T_ref=T.copy(), t_ref=int(k), s_ref=self.s,
                     phi_ref=self.phi, created_kf=int(k), last_p_w=o.copy())
        a.cells = G.cells_from_segment(seg.cells_ij, self.cs, self.K, seg.n, seg.d, T, a.grid_o, a.grid_B, self.h)
        Ja = jac_a(seg.n, seg.B, seg.n, seg.B)
        a.last_R_dd = float((Ja @ seg.R @ Ja.T)[2, 2])
        self.anchors[aid] = a
        return a

    def _sigma_min(self, segs, pose_pairs):
        if not pose_pairs:
            return 0.0
        M = np.stack([segs[s].n for s, _ in pose_pairs])
        sv = np.linalg.svd(M, compute_uv=False)
        return float(sv[2]) if sv.shape[0] >= 3 else 0.0

    def _ctx(self, segs, pose_pairs, device, dtype):
        if not pose_pairs:
            return None
        w = torch.tensor([float(segs[s].meas.n_px) for s, _ in pose_pairs], dtype=torch.float64)
        return dict(
            m=torch.tensor(np.stack([segs[s].n for s, _ in pose_pairs]), dtype=dtype, device=device),
            n_w=torch.tensor(np.stack([self.anchors[a].n for _, a in pose_pairs]), dtype=dtype, device=device),
            d_meas=torch.tensor([segs[s].d for s, _ in pose_pairs], dtype=dtype, device=device),
            d_w=torch.tensor([-float(self.anchors[a].n @ self.anchors[a].o) for _, a in pose_pairs], dtype=dtype,
                             device=device),
            w=(w / w.sum()).to(device=device, dtype=dtype),
            inv_sn2=1.0 / self.legacy.sigma_n ** 2,
            inv_sd2=1.0 / self.legacy.sigma_d ** 2,
        )

    # ------------------------------------------------------------------ frame 0
    def on_first_frame(self, k, planes, T0):
        t0 = time.perf_counter()
        T = np_pose(T0)
        segs = self._segs(planes)
        for sg in segs:
            self._new_anchor(sg, T, k, "active")
        self.last_T = T
        self.ghash_kf = self.ghash()
        self._write_kf(dict(k=int(k), created=[dict(id=a.id, state="active", reason="frame0")
                                               for a in self.anchors.values()], ekf=[], transitions=[],
                            n_primary=len(self.primaries())))
        return dict(planes=[s.meas for s in segs], pairs=[], init_source="prev", sigma_min=float("nan"),
                    ms_extract=0.0, ms_match=1000 * (time.perf_counter() - t0), ms_solve=0.0, T_init=T0, active_ids=[])

    # ------------------------------------------------------------------ every frame (steps 1–5)
    def on_frame(self, k, planes, T_pred):
        t0 = time.perf_counter()
        if self.ghash_kf is not None and self.ghash() != self.ghash_kf:
            self._violation("I6", "anchor geometry changed outside a keyframe")
        T = np_pose(T_pred)
        segs = self._segs(planes)
        Q = self.pred.Q()
        prim = self.primaries()
        cand, assign, info = ic_and_assign(segs, prim, T, lambda a: Q + self.C_to(a))
        t1 = time.perf_counter()
        ambig = ambiguity(segs, assign, prim, T, lambda a: Q + self.C_to(a), self.h)
        pose = [(s, j) for s, j in assign if self.anchors[j].state in ("active", "frozen")]
        res = init_from_jc(pose, segs, self.anchors, T, Q, self.cov_eps, self.obs_thr, self.gn_iters)
        removed = res["removed"]
        if removed:
            assign = [p for p in assign if p not in removed]
            self.counters["jc_shape_drop"] += len(removed)
        source = res["source"]
        T_init = res["T_init"]
        pose_pairs = res["pose_pairs"]
        added = []
        if source == "plane":
            self.counters["plane_init"] += 1
            used_s = {s for s, _ in assign}
            used_a = {j for _, j in assign}
            free_s = [i for i in range(len(segs)) if i not in used_s]
            free_a = {a.id for a in prim if a.id not in used_a}
            if free_s and free_a:
                Ip = res["Ip"]
                c2, a2, info2 = ic_and_assign(segs, prim, T_init, lambda a: Ip + self.C_to(a), restrict_s=free_s,
                                              restrict_a=free_a)
                if a2:
                    new_pose = [(s, j) for s, j in a2 if self.anchors[j].state in ("active", "frozen")]
                    ok = True
                    if new_pose:
                        allp = pose_pairs + new_pose
                        r0, J, Sig = build_jc(allp, segs, self.anchors, T, Q, self.cov_eps)
                        st = jc_stats(r0, J, Sig)
                        dof = 3 * len(allp) - st["rank"]
                        ok = dof <= 0 or st["shape"] <= chi2(dof)
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
        ctx = self._ctx(segs, pose_pairs, T_pred.device, T_pred.dtype)
        # tracking starts from the frontend's own T_pred unless the plane world provides a new pose
        T_init_t = (torch.from_numpy(T_init).to(device=T_pred.device, dtype=T_pred.dtype) if source == "plane"
                    else T_pred)
        self.cur = dict(k=int(k), segs=segs, T_pred=T, assign=assign, pose_pairs=pose_pairs, ambig=ambig, cand=cand,
                        info=info, jc=res["jc"], source=source, dxi=res["dxi"], Q=Q, added=added)
        if self.debug or True:
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
        ss = [s for s, _ in c["assign"]]
        aa = [j for _, j in c["assign"]]
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
            if self.anchors[j].state not in ("active", "frozen"):
                self._violation("I3", f"pose pair with {self.anchors[j].state} anchor {j}")
        if c["source"] == "jc_jump" and c["pose_pairs"]:
            self._violation("I3", "jump failed but pose pairs kept")
        if c["source"] == "plane":
            jc = c["jc"]
            if not (jc["rank"] == 6 and jc["jump"] <= chi2(6)):
                self._violation("I10", f"plane init with rank {jc['rank']} jump {jc['jump']:.2f}")

    # ------------------------------------------------------------------ after tracking (steps 7–8)
    def after_tracking(self, k, T_opt):
        c = self.cur
        T = np_pose(T_opt)
        self.pred.push(G.se3_log(T @ G.inv_T(c["T_pred"])))
        if self.last_T is not None:
            self.s += float(np.linalg.norm(G.centre(T) - G.centre(self.last_T)))
            self.phi += G.rot_angle(T[:3, :3] @ self.last_T[:3, :3].T)
        self.last_T = T
        ids = sorted(self.anchors[j].primary_id for _, j in c["assign"])
        for i in range(len(ids)):
            for l in range(i + 1, len(ids)):
                key = (ids[i], ids[l])
                self.diff[key] = self.diff.get(key, 0) + 1
        if self.f_frames is not None:
            sr, st = self.pred.sigmas()
            row = dict(k=c["k"], segs=[dict(src=s.src, N_s=s.N_s, n=s.n, p=s.p) for s in c["segs"]],
                       cand=[dict(s=x["s"], j=x["j"], state=x["state"], nis=x["nis"], ic=x["ic"],
                                  blocked=x.get("blocked", False), at=x.get("at", "T_pred")) for x in c["cand"]],
                       assign=c["assign"], pose=c["pose_pairs"], ambig=c["ambig"], added=c["added"],
                       jc=c["jc"], init=dict(source=c["source"], dxi=c["dxi"]), Qpred=[sr, st], s=self.s,
                       phi=self.phi, ghash=self.ghash(), T_opt=T[:3, :])
            self.f_frames.write(json.dumps(_j(row)) + "\n")
            self.f_frames.flush()

    # ------------------------------------------------------------------ keyframe (steps 9–15)
    def _diff(self, a, b):
        a, b = self.anchors[a].primary_id, self.anchors[b].primary_id
        return self.diff.get((min(a, b), max(a, b)), 0)

    def on_keyframe(self, k, T_opt, depth):
        c = self.cur
        T = np_pose(T_opt)
        Z = depth.detach().cpu().float().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
        segs = c["segs"]
        prim = self.primaries()
        cand, assign, info = ic_and_assign(segs, prim, T, lambda a: self.C_to(a), kind="seg_anchor_opt")
        ambig = ambiguity(segs, assign, prim, T, lambda a: self.C_to(a), self.h)
        kf = dict(k=int(k), reassign=assign, ambig=ambig, created=[], ekf=[], transitions=[], merges=[])
        # 10. EKF, support region, re-centre o
        for s, j in assign:
            a = self.anchors[j]
            sg = segs[s]
            if a.state in ("candidate", "active", "conflict"):
                Cj = self.C_to(a)
                r, Ja, Jb, Jx = pair_terms(sg, a, T)
                Hm = -Jb
                N = Ja @ sg.R @ Ja.T + Jx @ Cj @ Jx.T
                S = Hm @ a.P @ Hm.T + N
                Kg = a.P @ Hm.T @ np.linalg.inv(S)
                dx = Kg @ r
                IKH = np.eye(3) - Kg @ Hm
                trace0 = float(np.trace(a.P))
                a.P = IKH @ a.P @ IKH.T + Kg @ N @ Kg.T
                a.P = 0.5 * (a.P + a.P.T)
                a.n, a.o, a.B = G.oplus(a.n, a.o, a.B, dx)
                a.last_R_dd = float((Ja @ sg.R @ Ja.T)[2, 2])
                kf["ekf"].append(dict(id=j, nis=float(r @ np.linalg.solve(S, r)), trace0=trace0,
                                      trace1=float(np.trace(a.P))))
                if np.min(np.linalg.eigvalsh(a.P)) <= 0:
                    self._violation("I8", f"P of {j} not positive definite")
            a.last_p_w = G.to_world(T, sg.n, sg.p, sg.B)[1]
            a.cells |= G.cells_from_segment(sg.cells_ij, self.cs, self.K, sg.n, sg.d, T, a.grid_o, a.grid_B, self.h)
            if a.state != "frozen" and a.cells:
                cen = G.grid_centres(a.cells, a.grid_o, a.grid_B, self.h).mean(0)
                l2 = a.B.T @ (cen - a.o)
                Am = shift_matrix(l2)
                a.o = a.o + a.B @ l2
                a.P = Am @ a.P @ Am.T
        # 11. new anchors from unassigned segments
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
                    key = (min(a.id, self.anchors[j].primary_id), max(a.id, self.anchors[j].primary_id))
                    self.diff[key] = max(self.diff.get(key, 0), 1)
                if js:
                    reason = f"ambiguous_with_assigned_{js[0]}"
            a.conflict_diff_seen = self._diff(a.id, a.conflict_with) if a.conflict_with is not None else 0
            new_ids.add(a.id)
            kf["created"].append(dict(id=a.id, state=a.state, reason=reason))
        # 12. ledger: matched / in view
        for a in self.primaries():
            if a.id in new_ids:
                continue
            if a.id in assigned_anchors:
                a.matched_kf += 1
                a.inview_unmatched_kf = 0
                continue
            sig_rho = self.drift.kt * abs(self.s - a.s_ref)
            nvis = G.n_visible_cells(a.cells, a.grid_o, a.grid_B, self.h, T, self.K, Z, self.sensor.sigma_z, sig_rho,
                                     self.H, self.W)
            inview = nvis >= self.inview_min
            if inview:
                a.inview_unmatched_kf += 1
            if a.state == "conflict" and inview:
                d = self._diff(a.id, a.conflict_with)
                if d <= a.conflict_diff_seen:
                    a.conflict_quiet += 1
                a.conflict_diff_seen = d
        # 13. state transitions
        for a in list(self.primaries()):
            if a.id in new_ids:
                continue
            frm = a.state
            to = None
            if a.state == "conflict":
                jstar = self.anchors.get(a.conflict_with)
                if jstar is None:
                    to = "candidate"
                else:
                    if jstar.state == "merged":
                        a.conflict_with = jstar.primary_id
                    if self._diff(a.id, a.conflict_with) >= self.m_diff:
                        to = "candidate"
                    elif a.conflict_quiet >= self.L:
                        to = "deleted"
                        js = self.anchors[a.conflict_with]
                        nc, oc, Bc = G.to_camera(T, js.n, js.o, js.B)
                        an, ao, aB = G.to_camera(T, a.n, a.o, a.B)
                        r = residual(an, ao, nc, oc, Bc)
                        S = residual_cov("anchor_anchor", jac_a(an, aB, nc, Bc), a.P, jac_b(an, ao, nc, oc, Bc), js.P,
                                         jac_xi(an, ao, nc, Bc), self.drift.C(js.s_ref, js.phi_ref, a.s_ref, a.phi_ref))
                        self.counters["revisits"] += 1
                        self._event(dict(type="revisit", k=int(k), c=a.id, jstar=a.conflict_with, r=r, S=S))
            elif a.state == "candidate":
                if a.matched_kf >= 1:
                    to = "active"
                elif a.inview_unmatched_kf >= self.L:
                    to = "deleted"
                    self.counters["cand_deleted"] += 1
            elif a.state == "active" and a.last_p_w is not None:
                nc, oc, Bc = G.to_camera(T, a.n, a.o, a.B)
                p_c = T[:3, :3] @ a.last_p_w + T[:3, 3]
                row = jac_xi(nc, p_c, nc, Bc)[2]
                if float(row @ self.C_to(a) @ row) >= a.last_R_dd:
                    to = "frozen"
                    a.frozen_kf = int(k)
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
        # 14. anchor-anchor merging
        while True:
            best = self._best_merge(k)
            if best is None:
                break
            kf["merges"].append(self._merge(*best, k))
        # invariants on the anchor set
        for a in self.anchors.values():
            if a.state == "merged" and (a.primary_id not in self.anchors or self.anchors[a.primary_id].state == "merged"):
                self._violation("I9", f"secondary {a.id} -> {a.primary_id}")
            if a.state == "frozen":
                hsh = self._hash_frozen(a)
                if getattr(a, "_fh", None) is None:
                    a._fh = hsh
                elif a._fh != hsh:
                    self._violation("I5", f"frozen anchor {a.id} changed")
        kf["n_primary"] = len(self.primaries())
        kf["states"] = {st: sum(1 for a in self.primaries() if a.state == st)
                        for st in ("candidate", "active", "frozen", "conflict")}
        self._write_kf(kf)
        self.ghash_kf = self.ghash()

    def _pair_aa(self, aj, ak):
        """r(k|j) and r(j|k) with K5, in the camera of T_ref of the anchor created later."""
        later = aj if aj.t_ref >= ak.t_ref else ak
        T = later.T_ref
        Cjk = self.drift.C(aj.s_ref, aj.phi_ref, ak.s_ref, ak.phi_ref)
        out = []
        for a, b in ((ak, aj), (aj, ak)):
            an, ao, aB = G.to_camera(T, a.n, a.o, a.B)
            bn, bo, bB = G.to_camera(T, b.n, b.o, b.B)
            r = residual(an, ao, bn, bo, bB)
            S = residual_cov("anchor_anchor", jac_a(an, aB, bn, bB), a.P, jac_b(an, ao, bn, bo, bB), b.P,
                             jac_xi(an, ao, bn, bB), Cjk)
            out.append((r, S))
        return out

    def _regions_touch(self, a, b):
        if not a.cells or not b.cells:
            return False
        X = G.grid_centres(b.cells, b.grid_o, b.grid_B, self.h)
        return G.overlap_or_adjacent(G.grid_index(X, a.grid_o, a.grid_B, self.h), a.cells)

    def _best_merge(self, k):
        prim = [a for a in self.primaries() if a.state != "conflict"]
        best = None
        for i in range(len(prim)):
            for l in range(i + 1, len(prim)):
                aj, ak = prim[i], prim[l]
                if self._diff(aj.id, ak.id) >= self.m_diff:
                    continue
                (r1, S1), (r2, S2) = self._pair_aa(aj, ak)
                if ang_nis(r1, S1) > CHI2_2 or not self._regions_touch(aj, ak):
                    continue
                v1, v2 = nis(r1, S1), nis(r2, S2)
                if v1 <= CHI2_3 and v2 <= CHI2_3:
                    key = max(v1, v2)
                    if best is None or key < best[2]:
                        best = (aj.id, ak.id, key, (r1, S1, v1, v2))
        return best

    def _u(self, a):
        nc, oc, Bc = G.to_camera(a.T_ref, a.n, a.o, a.B)
        Jx = jac_xi(nc, oc, nc, Bc)
        return float((a.P + Jx @ self.drift.C(0.0, 0.0, a.s_ref, a.phi_ref) @ Jx.T)[2, 2])

    def _merge(self, jid, kid, key, extra, k):
        aj, ak = self.anchors[jid], self.anchors[kid]
        if self._diff(jid, kid) >= self.m_diff or "conflict" in (aj.state, ak.state):
            self._violation("I7", f"merge of {jid},{kid} in E or conflict")
        keep, drop = (aj, ak) if self._u(aj) <= self._u(ak) else (ak, aj)
        ev = dict(type="merge", k=int(k), keep=keep.id, drop=drop.id, nis=[extra[2], extra[3]], r=extra[0],
                  S=extra[1], keep_par=dict(n=keep.n, o=keep.o, state=keep.state, n_cells=len(keep.cells)),
                  drop_par=dict(n=drop.n, o=drop.o, state=drop.state, n_cells=len(drop.cells)),
                  dist_at_o_drop=float(keep.n @ (drop.o - keep.o)),
                  angle_deg=float(np.degrees(np.arccos(np.clip(abs(keep.n @ drop.n), -1, 1)))))
        if keep.state == "candidate" and drop.state in ("active", "frozen"):
            keep.state = "active"
        if drop.cells:
            X = G.grid_centres(drop.cells, drop.grid_o, drop.grid_B, self.h)
            keep.cells |= {tuple(x) for x in G.grid_index(X, keep.grid_o, keep.grid_B, self.h).tolist()}
        for (a, b), v in list(self.diff.items()):
            if drop.id in (a, b):
                o = b if a == drop.id else a
                if o == keep.id:
                    continue
                key2 = (min(keep.id, o), max(keep.id, o))
                self.diff[key2] = self.diff.get(key2, 0) + v
        keep.matched_kf += drop.matched_kf
        drop.state = "merged"
        drop.primary_id = keep.id
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

    def anchors_table(self):
        out = []
        for a in sorted(self.anchors.values(), key=lambda x: x.id):
            out.append(dict(id=a.id, n_w=a.n.tolist(), d_w=-float(a.n @ a.o), n_obs_kf=a.matched_kf + 1,
                            active=a.state in ("active", "frozen"), created_kf=a.created_kf, n_samples=len(a.cells),
                            primary_id=a.primary_id, state=a.state, o=a.o.tolist(), P=a.P.tolist(), t_ref=a.t_ref,
                            n_cells=len(a.cells), frozen_kf=a.frozen_kf, conflict_with=a.conflict_with))
        return out

    def n_primary(self):
        return len(self.primaries())

    def n_active(self):
        return sum(1 for a in self.primaries() if a.state in ("active", "frozen"))

    def summary(self):
        d = dict(self.counters)
        d.update(n_invariant=self.n_invariant, n_primary_final=self.n_primary(), n_active_final=self.n_active(),
                 ms_pw_mean=float(np.mean(self.ms_pw)) if self.ms_pw else None,
                 ms_pw_p95=float(np.percentile(self.ms_pw, 95)) if self.ms_pw else None)
        return d

    def close(self):
        for f in (self.f_frames, self.f_kf, self.f_ev):
            if f is not None:
                f.close()
