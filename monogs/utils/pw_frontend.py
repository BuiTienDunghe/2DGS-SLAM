"""PLAN MONOGS-PW: plane world of the methods full*, sgs inside the MonoGS frontend (created only when
PlaneTracking.pw is true; the E1c path of slam_frontend.py is untouched).

Per frame : match (nn / e1c) with the predicted pose -> robust closed-form init -> robust plane loss (E1c scale or
            1b information scale) in the decoupled pose step -> inlier bookkeeping at the final pose (T1, T2 state).
Keyframe  : anchors attached to the creator keyframe (or E1c average for full_nolife); closure candidates.
After the keyframe's local BA (frontend receives "keyframe"): PGO ours / sgs, applied to frontend, backend,
Gaussians and anchors; merge after PGO.
"""
import json
import os
import time

import numpy as np
import torch

from utils.logging_utils import Log
from utils.obs_decouple import DecoupledPoseStep
from utils.plane_anchor import (NoiseModel, PWAnchorMap, PlaneExtractor, plane_c2w, plane_w2c, plane_z,
                                pw_loss_ctx, pw_plane_loss)
from utils import pgo_planes as PG


def _np(T):
    return T.detach().cpu().double().numpy()


def basis_normals(amap, planes, act, T):
    """PLAN MONOGS-PW2 §2.3: measured normals of the active pairs that are inliers (e <= inlier_e) at pose T."""
    inl = [planes[pi].n.numpy() for pi, aid in act if amap.e_pair(planes[pi], amap.anchors[aid], T) <= amap.inlier_e]
    return np.stack(inl) if inl else np.zeros((0, 3))


class PWFront:
    def __init__(self, fe):
        cfg = fe.config
        self.pt = pt = cfg["PlaneTracking"]
        nz = cfg["Noise"]
        self.noise = NoiseModel(nz["sigma_n"], nz["sigma_d_a"], nz["sigma_d_b"])
        self.noise_pgo = self.noise
        if "sigma_eff_n" in nz:   # PLAN MONOGS-PW2 §2.1: sigma_eff for kernel / inlier, sigma_pgo for the PGO factors
            self.noise = NoiseModel(nz["sigma_eff_n"], nz["sigma_eff_d_a"], nz["sigma_eff_d_b"])
        self.basis_from = pt.get("basis_from", "all")
        self.s_odo_t, self.s_odo_r = float(nz["sigma_odo_t"]), float(nz["sigma_odo_r"])
        self.merge_deg, self.merge_m = float(nz["merge_deg"]), float(nz["merge_m"])
        self.scale_info = pt.get("scale_mode", "e1c") == "info"
        k = nz.get("kappa")
        if self.scale_info and k is None:
            Log("[PW] kappa not calibrated: using 1.0", tag="PW")
        self.kappa = float(k) if k is not None else 1.0
        self.K = np.array([[fe.dataset.fx, 0.0, fe.dataset.cx], [0.0, fe.dataset.fy, fe.dataset.cy], [0.0, 0.0, 1.0]])
        self.extractor = PlaneExtractor(pt)
        self.map = PWAnchorMap(pt, self.K, fe.dataset.height, fe.dataset.width, self.noise)
        self.robust = bool(pt.get("robust", True))
        self.sig_n_e1c, self.sig_d_e1c = float(pt.get("sigma_normal", 0.02)), float(pt.get("sigma_dist", 0.01))
        self.lam_e1c = float(pt.get("lambda_plane", 1.0))
        self.t1_on = bool(pt.get("kf_trigger_t1", False))
        self.t1_frames, self.t1_top, self.t1_gap = int(pt.get("t1_frames", 3)), int(pt.get("t1_top", 3)), int(pt.get("t1_min_gap", 10))
        pg = cfg.get("PGO", {})
        self.pgo_mode = pg.get("mode", "none")
        self.pgo_gap, self.force_every = int(pg.get("min_kf_gap", 3)), int(pg.get("force_every_kf", 0) or 0)
        self.huber_k, self.chi2 = float(pg.get("huber_k", 1.345)), float(pg.get("chi2_reject", 11.34))
        self.max_dt, self.max_dr = float(pg.get("max_dt_m", 1.0)), float(pg.get("max_dr_deg", 30.0))
        self.max_nfev = int(pg.get("max_nfev", 100))
        self.pgo_robust = pg.get("robust", "scipy_huber")
        self.t1_count = 0
        self.last_t1_kf = -10 ** 9
        self.frame_cls = []                 # classification of the current frame at its final pose
        self.frame_tiers = ([], [])
        self.pinfo = None
        self.ctx = None
        self.pending = None                 # closure candidates of the newest keyframe, waiting for its local BA
        self.kf_since_pgo = 10 ** 9
        self.n_kf = 0
        self.frozen = {}                    # (i, j) -> frozen relative pose of consecutive keyframes
        self.sgs_edges = []
        self.stats = dict(init_dropped=0, cand_pairs=0, rejected_pairs=0, pgo_applied=0, pgo_aborted=0, tier1=0,
                          tier2=0, t1_kf=0, t2_kf=0, mono_kf=0, closure_accepted=0, closure_dropped=0)
        d = fe.save_dir
        self.f = {n: (open(os.path.join(d, n), "w") if d else None)
                  for n in ("pw_frames.jsonl", "anchors.jsonl", "events.jsonl", "pgo.jsonl", "e1c_frames.jsonl")
                  + (("pgo_poses.jsonl",) if self.pgo_robust == "irls_factor" else ())}
        Log(f"[PW] method={fe.config['Repro']['method']} match={self.map.match_mode} robust={self.robust} "
            f"anchor={self.map.anchor_mode} t1={self.t1_on} scale={'info' if self.scale_info else 'e1c'} "
            f"pgo={self.pgo_mode} sigma_n={np.degrees(self.noise.sn):.3f}deg sigma_d=({self.noise.a:.4f}+{self.noise.b:.5f}z^2) "
            f"kappa={self.kappa:.4g}", tag="PW")

    # ------------------------------------------------------------------ logging
    def _w(self, name, row):
        f = self.f.get(name)
        if f is not None:
            f.write(json.dumps(row) + "\n")
            f.flush()

    def _flush_events(self):
        for e in self.map.events:
            self._w("events.jsonl", e)
        self.map.events = []

    def _log_anchors(self, idx, event):
        rows = [dict(id=int(a.id), n_w=[float(v) for v in a.n_w.numpy()], d_w=float(a.d_w), n_obs_kf=int(a.n_obs_kf),
                     active=bool(a.active), created_kf=int(a.created_kf), creator_kf=int(getattr(a, "creator_kf", a.created_kf)),
                     alias_of=None, last_obs_kf=int(getattr(a, "last_obs_kf", a.created_kf)))
                for a in self.map.anchors.values()]
        self._w("anchors.jsonl", dict(frame=int(idx), event=event, anchors=rows,
                                      aliases={str(k): int(v) for k, v in self.map.alias.items()}))

    def kf_poses(self, fe):
        return {k: _np(fe.T_w2c(fe.cameras[k])) for k in fe.kf_indices if k in fe.cameras}

    # ------------------------------------------------------------------ frame 0
    def init_first(self, fe, idx, viewpoint):
        planes = self.extractor(viewpoint.depth, self.K)
        T = _np(fe.T_w2c(viewpoint))
        self.map.update_at_keyframe_pw(idx, planes, T)
        self.n_kf = 1
        self.pinfo = dict(planes=planes, pairs=[], init_source="gt", sigma_min=float("nan"), n_init_dropped=0,
                          ms_extract=0.0, ms_match=0.0, ms_solve=0.0)
        self.frame_cls = [(pi, self.map.resolve(aid), 0.0, 1.0, True) for pi, aid in []]
        self.log_frame(fe, idx, viewpoint, True, "init")
        self._log_anchors(idx, "kf")
        self._flush_events()

    # ------------------------------------------------------------------ tracking
    def before_tracking(self, fe, viewpoint):
        t0 = time.perf_counter()
        planes = self.extractor(viewpoint.depth, self.K)
        t1 = time.perf_counter()
        T_init = fe.T_w2c(viewpoint)
        pairs = self.map.match_any(planes, _np(T_init))
        t2 = time.perf_counter()
        if self.robust:
            T_plane, n_drop = self.map.solve_pose_robust(planes, pairs, T_init)
        else:
            T_plane, n_drop = self.map.solve_pose(planes, pairs, T_init), 0
        t3 = time.perf_counter()
        self.stats["init_dropped"] += n_drop
        init_source = "rejected" if self.map.last_solve_status in ("rejected", "robust_drop") else "prev"
        if bool(self.pt.get("use_init", True)) and T_plane is not None:
            fe.set_T_w2c(viewpoint, T_plane)
            init_source = "plane"
        act = self.map.active_pairs(pairs)
        self.ctx = pw_loss_ctx(planes, act, self.map.anchors, self.noise, viewpoint.R.device, torch.float32,
                               self.sig_n_e1c, self.sig_d_e1c) if bool(self.pt.get("use_loss", True)) else None
        self.pinfo = dict(planes=planes, pairs=pairs, init_source=init_source, sigma_min=float(self.map.last_sigma_min),
                          n_init_dropped=int(n_drop), ms_extract=1000 * (t1 - t0), ms_match=1000 * (t2 - t1),
                          ms_solve=1000 * (t3 - t2))
        fe.plane_ms.append(1000 * (t3 - t0))
        fe._pinfo = self.pinfo
        dec = None
        if bool(self.pt.get("decouple", True)):
            m_np = self.ctx["m"].double().cpu().numpy() if self.ctx is not None else np.zeros((0, 3))
            if self.basis_from == "inliers":   # PLAN MONOGS-PW2 §2.3: basis from the inlier active pairs only
                m_np = basis_normals(self.map, planes, act, _np(fe.T_w2c(viewpoint)))
                self.pinfo["n_basis_pairs"] = int(len(m_np))
            dec = DecoupledPoseStep(m_np, float(self.pt.get("obs_threshold", self.map.min_sv)),
                                    fe.config["Training"]["lr"]["cam_trans_delta"], fe.config["Training"]["lr"]["cam_rot_delta"],
                                    device=viewpoint.R.device, dtype=torch.float32)
        return self.ctx, dec

    def loss(self, T_cw, ctx):
        l, _ = pw_plane_loss(T_cw, ctx, self.robust, self.scale_info, self.kappa, self.map.robust_c)
        return l if self.scale_info else self.lam_e1c * l

    def plane_stiffness(self, T_cw, dt, dr):
        if self.ctx is None:
            return None
        from utils.render_stiffness import plane_stiffness
        return plane_stiffness(lambda T: pw_plane_loss(T, self.ctx, self.robust, True, 1.0, self.map.robust_c)[0],
                               T_cw, dt, dr)

    def after_tracking(self, fe, idx, viewpoint):
        """Step D: classification at the final pose; T1 state; closure state of the frame for T2."""
        p = self.pinfo
        T = _np(fe.T_w2c(viewpoint))
        cls = self.map.classify(p["planes"], p["pairs"], T)
        self.frame_cls = cls
        act_cls = [c for c in cls if self.map.anchors[c[1]].active]
        self.stats["cand_pairs"] += len(act_cls)
        self.stats["rejected_pairs"] += sum(1 for c in act_cls if not c[4])
        top = sorted(range(len(p["planes"])), key=lambda i: -p["planes"][i].n_px)[:self.t1_top]
        anchored = {pi for pi, aid, e, w, inl in cls if inl and self.map.anchors[aid].active}
        self.t1_count = self.t1_count + 1 if any(pi not in anchored for pi in top) else 0
        self.frame_tiers = ([], [])
        if self.pgo_mode != "none":
            self.frame_tiers = self.map.closure_candidates(p["planes"], cls, T, fe.current_window, idx)

    def extra_kf_reason(self, fe, idx):
        if self.t1_on and self.t1_count >= self.t1_frames and idx - self.last_t1_kf >= self.t1_gap:
            self.last_t1_kf = idx
            self.t1_count = 0
            return "T1"
        if self.pgo_mode != "none" and (self.frame_tiers[0] or self.frame_tiers[1]):
            return "T2"
        return None

    def log_frame(self, fe, idx, viewpoint, is_kf, reason):
        p = self.pinfo
        if p is None:
            return
        if is_kf:
            key = {"mono": "mono_kf", "T1": "t1_kf", "T2": "t2_kf"}.get(reason)
            if key:
                self.stats[key] += 1
        T = _np(fe.T_w2c(viewpoint))
        cl = {c[0]: c for c in self.frame_cls}
        t1s = {pi for pi, _ in self.frame_tiers[0]}
        t2s = {pi for pi, _ in self.frame_tiers[1]}
        pl, pl_e1c = [], []
        for pi, q in enumerate(p["planes"]):
            n_w, d_w = plane_c2w(q.n.numpy(), q.d, T)
            c = cl.get(pi)
            inl = bool(c[4]) if c else False
            aid = int(c[1]) if c else None
            pl.append(dict(n_c=[float(v) for v in q.n.numpy()], d_c=float(q.d), n_px=int(q.n_px), z=plane_z(q),
                           anchor=aid, e=None if c is None else float(c[2]), w=None if c is None else float(c[3]),
                           inlier=inl, tier=1 if pi in t1s else (2 if pi in t2s else None)))
            pl_e1c.append(dict(n_w=[float(v) for v in n_w], d_w=float(d_w), n_px=int(q.n_px),
                               anchor=aid if inl else None, e=None if c is None else float(c[2]),
                               w=None if c is None else float(c[3]), inlier=inl))
        act = [c for c in self.frame_cls if c[4] and self.map.anchors[c[1]].active] if self.frame_cls else []
        base = dict(frame=int(idx), init_source=p["init_source"], n_planes=len(p["planes"]), n_pairs=len(p["pairs"]),
                    n_active_pairs=len(act), sigma_min=p["sigma_min"], iters=getattr(fe, "_track_iters", None),
                    ms_extract=p["ms_extract"], ms_match=p["ms_match"], ms_solve=p["ms_solve"],
                    n_anchors=len(self.map.anchors), n_anchors_active=self.map.n_active(), is_kf=bool(is_kf),
                    kf_reason=reason, n_init_dropped=p.get("n_init_dropped", 0))
        self._w("pw_frames.jsonl", dict(base, planes=pl))
        self._w("e1c_frames.jsonl", dict(base, planes=pl_e1c))

    # ------------------------------------------------------------------ keyframes
    def on_keyframe(self, fe, idx, viewpoint):
        T = _np(fe.T_w2c(viewpoint))
        planes = self.pinfo["planes"]
        cls, created = self.map.update_at_keyframe_pw(idx, planes, T)
        self.n_kf += 1
        self.kf_since_pgo += 1
        t1, t2 = ([], [])
        if self.pgo_mode != "none":
            t1, t2 = self.map.closure_candidates(planes, cls, T, fe.current_window, idx)
            for pi, aid in t1:
                self.map._ev(type="closure_detect", tier=1, anchor=int(aid), plane=int(pi), kf=int(idx))
            for pi, aid in t2:
                self.map._ev(type="closure_detect", tier=2, anchor=int(aid), plane=int(pi), kf=int(idx))
            self.stats["tier1"] += len(t1)
            self.stats["tier2"] += len(t2)
        self.pending = dict(k=int(idx), planes=planes, t1=t1, t2=t2, created=created)
        self._log_anchors(idx, "kf")
        self._flush_events()

    def on_sync(self, fe):
        self.map.refresh(self.kf_poses(fe))

    def maybe_pgo(self, fe):
        pend, self.pending = self.pending, None
        if self.pgo_mode == "none" or pend is None:
            return
        forced = self.force_every > 0 and self.n_kf % self.force_every == 0
        have = bool(pend["t1"] or pend["t2"])
        if not forced and not (have and self.kf_since_pgo >= self.pgo_gap):
            return
        if self.pgo_mode == "sgs":
            self._pgo_sgs(fe, pend, forced)
        else:
            self._pgo_ours(fe, pend, forced)

    # ------------------------------------------------------------------ PGO common
    def _odometry(self, fe, kfs, P):
        newest = set(kfs[-3:])
        odo = []
        for i, j in zip(kfs[:-1], kfs[1:]):
            if (i, j) in self.frozen:
                odo.append((i, j, self.frozen[(i, j)]))
                continue
            M = P[j] @ PG.inv(P[i])
            if j not in newest:
                self.frozen[(i, j)] = M
            odo.append((i, j, M))
        return odo

    def _apply(self, fe, k, res, mode, t0, pend):
        P = res["poses"]
        moved = {kf: P[kf] for kf in P}
        for kf, Tn in moved.items():
            Tt = torch.from_numpy(Tn).float().to(fe.cameras[kf].R.device)
            fe.cameras[kf].update_RT(Tt[:3, :3].clone(), Tt[:3, 3].clone())
        msg = {int(kf): (torch.from_numpy(Tn[:3, :3]).float(), torch.from_numpy(Tn[:3, 3]).float()) for kf, Tn in moved.items()}
        fe.backend_queue.put(["pgo", msg])
        fe.requested_pgo += 1
        self.kf_since_pgo = 0
        self.stats["pgo_applied"] += 1

    def _pgo_log(self, idx, mode, res, ms, forced, extra=None):
        row = dict(frame=int(idx), mode=mode, forced=bool(forced), status=res["status"], reason=res.get("reason"),
                   n_nodes=res.get("n_nodes"), n_odo=res.get("n_odo"), n_plane=res.get("n_plane"),
                   n_closure=res.get("n_closure"), dropped=len(res.get("dropped", [])),
                   max_kf_shift_m=res.get("max_kf_shift_m"), max_kf_rot_deg=res.get("max_kf_rot_deg"),
                   k_shift_m=res.get("k_shift_m"), k_rot_deg=res.get("k_rot_deg"), ms=ms)
        if extra:
            row.update(extra)
        self._w("pgo.jsonl", row)
        self.map._ev(type="pgo", kf=int(idx), mode=mode, status=res["status"])

    # ------------------------------------------------------------------ PGO ours
    def _pgo_ours(self, fe, pend, forced):
        t0 = time.perf_counter()
        k = pend["k"]
        kfs = [q for q in fe.kf_indices if q in fe.cameras]
        P = self.kf_poses(fe)
        self.map.refresh(P)
        odo = self._odometry(fe, kfs, P)
        planes, obs = {}, []
        for aid, a in self.map.anchors.items():
            if not a.active:
                continue
            ob = [o for o in a.kf_obs if o["kf"] in P]
            if not ob:
                continue
            planes[aid] = (a.n_w.numpy().copy(), float(a.d_w))
            for o in ob:
                obs.append((aid, o["kf"], np.asarray(o["n_c"]), float(o["d_c"]), float(o["z"]), "kf_obs"))
        closure_idx = [i for i, ob in enumerate(obs) if ob[1] == k and ob[0] in {aid for _, aid in pend["t1"]}]
        for pi, aid in pend["t2"]:
            if aid in planes:
                q = pend["planes"][pi]
                obs.append((aid, k, q.n.numpy().copy(), float(q.d), plane_z(q), f"t2:{pi}"))
                closure_idx.append(len(obs) - 1)
        self._log_anchors(k, "pre_pgo")
        if self.pgo_robust == "irls_factor":   # PLAN MONOGS-PW2 §2.4
            res = PG.solve_ours_irls(kfs, P, odo, planes, obs, closure_idx, self.noise_pgo, self.s_odo_t, self.s_odo_r, k,
                                     self.huber_k, self.chi2, self.max_dt, self.max_dr, self.max_nfev)
        else:
            res = PG.solve_ours(kfs, P, odo, planes, obs, closure_idx, self.noise, self.s_odo_t, self.s_odo_r, k,
                                self.huber_k, self.chi2, self.max_dt, self.max_dr, self.max_nfev)
        if self.pgo_robust == "irls_factor":
            dset = set(res.get("dropped", []))
            self._w("pgo_poses.jsonl", dict(
                frame=int(k), status=res["status"], solve_s=res.get("solve_s"),
                before={int(q): P[q].tolist() for q in kfs},
                after={int(q): res["poses"][q].tolist() for q in kfs} if res.get("poses") else None,
                closures=[dict(anchor=int(obs[i][0]), kf=int(obs[i][1]), tier=2 if obs[i][5].startswith("t2:") else 1,
                               plane=int(obs[i][5][3:]) if obs[i][5].startswith("t2:") else None, dropped=i in dset,
                               e2=res.get("closure_e2", {}).get(i), irls_w=res.get("closure_w", {}).get(i))
                          for i in closure_idx]))
        dropped = set(res.get("dropped", []))
        self.stats["closure_dropped"] += len(dropped)
        n_acc = 0
        if res["status"] == "applied":
            # anchors: landmark written back into the creating measurement (attach) or the value itself (nolife)
            Pn = res["poses"]
            for aid, (n_w, d_w) in res["planes"].items():
                a = self.map.anchors[aid]
                if self.map.anchor_mode == "attach":
                    n_c, d_c = plane_w2c(n_w, d_w, Pn[a.creator_kf])
                    a.obs_c = (np.asarray(n_c), float(d_c))
                else:
                    a.n_w, a.d_w = torch.from_numpy(np.asarray(n_w)), float(d_w)
            # tier 1 dropped: remove those observations of keyframe k
            for i in dropped:
                aid, kf, m, d, z, tag = obs[i]
                if tag == "kf_obs":
                    a = self.map.anchors.get(aid)
                    if a is not None:
                        a.kf_obs = [o for o in a.kf_obs if not (o["kf"] == kf and np.allclose(o["n_c"], m))]
            # tier 2 accepted: the candidate made from that plane at k becomes an alias of the old anchor
            for i, ob in enumerate(obs):
                if i in dropped or not ob[5].startswith("t2:"):
                    continue
                pi = int(ob[5][3:])
                aid = ob[0]
                cand = pend["created"].get(pi)
                q = pend["planes"][pi]
                if cand is not None and cand in self.map.anchors and cand != aid:
                    self.map.make_alias(cand, aid, "closure", k)
                elif aid in self.map.anchors:
                    self.map._add_obs(self.map.anchors[aid], k, q)
                n_acc += 1
            n_acc += sum(1 for i in closure_idx if i not in dropped and obs[i][5] == "kf_obs")
            self.stats["closure_accepted"] += n_acc
            self._apply(fe, k, res, "ours", t0, pend)
            self.map.refresh(Pn)
            self.map.merge_after_pgo(self.merge_deg, self.merge_m, k)
        else:
            self.stats["pgo_aborted"] += 1
        ms = 1000 * (time.perf_counter() - t0)
        self._pgo_log(k, "ours", res, ms, forced, dict(closure_accepted=n_acc, tier1=len(pend["t1"]), tier2=len(pend["t2"])))
        self._log_anchors(k, "post_pgo")
        self._flush_events()
        Log(f"[PW] PGO ours at kf {k}: {res['status']} nodes {res.get('n_nodes')} plane factors {res.get('n_plane')} "
            f"closure {res.get('n_closure')} dropped {len(dropped)} max shift "
            f"{100 * (res.get('max_kf_shift_m') or 0):.2f} cm, {ms:.0f} ms", tag="PW")

    # ------------------------------------------------------------------ PGO sgs (StructureGS-style closure)
    def _pgo_sgs(self, fe, pend, forced):
        t0 = time.perf_counter()
        k = pend["k"]
        kfs = [q for q in fe.kf_indices if q in fe.cameras]
        P = self.kf_poses(fe)
        meas_k = {}
        for pi, aid in pend["t1"] + pend["t2"]:
            q = pend["planes"][pi]
            meas_k.setdefault(aid, (q.n.numpy().copy(), float(q.d), q.pts_c))
        aids = list(meas_k)
        ok_pair = any(np.degrees(np.arccos(np.clip(abs(meas_k[a][0] @ meas_k[b][0]), -1, 1))) >= 30
                      for i, a in enumerate(aids) for b in aids[i + 1:])
        res = dict(status="skipped", reason="no 2 non-parallel anchors", n_nodes=len(kfs))
        if ok_pair or forced:
            win = set(fe.current_window)
            best = None
            for j in sorted([q for q in kfs if q not in win and q != k], key=lambda q: abs(k - q)):
                have = {}
                for a in aids:
                    an = self.map.anchors.get(a)
                    if an is None:
                        continue
                    o = next((o for o in an.kf_obs if o["kf"] == j), None)
                    if o is not None:
                        have[a] = o
                ks = list(have)
                if any(np.degrees(np.arccos(np.clip(abs(have[a]["n_c"] @ have[b]["n_c"]), -1, 1))) >= 30
                       for i, a in enumerate(ks) for b in ks[i + 1:]):
                    best = (j, have)
                    break
            if best is not None:
                j, have = best
                pk = [(meas_k[a][0], meas_k[a][1]) for a in have]
                pj = [(np.asarray(have[a]["n_c"]), float(have[a]["d_c"])) for a in have]
                T_jk = PG.relpose_from_planes(pk, pj)
                src = np.concatenate([meas_k[a][2] for a in have])
                dst = np.concatenate([have[a]["pts_c"] for a in have])
                nrm = np.concatenate([np.repeat(np.asarray(have[a]["n_c"])[None], len(have[a]["pts_c"]), 0) for a in have])
                try:
                    T_jk, fit, rmse = PG.icp_refine(T_jk, src, dst, nrm)
                except Exception as e:  # noqa: BLE001
                    fit, rmse = None, None
                    Log(f"[PW] sgs ICP failed: {e}", tag="PW")
                self.sgs_edges.append((j, k, T_jk))
                odo = self._odometry(fe, kfs, P)
                res = PG.solve_sgs(kfs, P, odo, self.sgs_edges, self.s_odo_t, self.s_odo_r, k, self.max_dt, self.max_dr,
                                   self.max_nfev)
                res["j"] = int(j)
                if res["status"] == "applied":
                    self._apply(fe, k, res, "sgs", t0, pend)
                    self.map.refresh(res["poses"])
                    self.map.merge_after_pgo(self.merge_deg, self.merge_m, k)
                    self.stats["closure_accepted"] += 1
                else:
                    self.sgs_edges.pop()
                    self.stats["pgo_aborted"] += 1
            else:
                res = dict(status="skipped", reason="no keyframe j", n_nodes=len(kfs))
        ms = 1000 * (time.perf_counter() - t0)
        self._pgo_log(k, "sgs", res, ms, forced, dict(j=res.get("j"), tier1=len(pend["t1"]), tier2=len(pend["t2"])))
        self._flush_events()

    def close(self):
        for f in self.f.values():
            if f is not None:
                f.close()
