"""PLAN E1 §7 per-run outputs (write-only; never changes what SLAM computes).

frame_log.csv, timing.csv, renders/<frame>_*.png (+ .npz), traj_est_all.txt, traj_gt_all.txt (TUM),
ape_per_frame.csv, vram.txt, map_xyz.npy, e1_metrics.json (ATE-all, ATE-kf, RPE, timing, plane stats).
"""
import csv
import json
import math
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gaussian_renderer import render
from utils.eval_utils import evaluate_evo_e1
from utils.logging_utils import Log

FRAME_FIELDS = ["frame_id", "is_kf", "iters_done", "loss_render_final", "loss_plane_final", "n_planes",
                "n_matched", "n_matched_active", "sigma_min", "init_source", "n_anchors", "n_anchors_active",
                "T_init", "pairs", "active_ids"]                     # last three: PLAN E1c §5
TIMING_FIELDS = ["frame_id", "ms_track", "ms_plane_extract", "ms_match", "ms_solve", "obs_basis_ms", "edge_ms",
                 "ms_pw"]                                              # last: PLAN A2G-P1 plane world
RENDER_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
EXTRA_RENDER_FRAMES = (260, 600)                                      # PLAN E1c §5: one frame in S3, one in S5
DEC_FIELDS = (["frame_id"] + [f"sig_t{i}" for i in range(3)] + [f"sig_r{i}" for i in range(3)] + ["n_obs_t", "n_obs_r"]
              + [f"Vt{i}" for i in range(9)] + [f"Vr{i}" for i in range(9)] + ["gp_removed_norm"]
              + [f"sum_abs_dxi_t{i}" for i in range(3)] + [f"sum_abs_dxi_r{i}" for i in range(3)]
              + [f"mean_abs_gr_xi_t{i}" for i in range(3)])


def _anchor_table(am):
    return [dict(id=a.id, n_w=[float(x) for x in a.n_w.numpy()], d_w=float(a.d_w), n_obs_kf=int(a.n_obs_kf),
                 active=bool(a.active), created_kf=int(a.created_kf), n_samples=int(a.samples_w.shape[0]))
            for a in sorted(am.anchors.values(), key=lambda a: a.id)]


def _so3_project(P):
    """Nearest rotation (SVD) for poses kept in float32: evo rejects slightly non-orthonormal matrices."""
    if not np.all(np.isfinite(P)):
        return P
    U, _, Vt = np.linalg.svd(P[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    Q = P.copy()
    Q[:3, :3] = R
    return Q


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return "" if math.isnan(x) else f"{x:.6g}"
    return x


class E1Logger:
    def __init__(self, fe):
        self.fe = fe
        self.dir = fe.save_dir
        os.makedirs(os.path.join(self.dir, "renders"), exist_ok=True)
        self._f_frame = open(os.path.join(self.dir, "frame_log.csv"), "w", newline="")
        self._w_frame = csv.DictWriter(self._f_frame, fieldnames=FRAME_FIELDS)
        self._w_frame.writeheader()
        self._f_time = open(os.path.join(self.dir, "timing.csv"), "w", newline="")
        self._w_time = csv.DictWriter(self._f_time, fieldnames=TIMING_FIELDS)
        self._w_time.writeheader()
        # 4 fixed render frames at 20/40/60/80 % of the dataset length, snapped to the tracked grid
        n = len(fe.dataset)
        b, s = fe.frame_begin, fe.frame_step
        end = n if fe.frame_end_exclusive is None else min(n, fe.frame_end_exclusive)
        tg = [b + s * int(round((p * n - b) / s)) for p in RENDER_FRACTIONS]
        tg += [f for f in EXTRA_RENDER_FRAMES if (f - b) % s == 0]
        self.render_frames = sorted({f for f in tg if b < f < end})
        self._f_planes = self._f_anchors = self._f_dec = self._w_dec = None
        if fe.pt_enable:
            self._f_planes = open(os.path.join(self.dir, "planes.jsonl"), "w")
            self._f_anchors = open(os.path.join(self.dir, "anchors_kf.jsonl"), "w")
        self.rows = []
        self.n_rejected = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        Log(f"E1 logging -> {self.dir}; render frames {self.render_frames}", tag="E1")

    # ------------------------------------------------------------------ per frame
    def end_frame(self, fid, is_kf, track_stats, plane_info, timing):
        ts = track_stats or {}
        row = dict(frame_id=int(fid), is_kf=int(bool(is_kf)), iters_done=ts.get("iters_done"),
                   loss_render_final=ts.get("loss_render_final"), loss_plane_final=ts.get("loss_plane_final"))
        trow = dict(frame_id=int(fid), ms_track=timing.get("ms_track"))
        if plane_info is not None:
            am = self.fe.plane_anchors
            pw = getattr(self.fe, "pw", None)
            pairs = plane_info["pairs"]
            if pw is not None:  # PLAN A2G-P1: counts from the plane world (anchors reduced to primaries)
                row.update(n_planes=len(plane_info["planes"]), n_matched=len(pairs),
                           n_matched_active=len(plane_info.get("active_ids", [])), sigma_min=plane_info["sigma_min"],
                           init_source=plane_info["init_source"], n_anchors=pw.n_primary(),
                           n_anchors_active=pw.n_active())
            else:
                row.update(n_planes=len(plane_info["planes"]), n_matched=len(pairs),
                           n_matched_active=len(am.active_pairs(pairs)), sigma_min=plane_info["sigma_min"],
                           init_source=plane_info["init_source"], n_anchors=len(am.anchors),
                           n_anchors_active=am.n_active())
            trow.update(ms_plane_extract=plane_info["ms_extract"], ms_match=plane_info["ms_match"],
                        ms_solve=plane_info["ms_solve"], ms_pw=plane_info.get("ms_pw"))
            if plane_info["init_source"] == "rejected":
                self.n_rejected += 1
            T = plane_info.get("T_init")
            if T is not None:
                row["T_init"] = ";".join(f"{x:.7g}" for x in T.detach().cpu().double().numpy()[:3, :].reshape(-1))
            row["pairs"] = ";".join(f"{pi}:{aid}" for pi, aid in pairs)
            row["active_ids"] = ";".join(str(a) for a in plane_info.get("active_ids", []))
            self._f_planes.write(json.dumps(dict(frame_id=int(fid), planes=[
                dict(n=[float(x) for x in p.n.numpy()], d=float(p.d), n_px=int(p.n_px), mse=float(p.mse))
                for p in plane_info["planes"]])) + "\n")
            self._f_planes.flush()
            if is_kf:  # anchor table after this keyframe's update
                tab = pw.anchors_table() if pw is not None else _anchor_table(am)
                self._f_anchors.write(json.dumps(dict(frame_id=int(fid), anchors=tab)) + "\n")
                self._f_anchors.flush()
        dec_row = getattr(self.fe, "_dec_row", None) if track_stats is not None else None
        if dec_row is not None:
            if self._w_dec is None:
                self._f_dec = open(os.path.join(self.dir, "dec_log.csv"), "w", newline="")
                self._w_dec = csv.DictWriter(self._f_dec, fieldnames=DEC_FIELDS)
                self._w_dec.writeheader()
            d = {"frame_id": int(fid), "n_obs_t": dec_row["n_obs_t"], "n_obs_r": dec_row["n_obs_r"],
                 "gp_removed_norm": dec_row["gp_removed_norm"]}
            for key, n_ in (("sig_t", 3), ("sig_r", 3), ("Vt", 9), ("Vr", 9), ("sum_abs_dxi_t", 3),
                            ("sum_abs_dxi_r", 3), ("mean_abs_gr_xi_t", 3)):
                for i in range(n_):
                    d[f"{key}{i}"] = float(dec_row[key][i])
            self._w_dec.writerow({k: _fmt(v) if isinstance(v, float) else v for k, v in d.items()})
            self._f_dec.flush()
            trow["obs_basis_ms"] = dec_row["obs_basis_ms"]
        self.rows.append(row)
        self._w_frame.writerow({k: _fmt(row.get(k)) for k in FRAME_FIELDS})
        self._w_time.writerow({k: _fmt(trow.get(k)) for k in TIMING_FIELDS})
        self._f_frame.flush()
        self._f_time.flush()

    @torch.no_grad()
    def maybe_render(self, cam, fid, gt_img, depth, _render_pkg):
        """Render the current full map from the just-tracked pose (PLAN §7)."""
        if int(fid) not in self.render_frames:
            return
        fe = self.fe
        pkg = render(cam, pc=fe.gaussians, render_option="all", bg_color=fe.bg_color)
        img = torch.clamp(torch.exp(cam.exposure_a) * pkg["render"] + cam.exposure_b, 0, 1)
        d_ren = pkg[fe.depth_type].squeeze(0).float().cpu().numpy()
        d_gt = depth.float().cpu().numpy()
        rgb_gt = (gt_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rgb_ren = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        err = np.where(d_gt > 0, np.abs(d_gt - d_ren), 0.0)
        p = os.path.join(self.dir, "renders", f"{int(fid):05d}")
        cv2.imwrite(p + "_rgb_gt.png", cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2BGR))
        cv2.imwrite(p + "_rgb_ren.png", cv2.cvtColor(rgb_ren, cv2.COLOR_RGB2BGR))
        cv2.imwrite(p + "_depth_gt.png", np.clip(d_gt * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(p + "_depth_ren.png", np.clip(d_ren * 1000, 0, 65535).astype(np.uint16))
        err_u8 = (np.clip(err / 0.20, 0, 1) * 255).astype(np.uint8)          # fixed 0–20 cm scale
        cv2.imwrite(p + "_depth_err.png", cv2.applyColorMap(err_u8, cv2.COLORMAP_INFERNO))
        np.savez_compressed(p + ".npz", rgb_gt=rgb_gt, rgb_ren=rgb_ren,
                            depth_gt=d_gt.astype(np.float16), depth_ren=d_ren.astype(np.float16))

    # ------------------------------------------------------------------ end of run
    def _timestamp(self, fid):
        paths = getattr(self.fe.dataset, "color_paths", None)
        if paths is None:
            return float(fid)
        return float(os.path.splitext(os.path.basename(paths[fid]))[0])

    @staticmethod
    def _write_tum(path, ts, poses):
        with open(path, "w") as f:
            for t, P in zip(ts, poses):
                q = Rotation.from_matrix(P[:3, :3]).as_quat()
                f.write(f"{t:.6f} {P[0, 3]:.6f} {P[1, 3]:.6f} {P[2, 3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    def finalize(self, ate_kf, ate_all, rend_out, slam_time_s):
        fe = self.fe
        self._f_frame.close()
        self._f_time.close()
        for f in (self._f_planes, self._f_anchors, self._f_dec):
            if f is not None:
                f.close()
        ids = sorted(fe.cameras.keys())
        est = [_so3_project(np.linalg.inv(fe.cameras[i].T.detach().cpu().double().numpy())) for i in ids]
        gt = [_so3_project(np.linalg.inv(fe.cameras[i].gt_pose.detach().cpu().double().numpy())) for i in ids]
        ts = [self._timestamp(i) for i in ids]
        self._write_tum(os.path.join(self.dir, "traj_est_all.txt"), ts, est)
        self._write_tum(os.path.join(self.dir, "traj_gt_all.txt"), ts, gt)
        has_nan = bool(any(not np.all(np.isfinite(P)) for P in est))
        m = dict(n_frames=len(ids), n_keyframes=len(fe.key_frame_ids), has_nan_pose=has_nan,
                 ate_kf_m=None if ate_kf is None else float(ate_kf),
                 ate_all_m_repo=None if ate_all is None else float(ate_all),
                 slam_time_s=float(slam_time_s), fps=len(ids) / float(slam_time_s),
                 n_gaussians=int(fe.gaussians.get_xyz.shape[0]))
        if rend_out is not None:
            m.update({k: float(v) for k, v in rend_out.items()})
        if not has_nan:
            ev = evaluate_evo_e1(gt, est)
            ape = ev.pop("ape_per_frame_m")
            m.update(ev)
            with open(os.path.join(self.dir, "ape_per_frame.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame_id", "timestamp", "ape_m"])
                for i, t, e in zip(ids, ts, ape):
                    w.writerow([i, f"{t:.6f}", f"{e:.6f}"])
            m["frac_frames_ape_gt_30cm"] = float(np.mean(ape > 0.30))
        # timing and plane diagnostics
        with open(os.path.join(self.dir, "timing.csv")) as f:
            tr = list(csv.DictReader(f))

        def mean_of(key):
            v = [float(r[key]) for r in tr if r.get(key) not in (None, "")]
            return float(np.mean(v)) if v else None

        m.update(ms_track_mean=mean_of("ms_track"), ms_plane_extract_mean=mean_of("ms_plane_extract"),
                 ms_match_mean=mean_of("ms_match"), ms_solve_mean=mean_of("ms_solve"))
        if fe.pt_enable:
            tracked = [r for r in self.rows if r.get("init_source") is not None][1:]   # skip first frame
            am = fe.plane_anchors
            m.update(
                pct_init_plane=100.0 * np.mean([r["init_source"] == "plane" for r in tracked]) if tracked else None,
                n_init_rejected=self.n_rejected,
                planes_per_frame=float(np.mean([r["n_planes"] for r in tracked])) if tracked else None,
                matched_active_per_frame=float(np.mean([r["n_matched_active"] for r in tracked])) if tracked else None,
                n_anchors_final=len(am.anchors), n_anchors_active_final=am.n_active(),
                n_strip_rejected=fe.plane_extractor.n_strip_rejected, n_back_facing=am.n_back_facing,
            )
            pw = getattr(fe, "pw", None)
            if pw is not None:  # PLAN A2G-P1 summary (counters, invariant violations, timing)
                m.update(n_anchors_final=pw.n_primary(), n_anchors_active_final=pw.n_active(),
                         ms_pw_mean=mean_of("ms_pw"), pw=pw.summary())
                pw.close()
        # memory and map
        if torch.cuda.is_available():
            gib = torch.cuda.max_memory_allocated() / 1024 ** 3
            m["torch_max_allocated_frontend_gib"] = gib
            with open(os.path.join(self.dir, "vram.txt"), "a") as f:
                f.write(f"torch_max_allocated_frontend_gib {gib:.3f}\n")
        np.save(os.path.join(self.dir, "map_xyz.npy"), fe.gaussians.get_xyz.detach().cpu().numpy().astype(np.float32))
        with open(os.path.join(self.dir, "e1_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        Log(f"E1 metrics: ATE-all {m.get('ate_rmse_m')} m, RPE d1 {m.get('rpe_trans_rmse_m_d1')} m", tag="E1")
