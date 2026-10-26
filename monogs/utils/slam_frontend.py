import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import SE3_exp, update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
# PLAN MONOGS-E1C §5: E1c (plane anchors + direction-decoupled pose step); only used when PlaneTracking.enable
import json
from utils.obs_decouple import DecoupledPoseStep
from utils.plane_anchor import PlaneAnchorMap, PlaneExtractor, plane_c2w


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

        # PLAN MONOGS-PW §2.6 / §3.3-3.4: stiffness log, corrected trajectory (write only, every method)
        st = self.config.get("Stiffness", {}) or {}
        self.stiff_every = int(self.config.get("Repro", {}).get("stiffness_every", 0) or 0)
        self.stiff_at_kf = bool(st.get("at_keyframes", False))
        self.stiff_dt = float(st.get("delta_t_m", 0.01))
        self.stiff_dr = float(st.get("delta_r_deg", 0.5))
        self._f_stiff = None
        self.ref_rel = {}
        self.pw = None
        self.requested_pgo = 0

        # PLAN MONOGS-E1C §5.2 (2): E1c objects; nothing is created when PlaneTracking.enable is false
        pt = self.config.get("PlaneTracking", {}) or {}
        self.pt_cfg = pt
        self.pt_enable = bool(pt.get("enable", False))
        self.plane_extractor = self.plane_anchors = None
        self.plane_ms = []
        self._pinfo = None
        self._f_e1c = self._f_anc = None
        if self.pt_enable:
            self.pt_use_init = bool(pt.get("use_init", True))
            self.pt_use_loss = bool(pt.get("use_loss", True))
            self.pt_lambda = float(pt.get("lambda_plane", 1.0))
            self.pt_decouple = bool(pt.get("decouple", False))
            self.plane_K = np.array([[self.dataset.fx, 0.0, self.dataset.cx], [0.0, self.dataset.fy, self.dataset.cy],
                                     [0.0, 0.0, 1.0]])
            self.plane_extractor = PlaneExtractor(pt)
            self.plane_anchors = PlaneAnchorMap(pt, self.plane_K, self.dataset.height, self.dataset.width)
            Log(f"[E1c] enabled: lambda_plane={self.pt_lambda} decouple={self.pt_decouple} "
                f"use_init={self.pt_use_init} use_loss={self.pt_use_loss}", tag="E1c")
            if bool(pt.get("pw", False)):  # PLAN MONOGS-PW: plane world of the methods full*, sgs
                from utils.pw_frontend import PWFront
                self.pw = PWFront(self)

    # ------------------------------------------------------------------ PLAN MONOGS-E1C §5.2 (1): pose forms
    @staticmethod
    def T_w2c(viewpoint):
        T = torch.eye(4, device=viewpoint.R.device, dtype=torch.float32)
        T[0:3, 0:3] = viewpoint.R
        T[0:3, 3] = viewpoint.T
        return T

    @staticmethod
    def set_T_w2c(viewpoint, T):
        viewpoint.update_RT(T[0:3, 0:3].clone().float(), T[0:3, 3].clone().float())

    def _e1c_log_frame(self, idx, viewpoint, pinfo, iters):
        """e1c_frames.jsonl (write only): per frame and per measured plane, (n, d) in the world with the final pose."""
        if self.save_dir is None or pinfo is None:
            return
        if self._f_e1c is None:
            self._f_e1c = open(f"{self.save_dir}/e1c_frames.jsonl", "w")
        T = self.T_w2c(viewpoint).detach().cpu().double().numpy()
        match = dict(pinfo["pairs"])
        pl = []
        for pi, p in enumerate(pinfo["planes"]):
            n_w, d_w = plane_c2w(p.n.numpy(), p.d, T)
            pl.append(dict(n_w=[float(x) for x in n_w], d_w=float(d_w), n_px=int(p.n_px),
                           anchor=None if pi not in match else int(match[pi])))
        act = self.plane_anchors.active_pairs(pinfo["pairs"]) if pinfo["pairs"] else []
        row = dict(frame=int(idx), init_source=pinfo["init_source"], n_planes=len(pinfo["planes"]),
                   n_pairs=len(pinfo["pairs"]), n_active_pairs=len(act), sigma_min=pinfo["sigma_min"], iters=iters,
                   ms_extract=pinfo["ms_extract"], ms_match=pinfo["ms_match"], ms_solve=pinfo["ms_solve"],
                   n_anchors=len(self.plane_anchors.anchors), n_anchors_active=self.plane_anchors.n_active(), planes=pl)
        self._f_e1c.write(json.dumps(row) + "\n")
        self._f_e1c.flush()

    def _log_stiffness(self, idx, viewpoint, is_kf):
        """PLAN MONOGS-PW §2.6: c_render (6 axes) at the final pose (+ c_plane, share for PW methods); write only."""
        if self.save_dir is None:
            return
        from utils.render_stiffness import render_stiffness
        c_r, L0, ms = render_stiffness(viewpoint, self.gaussians, self.pipeline_params, self.background, self.config,
                                       self.stiff_dt, self.stiff_dr)
        row = dict(frame=int(idx), is_kf=bool(is_kf), c_render=c_r.tolist(), L0=L0, ms=ms,
                   iters=getattr(self, "_track_iters", None))
        if self.pw is not None:
            c_p = self.pw.plane_stiffness(self.T_w2c(viewpoint), self.stiff_dt, self.stiff_dr)
            if c_p is not None:
                kap = self.pw.kappa
                row["c_plane"] = c_p.tolist()
                row["share"] = [None if cp <= 0 else float(cp / (cp + kap * cr)) for cp, cr in zip(c_p, c_r)]
        if self._f_stiff is None:
            self._f_stiff = open(f"{self.save_dir}/stiffness.jsonl", "w")
        self._f_stiff.write(json.dumps(row) + "\n")
        self._f_stiff.flush()

    def _e1c_log_anchors(self, idx):
        """anchors.jsonl (write only), at every keyframe."""
        if self.save_dir is None:
            return
        if self._f_anc is None:
            self._f_anc = open(f"{self.save_dir}/anchors.jsonl", "w")
        a = [dict(id=int(x.id), n_w=[float(v) for v in x.n_w.numpy()], d_w=float(x.d_w), n_obs_kf=int(x.n_obs_kf),
                  active=bool(x.active), created_kf=int(x.created_kf)) for x in self.plane_anchors.anchors.values()]
        self._f_anc.write(json.dumps(dict(frame=int(idx), anchors=a)) + "\n")
        self._f_anc.flush()

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        if self.pw is not None:  # PLAN MONOGS-PW: anchors of frame 0 (GT pose), active at once
            self.pw.init_first(self, cur_frame_idx, viewpoint)
        elif self.pt_enable:  # PLAN MONOGS-E1C §5.2 (3): anchors of frame 0 (GT pose), active at once
            planes = self.plane_extractor(viewpoint.depth, self.plane_K)
            self.plane_anchors.update_at_keyframe(cur_frame_idx, planes, [], self.T_w2c(viewpoint))
            self._e1c_log_frame(cur_frame_idx, viewpoint, dict(planes=planes, pairs=[], init_source="gt",
                                                               sigma_min=float("nan"), ms_extract=0.0, ms_match=0.0,
                                                               ms_solve=0.0), 0)
            self._e1c_log_anchors(cur_frame_idx)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        # PLAN MONOGS-E1C §5.2 (4): planes -> match -> closed-form init -> loss context (E1c only)
        ctx, dec = None, None
        self._pinfo = None
        if self.pw is not None:  # PLAN MONOGS-PW §2.2 A-B: nn / e1c match, robust closed-form init, loss context
            ctx, dec = self.pw.before_tracking(self, viewpoint)
        elif self.pt_enable:
            t0 = time.perf_counter()
            planes = self.plane_extractor(viewpoint.depth, self.plane_K)
            t1 = time.perf_counter()
            T_init = self.T_w2c(viewpoint)
            pairs = self.plane_anchors.match(planes, T_init)
            t2 = time.perf_counter()
            T_plane = self.plane_anchors.solve_pose(planes, pairs, T_init)
            t3 = time.perf_counter()
            init_source = "rejected" if self.plane_anchors.last_solve_status == "rejected" else "prev"
            if self.pt_use_init and T_plane is not None:
                self.set_T_w2c(viewpoint, T_plane)
                init_source = "plane"
            if self.pt_use_loss:
                ctx = self.plane_anchors.make_loss_ctx(planes, pairs, viewpoint.R.device, torch.float32)
            self._pinfo = dict(planes=planes, pairs=pairs, init_source=init_source,
                               sigma_min=float(self.plane_anchors.last_sigma_min), ms_extract=1000 * (t1 - t0),
                               ms_match=1000 * (t2 - t1), ms_solve=1000 * (t3 - t2))
            self.plane_ms.append(1000 * (t3 - t0))
            if self.pt_decouple:  # §5.2 (5): pose deltas leave MonoGS's Adam; DecoupledPoseStep with MonoGS lrs
                m_np = ctx["m"].double().cpu().numpy() if ctx is not None else np.zeros((0, 3))
                dec = DecoupledPoseStep(m_np, float(self.pt_cfg.get("obs_threshold", self.plane_anchors.min_sv)),
                                        self.config["Training"]["lr"]["cam_trans_delta"],
                                        self.config["Training"]["lr"]["cam_rot_delta"],
                                        device=viewpoint.R.device, dtype=torch.float32)

        opt_params = []
        if dec is None:
          opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
          )
          opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
          )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            g_p_t = g_p_r = None
            if ctx is not None:  # PLAN MONOGS-E1C §5.2 (5): plane term on SE3_exp(tau) @ T_w2c (same tau as render)
                tau = torch.cat([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], 0)
                if self.pw is not None:  # PLAN MONOGS-PW §2.2 C / §2.6: robust weights, E1c or information scale
                    plane_loss = self.pw.loss(SE3_exp(tau) @ self.T_w2c(viewpoint), ctx)
                else:
                    plane_loss = self.pt_lambda * self.plane_anchors.plane_loss_ctx(SE3_exp(tau) @ self.T_w2c(viewpoint), ctx)
                if dec is not None:
                    g_p_t, g_p_r = torch.autograd.grad(plane_loss,
                                                       [viewpoint.cam_trans_delta, viewpoint.cam_rot_delta])
                else:
                    loss_tracking = loss_tracking + plane_loss   # lambda_p already inside plane_loss
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                if dec is not None:  # merge render / plane gradients per observable direction
                    d_t, d_r = dec.step(viewpoint.cam_trans_delta.grad, viewpoint.cam_rot_delta.grad, g_p_t, g_p_r)
                    viewpoint.cam_trans_delta.copy_(d_t)
                    viewpoint.cam_rot_delta.copy_(d_r)
                    viewpoint.cam_trans_delta.grad = None
                    viewpoint.cam_rot_delta.grad = None
                converged = update_pose(viewpoint)
                self._track_iters = tracking_itr + 1

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        # PLAN MONOGS-E1C §3.1-3.3 (log only)
        Log(f"[Repro] proc=frontend single_thread={self.single_thread} "
            f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32} tf32_cudnn={torch.backends.cudnn.allow_tf32}",
            tag="Repro")
        self.track_ms = []

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and (self.requested_keyframe > 0 or self.requested_pgo > 0):
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                t_trk = time.perf_counter()
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                torch.cuda.synchronize()
                self.track_ms.append(1000 * (time.perf_counter() - t_trk))  # log only (excludes the 3 fps sleep)
                if self.pt_enable and self.pw is None:  # PLAN MONOGS-E1C §5.2 (7), write only
                    self._e1c_log_frame(cur_frame_idx, viewpoint, self._pinfo, getattr(self, "_track_iters", None))
                # PLAN MONOGS-PW §3.4: pose relative to the latest keyframe (corrected trajectory, write only)
                _rk = self.current_window[0]
                self.ref_rel[cur_frame_idx] = (int(_rk), (self.T_w2c(viewpoint).double()
                                                          @ torch.linalg.inv(self.T_w2c(self.cameras[_rk]).double())).cpu().numpy())
                if self.pw is not None:
                    self.pw.after_tracking(self, cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                kf_reason = "mono" if create_kf else None
                if (self.pw is not None and not create_kf and check_time
                        and len(self.current_window) >= self.window_size):
                    kf_reason = self.pw.extra_kf_reason(self, cur_frame_idx)   # PLAN MONOGS-PW §2.4: T1 / T2 (OR)
                    create_kf = kf_reason is not None
                if self.pw is not None:
                    self.pw.log_frame(self, cur_frame_idx, viewpoint, create_kf, kf_reason)
                if (self.stiff_every > 0 and cur_frame_idx % self.stiff_every == 0) or (self.stiff_at_kf and create_kf):
                    self._log_stiffness(cur_frame_idx, viewpoint, create_kf)   # PLAN MONOGS-PW §2.6, write only
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    if self.pw is not None:
                        self.pw.on_keyframe(self, cur_frame_idx, viewpoint)   # PLAN MONOGS-PW §2.3, §2.5
                    elif self.pt_enable and self._pinfo is not None:
                        # PLAN MONOGS-E1C §5.2 (6): planes of this frame (not re-extracted), re-matched with the
                        # tracked pose, then the anchor update (pose before MonoGS's local BA, as E1c on 2DGS-SLAM)
                        T_kf = self.T_w2c(viewpoint)
                        pairs_kf = self.plane_anchors.match(self._pinfo["planes"], T_kf)
                        self.plane_anchors.update_at_keyframe(cur_frame_idx, self._pinfo["planes"], pairs_kf, T_kf)
                        self._e1c_log_anchors(cur_frame_idx)
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                    if self.pw is not None:
                        self.pw.on_sync(self)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1
                    if self.pw is not None:   # PLAN MONOGS-PW §2.5: closure + PGO after the keyframe's local BA
                        self.pw.on_sync(self)
                        self.pw.maybe_pgo(self)

                elif data[0] == "pgo":
                    self.sync_backend(data)
                    self.requested_pgo -= 1
                    if self.pw is not None:
                        self.pw.on_sync(self)

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
