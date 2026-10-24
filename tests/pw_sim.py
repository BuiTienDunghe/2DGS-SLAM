"""PLAN A2G-P1 v2 §5 Q11/Q12: synthetic room driven through the PlaneWorld API (no frontend).

Scene: floor, walls A (x = -2), B (z = 5), C (x = +2); panel P4 parallel to A 4 cm in front of it (half of A);
panel P15 parallel to B 15 cm in front of it. Segments are built at the TRUE pose from per-cell statistics of
points on the true surface (4x4 rays per 20 px cell, TUM fr3 K): per-point noise sigma_z(z), a common per-cell depth
offset sigma_z(z), and a per-segment-per-frame plane error with the default floors (0.5 deg, 5 mm) - the error model
of §3.2(a). The tracker pose is T_opt = Exp(e(t)) T_true (+) jitter; T_pred = T_opt of the previous frame.
"""
import numpy as np
import torch

from utils.plane_anchor import PlaneMeas
from utils.pw import geometry as G
from utils.pw.model import CHI2_3
from utils.pw.world import PlaneWorld

H, W, CS = 480, 640, 20
K = np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])
KINV = np.linalg.inv(K)
MAXD = 4.0
N_FRAMES, KF_EVERY = 400, 10
SEG_ERR = 1.0          # 1: per-segment-per-frame plane error at the default floors; 0: plan text (depth noise only)
CFG = dict(sensor=dict(a=0.0, b=1.425e-3, sigma_theta0_deg=0.5, sigma_d0_m=0.005), pred_window=30,
           pred_init=dict(sigma_rho_m=0.01, sigma_theta_deg=0.5), offset_window=30,
           gamma0=dict(theta_deg=0.15, d_m=0.005), gamma_min_gap=5, grid_h=0.05, L_kf=10, m_diff=2,
           inview_min_cells=3, obs_threshold=0.3, gn_iters=3, debug_checks=False)


def sz(z):
    return 1.425e-3 * np.square(z)


class Rect:
    def __init__(self, name, n, p0, u, hu, v, hv):
        self.name, self.n, self.p0 = name, np.array(n, float), np.array(p0, float)
        self.u, self.v, self.hu, self.hv = np.array(u, float), np.array(v, float), hu, hv


SCENE = [
    Rect("floor", [0, -1, 0], [0, 1.2, 2.5], [1, 0, 0], 2.0, [0, 0, 1], 3.5),
    Rect("A", [1, 0, 0], [-2.0, -0.15, 2.5], [0, 0, 1], 3.5, [0, 1, 0], 1.35),
    Rect("B", [0, 0, -1], [0, -0.15, 5.0], [1, 0, 0], 2.0, [0, 1, 0], 1.35),
    Rect("C", [-1, 0, 0], [2.0, -0.15, 2.5], [0, 0, 1], 3.5, [0, 1, 0], 1.35),
    Rect("P4", [1, 0, 0], [-1.96, -0.15, 1.0], [0, 0, 1], 1.75, [0, 1, 0], 1.35),
    Rect("P15", [0, 0, -1], [0.3, 0.3, 4.85], [1, 0, 0], 0.9, [0, 1, 0], 0.9),
]
NAMES = [r.name for r in SCENE]


def trajectory():
    """Camera centres and yaw (rotation about y; yaw 0 looks along +z). Phases: 3 directions (0-100), 2 directions
    (100-200, wall A/P4 + floor), 3 directions (200-300, B/P15/C + floor), revisit of the start region (300-360),
    back to the start view (360-400)."""
    c, yaw = [], []
    for t in range(N_FRAMES):
        if t < 100:
            ci, ps = [0.0, 0.0, 1.5 * t / 100], 0.0
        elif t < 200:
            ci, ps = [-0.3 * (t - 100) / 100, 0.0, 1.5], -70.0 * min(1.0, (t - 100) / 40)
        elif t < 300:
            f = (t - 200) / 100
            ci, ps = [-0.3 + 0.3 * f, 0.0, 1.5 + 1.0 * f], -70.0 + 100.0 * min(1.0, (t - 200) / 50)
        elif t < 360:
            f = (t - 300) / 60
            ci, ps = [0.0, 0.0, 2.5 - 2.2 * f], 30.0 - 85.0 * min(1.0, f * 1.5)
        else:
            f = (t - 360) / 40
            ci, ps = [0.0, 0.0, 0.3], -55.0 + 55.0 * f
        c.append(np.array(ci))
        yaw.append(np.deg2rad(ps))
    return c, yaw


def pose_true(c, yaw):
    Rwc = G.so3_exp(np.array([0.0, yaw, 0.0]))
    T = np.eye(4)
    T[:3, :3] = Rwc.T
    T[:3, 3] = -Rwc.T @ c
    return T


def raycast(T, rays_c):
    """nearest rectangle hit for camera-frame rays (N,3) with z = 1; returns (plane index or -1, depth z)."""
    Rwc, cw = T[:3, :3].T, G.centre(T)
    d = rays_c @ Rwc.T
    best = np.full(len(rays_c), np.inf)
    idx = np.full(len(rays_c), -1)
    for i, r in enumerate(SCENE):
        if r.n @ (cw - r.p0) <= 0:
            continue
        den = d @ r.n
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (r.n @ (r.p0 - cw)) / den
        X = cw + d * lam[:, None]
        ok = (lam > 0) & (np.abs((X - r.p0) @ r.u) <= r.hu) & (np.abs((X - r.p0) @ r.v) <= r.hv) & (lam <= MAXD)
        upd = ok & (lam < best)
        best[upd], idx[upd] = lam[upd], i
    return idx, best


_CELL_RAYS = None


def cell_rays():
    global _CELL_RAYS
    if _CELL_RAYS is None:
        Hc, Wc = H // CS, W // CS
        centres, sub = [], []
        o = (np.arange(4) + 0.5) * CS / 4
        for i in range(Hc):
            for j in range(Wc):
                centres.append(KINV @ [j * CS + CS / 2, i * CS + CS / 2, 1.0])
                uu, vv = np.meshgrid(j * CS + o, i * CS + o)
                sub.append(np.stack([uu.ravel(), vv.ravel(), np.ones(16)], 1) @ KINV.T)
        _CELL_RAYS = (np.array(centres), np.array(sub))
    return _CELL_RAYS


def measure(T_true, rng, split=None):
    """PlaneMeas list (with cells) seen from the true pose; split = plane name to cut into two segments."""
    centres, sub = cell_rays()
    Hc, Wc = H // CS, W // CS
    pid, _ = raycast(T_true, centres)
    out, truth = [], []
    for i, r in enumerate(SCENE):
        cells = np.nonzero(pid == i)[0]
        if len(cells) < 8:
            continue
        m = np.zeros((Hc, Wc), bool)
        m.flat[cells] = True
        if not (m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]).any():
            continue
        groups = [cells]
        if split == r.name and len(cells) >= 24:
            cj = cells % Wc
            med = np.median(cj)
            groups = [cells[cj <= med], cells[cj > med]]
        n_c = T_true[:3, :3] @ r.n
        d_c = -float(n_c @ (T_true[:3, :3] @ r.p0 + T_true[:3, 3]))
        # per-segment-per-frame plane error (default floors, §3.2a)
        for g in groups:
            B = G.basis(n_c)
            dth = rng.normal(0, np.deg2rad(0.5), 2) * SEG_ERR
            dd = rng.normal(0, 0.005) * SEG_ERR
            rays = sub[g]                                    # (Ng, 16, 3)
            lam0 = -d_c / (rays @ n_c)
            X0 = rays * lam0[..., None]
            ctr = X0.reshape(-1, 3).mean(0)
            n_p = n_c + B @ dth
            n_p /= np.linalg.norm(n_p)
            d_p = -float(n_p @ (ctr + dd * n_c))
            lam = -d_p / (rays @ n_p)
            z = lam.copy()
            zc = z.mean(1, keepdims=True)
            z = z + rng.normal(0, 1, (len(g), 1)) * sz(zc) + rng.normal(0, 1, z.shape) * sz(z)
            X = rays * z[..., None]
            ij = np.stack([g // Wc, g % Wc], 1)
            cnt = np.full(len(g), 16.0)
            s1 = X.sum(1)
            s2 = np.einsum("gni,gnj->gij", X, X)
            mask = torch.zeros(Hc, Wc, dtype=torch.bool)
            mask.view(-1)[torch.as_tensor(g)] = True
            S1, S2 = s1.sum(0), s2.sum(0)
            p = S1 / cnt.sum()
            w, V = np.linalg.eigh(S2 / cnt.sum() - np.outer(p, p))
            nn = V[:, 0] if V[:, 0] @ p < 0 else -V[:, 0]
            pm = PlaneMeas(n=torch.from_numpy(nn.copy()), d=-float(nn @ p), mask_cells=mask, n_px=int(cnt.sum() * 25),
                           mse=float(max(w[0], 0)), pts_c=X.reshape(-1, 3)[::max(1, X.size // 1500)][:500])
            pm.cells = dict(ij=ij, cnt=cnt, s1=s1, s2=s2)
            out.append(pm)
            truth.append(i)
    return out, truth


def depth_image(T_true):
    v, u = np.mgrid[0:H, 0:W].astype(float)
    rays = np.stack([u.ravel(), v.ravel(), np.ones(H * W)], 1) @ KINV.T
    idx, z = raycast(T_true, rays)
    z[idx < 0] = 0.0
    return z.reshape(H, W).astype(np.float32)


def drift(kind, t, s, yaw, rng_state):
    """e(t) in xi = [rho, theta] (left perturbation of T_cw)."""
    e = np.zeros(6)
    if kind == "P1":                                     # linear in path length
        e[:3] = s * np.array([0.012, 0.0, 0.008])
        e[4] = s * np.deg2rad(0.3)
    elif kind == "P2":                                   # random walk
        rng_state["rw"] = rng_state.get("rw", np.zeros(6)) + np.concatenate(
            [rng_state["rng"].normal(0, 0.0015, 3), rng_state["rng"].normal(0, np.deg2rad(0.03), 3)])
        e = rng_state["rw"].copy()
    elif kind == "P3":                                   # rotation error proportional to signed yaw (+-20 deg swing)
        e[4] = 0.05 * yaw
        e[0] = 0.02 * np.sin(yaw)
    elif kind == "P4":                                   # no trend, jitter 3 mm / 0.2 deg
        e[:3] = rng_state["rng"].normal(0, 0.003, 3)
        e[3:] = rng_state["rng"].normal(0, np.deg2rad(0.2), 3)
    elif kind == "P5":                                   # 5 cm jump at frame 150
        if t >= 150:
            e[0] = 0.05
    elif kind == "P6":                                   # drift only along the unobserved direction (z) in 100-200
        e[2] = 0.06 * np.clip((t - 100) / 100, 0, 1)
    elif kind == "P7":                                   # 8 cm by the revisit of the start region
        f = np.clip(t / 300, 0, 1)
        e[0], e[2] = 0.06 * f, 0.053 * f
    elif kind == "P8":
        pass
    return e


def run_case(kind, seed, inject_wrong_at=None, dup_frames=(100, 250)):
    rng = np.random.default_rng(1000 * seed + sum(map(ord, kind)))
    st = dict(rng=np.random.default_rng(seed + 77))
    cs, yaws = trajectory()
    Tt = [pose_true(c, y) for c, y in zip(cs, yaws)]
    s_acc = np.concatenate([[0], np.cumsum([np.linalg.norm(cs[i + 1] - cs[i]) for i in range(N_FRAMES - 1)])])
    Topt = []
    for t in range(N_FRAMES):
        e = drift(kind, t, s_acc[t], yaws[t], st)
        jit = np.concatenate([rng.normal(0, 0.001, 3), rng.normal(0, np.deg2rad(0.05), 3)]) if t > 0 else np.zeros(6)
        Topt.append(G.se3_exp(jit) @ G.se3_exp(e) @ Tt[t])
    pw = PlaneWorld(CFG, K, H, W, legacy=None, out_dir=None)
    tt = lambda T: torch.from_numpy(T)
    planes, truth = measure(Tt[0], rng)
    pw.on_first_frame(0, planes, tt(Topt[0]))
    a_truth = {}

    def label_new(T, segs, seg_truth):
        for a in pw.anchors.values():
            if a.id in a_truth:
                continue
            n_c = T[:3, :3] @ a.n
            o_c = T[:3, :3] @ a.o + T[:3, 3]                 # = the creating segment's centroid (exact)
            best = min(range(len(segs)), key=lambda i: np.linalg.norm(segs[i].n - n_c) + np.linalg.norm(segs[i].p - o_c))
            a_truth[a.id] = seg_truth[best]

    # anchors of frame 0: one per merged segment, in order
    from utils.pw.segments import build_segments, merge_in_frame
    s0 = merge_in_frame(build_segments(planes, pw.sensor), planes, pw.sensor)
    label_new(Topt[0], s0, [truth[sg.src[0]] for sg in s0])
    nis_fresh, n_assign, n_correct, seen_kf = [], 0, 0, {}
    confusion = {}
    wrong_log = []
    wrong = None
    for k in range(1, N_FRAMES):
        is_kf = (k % KF_EVERY) == 0
        split = None
        if is_kf and k in dup_frames:
            split = "A" if k == dup_frames[0] else "floor"
            pw.skip_merge_once = True
        planes, truth = measure(Tt[k], rng, split)
        out = pw.on_frame(k, planes, tt(Topt[k - 1]))
        c = pw.cur
        seg_truth = [truth[sg.src[0]] for sg in c["segs"]]
        # assignment statistics (at the gate pose)
        for s, j in c["assign"]:
            a = pw.anchors[j]
            pt = a_truth.get(pw.anchors[a.primary_id].id, a_truth.get(j))
            n_assign += 1
            n_correct += int(pt == seg_truth[s])
            if pt != seg_truth[s]:
                key = f"{NAMES[seg_truth[s]]}->{NAMES[pt] if pt is not None else None}({a.state})"
                confusion[key] = confusion.get(key, 0) + 1
                if len(wrong_log) < 12:
                    right = [x.id for x in pw.primaries() if a_truth.get(x.id) == seg_truth[s]]
                    wrong_log.append(dict(k=k, seg=NAMES[seg_truth[s]], j=j, j_truth=NAMES[pt] if pt is not None else None,
                                          gap=c["kt"] - a.est.tau, sdV=np.sqrt(np.diag(a.est.cov_at(c["kt"], pw.Gamma))).round(4).tolist(),
                                          bhat=a.est.b.round(4).tolist(), nis=round(float(c["info"][(s, j)][0]), 2),
                                          right=[(x, dict((jj, ss) for ss, jj in c["assign"]).get(x),
                                                  round(float(c["info"].get((s, x), (np.nan,))[0]), 1)) for x in right],
                                          nsegs=len(c["segs"]), seg_truths=[NAMES[t] for t in seg_truth]))
            v = c["info"].get((s, j), (None,))[0]
            if v is not None and (c["kt"] - a.est.tau) < pw.Wb and (s, j) not in c["added"]:
                nis_fresh.append(v)
        if inject_wrong_at is not None and k == inject_wrong_at:
            ia = [s for s, t in enumerate(seg_truth) if NAMES[t] == "A"]
            ip = [s for s, t in enumerate(seg_truth) if NAMES[t] == "P4"]
            wa = [j for s, j in c["assign"] if s in ia]
            if ip and wa:
                j = wa[0]
                a = pw.anchors[j]
                b0, V0 = a.est.b.copy(), a.est.V.copy()
                c["assign"] = [(s, jj) for s, jj in c["assign"] if jj != j and s != ip[0]] + [(ip[0], j)]
                wrong = dict(j=j, b0=b0, V0=V0, n_rej0=a.est.n_reject)
        pw.after_tracking(k, tt(Topt[k]))
        if wrong is not None and "done" not in wrong:
            a = pw.anchors[wrong["j"]]
            wrong.update(done=True, rejected=a.est.n_reject > wrong["n_rej0"],
                         shift_sigma=float(np.max(np.abs(a.est.b - wrong["b0"]) / np.sqrt(np.diag(wrong["V0"])))))
        if is_kf:
            pw.on_keyframe(k, tt(Topt[k]), depth_image(Tt[k]))
            label_new(Topt[k], c["segs"], seg_truth)
            for t in set(seg_truth):
                seen_kf[t] = seen_kf.get(t, 0) + 1
    # merges: false when the truths of keep and drop differ
    merges = []
    for a in pw.anchors.values():
        if a.state == "merged":
            merges.append((a_truth.get(a.primary_id), a_truth.get(a.id)))
    false_merges = sum(1 for x, y in merges if x != y)
    active = {}
    for a in pw.primaries():
        if a.state == "active":
            active.setdefault(a_truth.get(a.id), []).append(a.id)
    planes_req = [t for t, n in seen_kf.items() if n >= 3]
    one_each = all(len(active.get(t, [])) == 1 for t in planes_req)
    return dict(kind=kind, seed=seed, nis_mean=float(np.mean(nis_fresh)) if nis_fresh else None,
                n_nis=len(nis_fresh), pct_correct=100.0 * n_correct / max(n_assign, 1), n_assign=n_assign,
                merges=len(merges), false_merges=false_merges, conflicts=pw.counters["conflicts"],
                revisits=pw.counters["revisits"], gamma=np.diag(pw.Gamma).tolist(), invariants=pw.n_invariant,
                active_per_plane={NAMES[t]: len(active.get(t, [])) for t in planes_req}, one_each=bool(one_each),
                bias_reject=pw.counters["bias_reject"], bias_accept=pw.counters["bias_accept"], wrong=wrong,
                violations=pw.violation_msgs[:5], confusion=confusion, wrong_log=wrong_log)
