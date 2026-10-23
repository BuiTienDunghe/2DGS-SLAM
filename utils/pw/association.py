"""PLAN A2G-P1 §3.6 (IC, Hungarian, conflict rule, ambiguity flag) and §3.8 (JC, Gauss-Newton init, F5)."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.obs_decouple import obs_basis

from .geometry import grid_index, overlap_or_adjacent, se3_exp, to_camera, facing
from .model import CHI2_2, CHI2_3, ang_nis, chi2, jac_a, jac_b, jac_xi, nis, residual, residual_cov

BIG_M = 1e4
NO_PASS = 1e9


def pair_terms(seg, a, T):
    """r(s|j) with the anchor taken into the camera at T, and its Jacobians."""
    n_c, o_c, B_c = to_camera(T, a.n, a.o, a.B)
    r = residual(seg.n, seg.p, n_c, o_c, B_c)
    return r, jac_a(seg.n, seg.B, n_c, B_c), jac_b(seg.n, seg.p, n_c, o_c, B_c), jac_xi(seg.n, seg.p, n_c, B_c)


def pair_S(seg, a, T, X, kind="seg_anchor_pred", Delta=None):
    r, Ja, Jb, Jx = pair_terms(seg, a, T)
    S = residual_cov(kind, Ja, seg.R, Jb, a.P, Jx, X, Delta)
    return r, S, Ja, Jb, Jx


def ic_and_assign(segs, anchors, T, X_of, kind="seg_anchor_pred", restrict_s=None, restrict_a=None, Delta_of=None):
    """IC of every (segment, facing primary anchor), conflict rule, Hungarian (max #pairs, then NIS + ln det S).
    X_of(anchor) -> 6x6 pose covariance entering through J_xi. Returns (cand, assign, info) with
    info[(s, aid)] = (nis, S, r)."""
    S_idx = list(range(len(segs))) if restrict_s is None else list(restrict_s)
    A_list = [a for a in anchors if (restrict_a is None or a.id in restrict_a)]
    A_list = [a for a in A_list if facing(*to_camera(T, a.n, a.o, a.B)[:2])]
    cand, info, passed = [], {}, {}
    for si in S_idx:
        for a in A_list:
            Delta = Delta_of(a) if Delta_of is not None else None
            r, S, *_ = pair_S(segs[si], a, T, X_of(a), kind, Delta)
            v = nis(r, S)
            ok = v <= CHI2_3
            nll = v + float(np.linalg.slogdet(S)[1])
            info[(si, a.id)] = (v, S, r)
            passed[(si, a.id)] = ok
            cand.append(dict(s=si, j=a.id, state=a.state, nis=v, nll=nll, ic=bool(ok)))
    # conflict rule: c(j*) is no candidate for a segment that j* passes IC with
    for c in cand:
        a = next(x for x in A_list if x.id == c["j"])
        if a.state == "conflict" and passed.get((c["s"], a.conflict_with), False):
            c["ic"] = False
            c["blocked"] = True
    assign = []
    if S_idx and A_list:
        M = np.full((len(S_idx), len(A_list)), NO_PASS)
        pos_s = {s: i for i, s in enumerate(S_idx)}
        pos_a = {a.id: j for j, a in enumerate(A_list)}
        for c in cand:
            if c["ic"]:
                M[pos_s[c["s"]], pos_a[c["j"]]] = c["nll"] - BIG_M
        rr, cc = linear_sum_assignment(M)
        for i, j in zip(rr, cc):
            if M[i, j] < 1e8:
                assign.append((S_idx[i], A_list[j].id))
    assign.sort()
    return cand, assign, info


def seg_grid_idx(seg, T, a, h):
    Xw = (seg.mu - T[:3, 3]) @ T[:3, :3]
    return grid_index(Xw, a.grid_o, a.grid_B, h)


def ambiguity(segs, assign, anchors, T, X_of, h):
    """Unassigned segment s is ambiguous with active/frozen j when NIS_angle <= chi2_2 and the regions overlap
    or touch (s's cells projected into j's grid with the pose in use)."""
    used = {s for s, _ in assign}
    out = []
    for si, sg in enumerate(segs):
        if si in used:
            continue
        for a in anchors:
            if a.state not in ("active", "frozen"):
                continue
            n_c, o_c, B_c = to_camera(T, a.n, a.o, a.B)
            if not facing(n_c, o_c):
                continue
            r, S, *_ = pair_S(sg, a, T, X_of(a))
            if ang_nis(r, S) <= CHI2_2 and overlap_or_adjacent(seg_grid_idx(sg, T, a, h), a.cells):
                out.append((si, a.id))
    return out


# ------------------------------------------------------------------ JC (K4) and init (F5)


def jc_stats(r0, J, Sig):
    Si = np.linalg.inv(Sig)
    I = J.T @ Si @ J
    I = 0.5 * (I + I.T)
    w, V = np.linalg.eigh(I)
    keep = w > 1e-9 * max(float(w.max()), 1e-300)
    rank = int(keep.sum())
    Ip = (V[:, keep] / w[keep]) @ V[:, keep].T
    Wm = Si - Si @ J @ Ip @ J.T @ Si
    shape = float(r0 @ Wm @ r0)
    dxi = -Ip @ J.T @ Si @ r0
    jump = float(dxi @ I @ dxi)
    total = float(r0 @ Si @ r0)
    return dict(rank=rank, shape=shape, jump=jump, total=total, dxi=dxi, I=I, Ip=Ip, dof_shape=len(r0) - rank)


def build_jc(pairs, segs, A, T, Q, cov_eps):
    """Stacked residuals, J_xi and Sigma (K4). cov_eps(aid_i, aid_l) -> 6x6 (zeros for d_mode none)."""
    n = len(pairs)
    r0 = np.zeros(3 * n)
    J = np.zeros((3 * n, 6))
    Sig = np.zeros((3 * n, 3 * n))
    for i, (s, aid) in enumerate(pairs):
        r, Ja, Jb, Jx = pair_terms(segs[s], A[aid], T)
        r0[3 * i:3 * i + 3] = r
        J[3 * i:3 * i + 3] = Jx
        Sig[3 * i:3 * i + 3, 3 * i:3 * i + 3] += Ja @ segs[s].R @ Ja.T + Jb @ A[aid].P @ Jb.T
    for i, (_, ai) in enumerate(pairs):
        for l, (_, al) in enumerate(pairs):
            Sig[3 * i:3 * i + 3, 3 * l:3 * l + 3] += J[3 * i:3 * i + 3] @ (Q + cov_eps(ai, al)) @ J[3 * l:3 * l + 3].T
    return r0, J, 0.5 * (Sig + Sig.T)


def joint_compat(pairs, segs, A, T, Q, cov_eps):
    """Shape test with greedy removal (the pair whose removal lowers shape most). Returns (kept_pairs, stats, dropped)."""
    pairs = list(pairs)
    dropped = []
    while pairs:
        r0, J, Sig = build_jc(pairs, segs, A, T, Q, cov_eps)
        st = jc_stats(r0, J, Sig)
        dof = 3 * len(pairs) - st["rank"]
        st["dof_shape"] = dof
        if dof <= 0 or st["shape"] <= chi2(dof):
            return pairs, st, dropped
        best, best_shape = None, None
        for i in range(len(pairs)):
            sub = pairs[:i] + pairs[i + 1:]
            if not sub:
                sh = 0.0
            else:
                r1, J1, S1 = build_jc(sub, segs, A, T, Q, cov_eps)
                sh = jc_stats(r1, J1, S1)["shape"]
            if best_shape is None or sh < best_shape:
                best, best_shape = i, sh
        dropped.append(pairs.pop(best))
    return [], None, dropped


def kabsch_lsq(pairs, segs, A, T_prior=None, max_trans=None, max_rot=None):
    """E1c closed form (Kabsch on normals, weighted LSQ on d); optional E1c sanity check. None if < 3 pairs."""
    if len(pairs) < 3:
        return None
    M = np.stack([segs[s].meas.n.numpy() if hasattr(segs[s].meas.n, "numpy") else segs[s].n for s, _ in pairs])
    N = np.stack([A[a].n for _, a in pairs])
    dm = np.array([float(segs[s].meas.d) for s, _ in pairs])
    dw = np.array([-float(A[a].n @ A[a].o) for _, a in pairs])
    w = np.array([float(segs[s].meas.n_px) for s, _ in pairs])
    w = w / w.sum()
    Hm = (w[:, None, None] * np.einsum("ki,kj->kij", N, M)).sum(0)
    U, _, Vt = np.linalg.svd(Hm)
    V = Vt.T
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(V @ U.T))])
    Rm = V @ D @ U.T
    t = np.linalg.lstsq(M * np.sqrt(w)[:, None], (dw - dm) * np.sqrt(w), rcond=None)[0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm, t
    return T


def gn_cost(pairs, segs, A, T, Si):
    r = np.concatenate([pair_terms(segs[s], A[a], T)[0] for s, a in pairs])
    return float(r @ Si @ r)


def gauss_newton(pairs, segs, A, T0, Sig, iters=3):
    Si = np.linalg.inv(Sig)
    T = T0.copy()
    for _ in range(iters):
        rs, Js = [], []
        for s, a in pairs:
            r, _, _, Jx = pair_terms(segs[s], A[a], T)
            rs.append(r)
            Js.append(Jx)
        r = np.concatenate(rs)
        J = np.concatenate(Js)
        Hm = J.T @ Si @ J
        dx = -np.linalg.pinv(Hm, rcond=1e-12) @ J.T @ Si @ r
        T = se3_exp(dx) @ T
    return T, gn_cost(pairs, segs, A, T, Si)


def init_from_jc(pairs, segs, A, T_pred, Q, cov_eps, obs_threshold=0.3, gn_iters=3):
    """§3.8. Returns dict(source, T_init, pose_pairs, jc, dxi, removed)."""
    out = dict(source="none_pairs", T_init=T_pred, pose_pairs=[], jc=None, dxi=None, removed=[], Ip=None)
    if not pairs:
        return out
    kept, st, dropped = joint_compat(pairs, segs, A, T_pred, Q, cov_eps)
    out["removed"] = dropped
    if st is None:
        out["jc"] = dict(n=0, rank=0, total=0.0, shape=0.0, jump=0.0, dropped=len(dropped), decision="all_dropped")
        return out
    jc = dict(n=len(kept), rank=st["rank"], total=st["total"], shape=st["shape"], jump=st["jump"],
              dropped=len(dropped), dof_shape=st["dof_shape"])
    out["jc"] = jc
    if st["jump"] > chi2(st["rank"]):
        jc["decision"] = "jump"
        out["source"] = "jc_jump"
        return out
    ob = obs_basis(np.stack([segs[s].n for s, _ in kept]), obs_threshold)
    if st["rank"] == 6 and ob["Mt"].sum() == 3 and ob["Mr"].sum() == 3:
        _, _, Sig = build_jc(kept, segs, A, T_pred, Q, cov_eps)
        cands = [gauss_newton(kept, segs, A, T_pred, Sig, gn_iters)]
        Tk = kabsch_lsq(kept, segs, A)
        if Tk is not None:
            cands.append(gauss_newton(kept, segs, A, Tk, Sig, gn_iters))
        Tp, cost = min(cands, key=lambda c: c[1])
        out.update(source="plane", T_init=Tp, pose_pairs=kept, dxi=st["dxi"], Ip=st["Ip"], gn_cost=cost,
                   gn_start="kabsch" if (len(cands) == 2 and cands[1][1] < cands[0][1]) else "pred")
        jc["decision"] = "plane"
        return out
    jc["decision"] = "rank_lt6"
    out.update(source="rank_lt6", pose_pairs=kept)
    return out
