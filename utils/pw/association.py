"""PLAN A2G-P1 v2 §3.5: IC + Hungarian, ambiguity flag, conflict rule, joint compatibility and GN init.

Decisions (IC, ambiguity, JC) use the offset-corrected anchor y_j = x_j (+) b_hat with
S = J_a R_s J_a^T + J_b V_j(k) J_b^T (+ J_xi Q J_xi^T at the gate pose). The pose constraint (GN init, Kabsch, and
the frontend's plane_ctx) uses the original anchor parameters. Geometry is passed explicitly: geom(a) -> (n, o, B).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.obs_decouple import obs_basis

from .geometry import facing, grid_index, overlap_or_adjacent, se3_exp, to_camera
from .model import CHI2_2, CHI2_3, ang_nis, chi2, jac_a, jac_b, jac_xi, nis, residual

BIG_M = 1e4
NO_PASS = 1e9


def ORIG(a):
    return a.n, a.o, a.B


def pair_terms(seg, g, T):
    """r(s | plane g = (n, o, B) in the world, taken into the camera at T) and its Jacobians."""
    n_c, o_c, B_c = to_camera(T, *g)
    r = residual(seg.n, seg.p, n_c, o_c, B_c)
    return r, jac_a(seg.n, seg.B, n_c, B_c), jac_b(seg.n, seg.p, n_c, o_c, B_c), jac_xi(seg.n, seg.p, n_c, B_c)


def pair_S(seg, g, V, T, Q=None):
    r, Ja, Jb, Jx = pair_terms(seg, g, T)
    S = Ja @ seg.R @ Ja.T + Jb @ V @ Jb.T
    if Q is not None:
        S = S + Jx @ Q @ Jx.T
    return r, 0.5 * (S + S.T), Ja, Jb, Jx


def ic_and_assign(segs, anchors, T, geom, Vk, Q=None, restrict_s=None, restrict_a=None):
    """IC of every (segment, facing primary anchor), conflict rule, Hungarian (max #pairs, then NIS + ln det S).
    Returns (cand, assign, info) with info[(s, aid)] = (nis, S, r)."""
    S_idx = list(range(len(segs))) if restrict_s is None else list(restrict_s)
    A_list = [a for a in anchors if (restrict_a is None or a.id in restrict_a)]
    A_list = [a for a in A_list if facing(*to_camera(T, *geom(a))[:2])]
    cand, info, passed = [], {}, {}
    for si in S_idx:
        for a in A_list:
            r, S, *_ = pair_S(segs[si], geom(a), Vk(a), T, Q)
            v = nis(r, S)
            ok = v <= CHI2_3
            info[(si, a.id)] = (v, S, r)
            passed[(si, a.id)] = ok
            cand.append(dict(s=si, j=a.id, state=a.state, nis=v, nll=v + float(np.linalg.slogdet(S)[1]), ic=bool(ok)))
    by_id = {a.id: a for a in A_list}
    all_by_id = {a.id: a for a in anchors}

    def jstar_passes(si, jid):
        """conflict rule: does j* pass IC with segment si? (j* may be outside A_list, e.g. already assigned)"""
        if (si, jid) in passed:
            return passed[(si, jid)]
        js = all_by_id.get(jid)
        if js is None or not facing(*to_camera(T, *geom(js))[:2]):
            return False
        r, S, *_ = pair_S(segs[si], geom(js), Vk(js), T, Q)
        return nis(r, S) <= CHI2_3

    for c in cand:
        a = by_id[c["j"]]
        if a.state == "conflict" and a.conflict_with is not None and jstar_passes(c["s"], a.conflict_with):
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


def ambiguity(segs, assign, anchors, T, geom, Vk, h, Q=None):
    """Unassigned segment s is ambiguous with active j when NIS_angle <= chi2_2 and the regions overlap or touch."""
    used = {s for s, _ in assign}
    out = []
    for si, sg in enumerate(segs):
        if si in used:
            continue
        for a in anchors:
            if a.state != "active":
                continue
            g = geom(a)
            if not facing(*to_camera(T, *g)[:2]):
                continue
            r, S, *_ = pair_S(sg, g, Vk(a), T, Q)
            if ang_nis(r, S) <= CHI2_2 and overlap_or_adjacent(seg_grid_idx(sg, T, a, h), a.cells):
                out.append((si, a.id))
    return out


# ------------------------------------------------------------------ JC and init


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
    return dict(rank=rank, shape=shape, jump=jump, total=float(r0 @ Si @ r0), dxi=dxi, I=I, Ip=Ip,
                dof_shape=len(r0) - rank)


def build_jc(pairs, segs, A, T, Q, geom, Vk):
    """Stacked residuals r(s_i | g_i), J_xi and Sigma: Sigma_ii = J_a R J_a^T + J_b V_i J_b^T; all i, l:
    Sigma_il += J_xi,i Q J_xi,l^T."""
    n = len(pairs)
    r0 = np.zeros(3 * n)
    J = np.zeros((3 * n, 6))
    Sig = np.zeros((3 * n, 3 * n))
    for i, (s, aid) in enumerate(pairs):
        r, Ja, Jb, Jx = pair_terms(segs[s], geom(A[aid]), T)
        r0[3 * i:3 * i + 3] = r
        J[3 * i:3 * i + 3] = Jx
        Sig[3 * i:3 * i + 3, 3 * i:3 * i + 3] += Ja @ segs[s].R @ Ja.T + Jb @ Vk(A[aid]) @ Jb.T
    Sig += J @ Q @ J.T
    return r0, J, 0.5 * (Sig + Sig.T)


def joint_compat(pairs, segs, A, T, Q, geom, Vk):
    """Shape test with greedy removal (the pair whose removal lowers shape most)."""
    pairs = list(pairs)
    dropped = []
    while pairs:
        r0, J, Sig = build_jc(pairs, segs, A, T, Q, geom, Vk)
        st = jc_stats(r0, J, Sig)
        dof = st["dof_shape"]
        if dof <= 0 or st["shape"] <= chi2(dof):
            return pairs, st, dropped
        best, best_shape = None, None
        for i in range(len(pairs)):
            sub = pairs[:i] + pairs[i + 1:]
            sh = 0.0 if not sub else jc_stats(*build_jc(sub, segs, A, T, Q, geom, Vk))["shape"]
            if best_shape is None or sh < best_shape:
                best, best_shape = i, sh
        dropped.append(pairs.pop(best))
    return [], None, dropped


def kabsch_lsq(pairs, segs, A):
    """E1c closed form on ORIGINAL anchors (Kabsch on normals, weighted LSQ on d). None if < 3 pairs."""
    if len(pairs) < 3:
        return None
    M = np.stack([segs[s].n for s, _ in pairs])
    N = np.stack([A[a].n for _, a in pairs])
    dm = np.array([segs[s].d for s, _ in pairs])
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
    r = np.concatenate([pair_terms(segs[s], ORIG(A[a]), T)[0] for s, a in pairs])
    return float(r @ Si @ r)


def gauss_newton(pairs, segs, A, T0, Sig, iters=3):
    """GN on the ORIGINAL anchors (the pose constraint never uses b_hat), Sigma fixed."""
    Si = np.linalg.inv(Sig)
    T = T0.copy()
    for _ in range(iters):
        rs, Js = [], []
        for s, a in pairs:
            r, _, _, Jx = pair_terms(segs[s], ORIG(A[a]), T)
            rs.append(r)
            Js.append(Jx)
        r = np.concatenate(rs)
        J = np.concatenate(Js)
        dx = -np.linalg.pinv(J.T @ Si @ J, rcond=1e-12) @ J.T @ Si @ r
        T = se3_exp(dx) @ T
    return T, gn_cost(pairs, segs, A, T, Si)


def init_from_jc(pairs, segs, A, T_gate, Q, geom, Vk, obs_threshold=0.3, gn_iters=3):
    """JC on offset-corrected anchors at T_gate; GN init (rank 6) on original anchors from T_gate and Kabsch."""
    out = dict(source="none_pairs", T_init=T_gate, pose_pairs=[], jc=None, dxi=None, removed=[], Ip=None)
    if not pairs:
        return out
    kept, st, dropped = joint_compat(pairs, segs, A, T_gate, Q, geom, Vk)
    out["removed"] = dropped
    if st is None:
        out["jc"] = dict(n=0, rank=0, total=0.0, shape=0.0, jump=0.0, dropped=len(dropped), dof_shape=0,
                         decision="all_dropped")
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
        _, _, Sig = build_jc(kept, segs, A, T_gate, Q, geom, Vk)
        cands = [gauss_newton(kept, segs, A, T_gate, Sig, gn_iters)]
        Tk = kabsch_lsq(kept, segs, A)
        if Tk is not None:
            cands.append(gauss_newton(kept, segs, A, Tk, Sig, gn_iters))
        best = min(range(len(cands)), key=lambda i: cands[i][1])
        Tp, cost = cands[best]
        out.update(source="plane", T_init=Tp, pose_pairs=kept, dxi=st["dxi"], Ip=st["Ip"], gn_cost=cost,
                   gn_start="kabsch" if best == 1 else "gate")
        jc["decision"] = "plane"
        return out
    jc["decision"] = "rank_lt6"
    out.update(source="rank_lt6", pose_pairs=kept)
    return out
