"""PLAN A2G-P1 gen §3.3 (A1, A2): does the error genealogy (M3-M5) explain the true pose errors of real runs?

Per run: true eps (world frame, no alignment); sparse pixels (stride 8, depth 0.1-4 m) back-projected with the
estimated pose; provenance of each pixel among earlier keyframes (4 rules); information H_p per parent (depth
point-to-plane, optional photometric, optional anchor parents); Q_mot (M5); recursion M3-M4 over keyframe nodes.
E-local (A2): u_k = eps_k - (sum_p W_p eps_p + W_prev eps_{k-1}), NIS6 = u^T A_k u.
E-rel (A1): d = J_eps (eps_k - eps_t(j)) for every pose pair (s, j) of frame k; NIS3 and NLL under the genealogy
(G) and three fitted reference models (T: random walk in time, D: path length, C: constant).
Variant (owner rule x photometric x anchor parents) and lambda are chosen on snt_near only, then locked.
usage: python tools/a2g_gen/stage0_genealogy.py        -> stage0/a12_summary.json, a12_pairs_<seq>_<run>.npz
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT0, RUNS, G, Ad, frame_ids, jsonl, load_cfg, load_ds, load_run, read_frame  # noqa: E402

from utils.obs_decouple import obs_basis  # noqa: E402
from utils.pw.uncertainty import CHI2_3, jac_xi  # noqa: E402

STRIDE = 8
ZMIN, ZMAX = 0.1, 4.0
RULES = ["closest", "first", "closest_act", "first_act"]
LAMS = 10.0 ** np.arange(-1.0, 5.0 + 1e-9, 0.1)
MOT_WINDOW = 30
Q0 = np.diag([0.01 ** 2] * 3 + [np.deg2rad(0.5) ** 2] * 3)
LOG2PI = np.log(2 * np.pi)


# ------------------------------------------------------------------ pass over the dataset (per sequence)


def seq_pass(seq, runs):
    """Sparse depth / gradients of every tracked frame and the robust photometric residual sigma_c per run."""
    cfg = load_cfg(seq)
    ds = load_ds(cfg)
    K = np.array(ds.K, dtype=np.float64)
    fids = frame_ids(cfg, ds)
    H, W = int(ds.height), int(ds.width)
    vs = np.arange(STRIDE // 2, H, STRIDE)
    us = np.arange(STRIDE // 2, W, STRIDE)
    n = min(len(fids), max(len(R["fids"]) for R in runs.values()))
    for R in runs.values():
        assert R["fids"] == fids[:len(R["fids"])], "run frame ids differ from the loader's"
    D = np.zeros((n, len(vs), len(us)))
    GX = np.zeros_like(D)
    GY = np.zeros_like(D)
    res = {m: [] for m in runs}
    uu, vv = np.meshgrid(us, vs)
    Kinv = np.linalg.inv(K)
    prev = None
    for i in range(n):
        gray, depth, _ = read_frame(ds, fids[i])
        D[i] = depth[np.ix_(vs, us)]
        GX[i] = (cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) / 8.0)[np.ix_(vs, us)]
        GY[i] = (cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) / 8.0)[np.ix_(vs, us)]
        if prev is not None:
            ok = (D[i] > ZMIN) & (D[i] < ZMAX)
            pix = np.stack([uu[ok], vv[ok], np.ones(ok.sum())], 1)
            Xc = (pix @ Kinv.T) * D[i][ok][:, None]
            for m, R in runs.items():
                if i >= len(R["fids"]):
                    continue
                Tk, Tp = R["Th"][i], R["Th"][i - 1]
                Xw = (Xc - Tk[:3, 3]) @ Tk[:3, :3]
                Xp = Xw @ Tp[:3, :3].T + Tp[:3, 3]
                z = Xp[:, 2]
                u = K[0, 0] * Xp[:, 0] / z + K[0, 2]
                v = K[1, 1] * Xp[:, 1] / z + K[1, 2]
                inb = (z > ZMIN) & (u >= 0) & (u <= W - 2) & (v >= 0) & (v <= H - 2)
                u0, v0 = np.floor(u[inb]).astype(int), np.floor(v[inb]).astype(int)
                a, b = u[inb] - u0, v[inb] - v0
                Ip = (prev[v0, u0] * (1 - a) * (1 - b) + prev[v0, u0 + 1] * a * (1 - b) + prev[v0 + 1, u0] * (1 - a) * b
                      + prev[v0 + 1, u0 + 1] * a * b)
                res[m].append(gray[vv[ok], uu[ok]][inb] - Ip)
        prev = gray
    sig_c = {}
    for m, r in res.items():
        r = np.concatenate(r)
        sig_c[m] = float(1.4826 * np.median(np.abs(r - np.median(r))))
    return dict(cfg=cfg, K=K, fids=fids, vs=vs, us=us, D=D, GX=GX, GY=GY, sig_c=sig_c, H=H, W=W)


# ------------------------------------------------------------------ per run: parents and information


def plane_tables(R, cfg):
    """Anchor parents and E-rel pairs from the run logs. created_kf for both E1c and P1 core (E-rel); E1c anchor
    parents in M3 are the keyframes merged into the anchor (n_obs_kf increments), weighted by the matched n_px."""
    run = R["dir"]
    idx = {f: i for i, f in enumerate(R["fids"])}
    planes = {r["frame_id"]: r["planes"] for r in jsonl(os.path.join(run, "planes.jsonl"))}
    created, merges = {}, {}
    last = {}
    for row in jsonl(os.path.join(run, "anchors_kf.jsonl")):
        f = row["frame_id"]
        pairs_f = dict((a, s) for s, a in R["pairs"][idx[f]]) if f in idx else {}
        for an in row["anchors"]:
            aid = an["id"]
            created.setdefault(aid, an.get("created_kf", f))
            nk = an.get("n_obs_kf", 1)
            if nk > last.get(aid, 0):
                s = pairs_f.get(aid)
                w = float(planes[f][s]["n_px"]) if (s is not None and f in planes and s < len(planes[f])) else np.nan
                merges.setdefault(aid, []).append((idx.get(f, -1), w))
                last[aid] = nk
    for aid, lst in merges.items():
        ws = [w for _, w in lst if np.isfinite(w)]
        fill = float(np.mean(ws)) if ws else 1.0
        merges[aid] = [(i, w if np.isfinite(w) else fill) for i, w in lst]
    return planes, created, merges, idx


def precompute_run(R, P, method, seq):
    """Per tracked frame: per-node information for every rule (depth, photo), anchor-parent information, E-rel
    pairs (creation node, J_eps), Q_mot^-1. Nodes = keyframes (ordinal)."""
    cfg, K = P["cfg"], P["K"]
    se = cfg["PlaneWorld"]["sensor"]
    sz_a, sz_b = float(se.get("a", 0.0)), float(se.get("b", 1.425e-3))
    pt = cfg["PlaneTracking"]
    sig_n, sig_d = float(pt.get("sigma_n", 0.02)), float(pt.get("sigma_d", 0.01))
    obs_thr = float(pt.get("obs_threshold", pt.get("min_singular_value", 0.3)))
    Nwin = int(cfg["Training"]["old_than_N_keyframe"])
    sig_c = P["sig_c"][method]
    n = len(R["fids"])
    kf_idx = [i for i in range(n) if R["is_kf"][i]]
    node_of = {i: o for o, i in enumerate(kf_idx)}
    Kinv = np.linalg.inv(K)
    uu, vv = np.meshgrid(P["us"], P["vs"])
    pix = np.stack([uu, vv, np.ones_like(uu)], -1).astype(np.float64)
    rays = pix @ Kinv.T                                     # (60, 80, 3)
    has_pl = R["has_planes"] and method != "base"
    if has_pl:
        planes, created, merges, idx = plane_tables(R, cfg)
    # motion second moment (world frame, right perturbation = eps frame)
    dw = [np.zeros(6)] + [G.se3_log(G.inv_T(R["Th"][i - 1]) @ R["Th"][i]) for i in range(1, n)]
    out = []
    t0 = time.time()
    for i in range(n):
        T = R["Th"][i]
        Rcw, tcw = T[:3, :3], T[:3, 3]
        win = dw[max(1, i - MOT_WINDOW):i]
        # M5; the 6x6 second moment needs >= 6 motions for full rank, the start value is used until then
        Q = (np.mean([np.outer(d, d) for d in win], 0) + 1e-10 * np.eye(6)) if len(win) >= 6 else Q0.copy()
        fr = dict(Qi=np.linalg.inv(Q), Hd={r: {} for r in RULES}, Hc={r: {} for r in RULES}, Hp={}, pairs=[],
                  n_own={r: 0 for r in RULES}, n_px=0)
        if i == 0:
            out.append(fr)
            continue
        Dk = P["D"][i]
        valid = (Dk > ZMIN) & (Dk < ZMAX)
        Xc = rays * Dk[..., None]
        nc = np.cross(Xc[:, 1:, :][:-1] - Xc[:-1, :-1], Xc[1:, :-1] - Xc[:-1, :-1])     # (59, 79, 3)
        vn = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1]
        nn = np.linalg.norm(nc, axis=-1)
        vn &= nn > 0
        sel = np.zeros_like(valid)
        sel[:-1, :-1] = vn
        Xs = Xc[sel]
        ncs = nc[vn] / nn[vn][:, None]
        z = Xs[:, 2]
        Xw = (Xs - tcw) @ Rcw
        nw = ncs @ Rcw
        Jd = np.concatenate([nw, np.cross(Xw, nw)], 1)
        wd = 1.0 / (sz_a + sz_b * z ** 2) ** 2
        g = np.stack([P["GX"][i][sel], P["GY"][i][sel]], 1)
        fx, fy = K[0, 0], K[1, 1]
        dpi = np.zeros((len(z), 2, 3))
        dpi[:, 0, 0] = fx / z
        dpi[:, 0, 2] = -fx * Xs[:, 0] / z ** 2
        dpi[:, 1, 1] = fy / z
        dpi[:, 1, 2] = -fy * Xs[:, 1] / z ** 2
        av = np.einsum("pi,pij,jk->pk", g, dpi, Rcw)          # gradient^T dpi R_cw (world row vector)
        Jc = np.concatenate([av, np.cross(Xw, av)], 1)
        fr["n_px"] = int(len(z))
        # provenance over earlier keyframes
        prev_kf = [t for t in kf_idx if t < i]
        m = len(prev_kf)
        if m:
            Tk = np.stack([R["Th"][t] for t in prev_kf])
            Xt = np.einsum("tij,pj->tpi", Tk[:, :3, :3], Xw) + Tk[:, None, :3, 3]
            zt = Xt[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                ut = K[0, 0] * Xt[..., 0] / zt + K[0, 2]
                vt = K[1, 1] * Xt[..., 1] / zt + K[1, 2]
            inim = (zt > ZMIN) & (ut >= 0) & (ut < P["W"]) & (vt >= 0) & (vt < P["H"])
            cc = np.clip(np.rint((np.nan_to_num(ut) - STRIDE // 2) / STRIDE).astype(int), 0, len(P["us"]) - 1)
            rr = np.clip(np.rint((np.nan_to_num(vt) - STRIDE // 2) / STRIDE).astype(int), 0, len(P["vs"]) - 1)
            Dt = P["D"][np.array(prev_kf)][np.arange(m)[:, None], rr, cc]
            seen = inim & (Dt > ZMIN) & (Dt < ZMAX) & (np.abs(zt - Dt) <= 3 * (sz_a + sz_b * zt ** 2) + 0.01)
            dist = np.where(seen, np.linalg.norm(Xt, axis=-1), np.inf)
            anys = seen.any(0)
            own = {"closest": np.where(anys, np.argmin(dist, 0), -1), "first": np.where(anys, np.argmax(seen, 0), -1)}
            start = np.full(len(z), -1)
            last = np.full(len(z), -1)
            for o in range(m):
                s = seen[o]
                new = s & ((last < 0) | (o - last > Nwin))
                start[new] = o
                last[s] = o
            act = (last >= 0) & (last >= m - Nwin)
            own["first_act"] = np.where(act, start, -1)
            dmask = np.where(seen & (np.arange(m)[:, None] >= start[None, :]) & act[None, :], dist, np.inf)
            own["closest_act"] = np.where(act, np.argmin(dmask, 0), -1)
            Od = (Jd[:, :, None] * Jd[:, None, :]) * wd[:, None, None]
            Oc = (Jc[:, :, None] * Jc[:, None, :]) / sig_c ** 2
            for r in RULES:
                o = own[r]
                for node in np.unique(o[o >= 0]):
                    msk = o == node
                    fr["Hd"][r][int(node)] = Od[msk].sum(0)
                    fr["Hc"][r][int(node)] = Oc[msk].sum(0)
                fr["n_own"][r] = int((o >= 0).sum())
        # anchor parents and E-rel pairs
        if has_pl and R["pairs"][i]:
            fid = R["fids"][i]
            pl = planes.get(fid, [])
            prs = [(s, a) for s, a in R["pairs"][i] if s < len(pl)]
            ns = [np.array(pl[s]["n"], dtype=np.float64) for s, _ in prs]
            ns = [x / np.linalg.norm(x) for x in ns]
            wpx = np.array([float(pl[s]["n_px"]) for s, _ in prs])
            wpx = wpx / wpx.sum() if len(wpx) else wpx
            ob = obs_basis(ns, obs_thr)
            Pt = ob["Vt"] @ np.diag(ob["Mt"]) @ ob["Vt"].T
            Pr = ob["Vr"] @ np.diag(ob["Mr"]) @ ob["Vr"].T
            Pj = np.zeros((6, 6))
            Pj[:3, :3], Pj[3:, 3:] = Pt, Pr
            Sinv = np.diag([1 / sig_n ** 2, 1 / sig_n ** 2, 1 / sig_d ** 2])
            Adk = Ad(T)
            for (s, a), n_a, w in zip(prs, ns, wpx):
                d_a = float(pl[s]["d"])
                on = valid & (np.abs((Xc @ n_a) + d_a) <= 3 * (sz_a + sz_b * Dk ** 2) + 0.01)
                p_a = Xc[on].mean(0) if on.any() else -d_a * n_a
                B_c = G.basis(n_a)
                Jx = jac_xi(n_a, p_a, n_a, B_c)
                Hx = Pj @ (w * Jx.T @ Sinv @ Jx) @ Pj
                He = Adk.T @ Hx @ Adk
                # parent mixture (nodes) of the anchor as of frame i
                if method == "dec":
                    mix = [(t, wt) for t, wt in merges.get(a, []) if 0 <= t < i and t in node_of]
                else:
                    t = idx.get(created.get(a, -1), -1)
                    mix = [(t, 1.0)] if 0 <= t < i and t in node_of else []
                if mix:
                    tot = sum(wt for _, wt in mix)
                    for t, wt in mix:
                        nd = node_of[t]
                        fr["Hp"][nd] = fr["Hp"].get(nd, 0) + He * (wt / tot)
                tc = idx.get(created.get(a, -1), -1)
                if 0 <= tc < i and tc in node_of:
                    fr["pairs"].append((node_of[tc], Jx @ Adk, a))
        out.append(fr)
        if (i + 1) % 100 == 0:
            print(f"    {seq}/{method}: {i + 1}/{n} frames, {time.time() - t0:.0f} s", flush=True)
    return dict(frames=out, kf_idx=kf_idx, node_of=node_of, eps=R["eps"], n=n, sig_c=sig_c, Th=R["Th"])


# ------------------------------------------------------------------ M3-M4 for one variant


def frame_info(fr, rule, photo, anchor):
    Hn = {}
    for nd, Hm in fr["Hd"][rule].items():
        Hn[nd] = Hm.copy()
    if photo:
        for nd, Hm in fr["Hc"][rule].items():
            Hn[nd] = Hn.get(nd, 0) + Hm
    if anchor:
        for nd, Hm in fr["Hp"].items():
            Hn[nd] = Hn.get(nd, 0) + Hm
    return Hn


def build_variant(pre, rule, photo, anchor):
    eps, kf_idx = pre["eps"], pre["kf_idx"]
    eps_node = eps[np.array(kf_idx)]
    F = []
    for i, fr in enumerate(pre["frames"]):
        if i == 0:
            F.append(None)
            continue
        Hn = frame_info(fr, rule, photo, anchor)
        Hs = sum(Hn.values()) if Hn else np.zeros((6, 6))
        b = sum(Hm @ eps_node[nd] for nd, Hm in Hn.items()) if Hn else np.zeros(6)
        F.append(dict(Hn=Hn, H=0.5 * (Hs + Hs.T), b=b, Qi=fr["Qi"]))
    return F


def local_nll(F, eps, lams):
    """Total E-local negative log-likelihood for every lambda, and NIS6 per frame for every lambda."""
    n = len(F)
    nll = np.zeros(len(lams))
    nis = np.zeros((len(lams), n - 1))
    for i in range(1, n):
        f = F[i]
        A = f["H"][None] / lams[:, None, None] + f["Qi"][None]
        rhs = f["b"][None] / lams[:, None] + (f["Qi"] @ eps[i - 1])[None]
        mean = np.linalg.solve(A, rhs[..., None])[..., 0]
        u = eps[i][None] - mean
        q = np.einsum("li,lij,lj->l", u, A, u)
        _, ld = np.linalg.slogdet(A)
        nll += 0.5 * (q - ld + 6 * LOG2PI)
        nis[:, i - 1] = q
    return nll, nis


def recursion(F, pre, lam):
    """M4: rows of every frame (Cov(eps_k, nodes), Var(eps_k)); C over keyframe nodes."""
    n = len(F)
    kf_set = set(pre["kf_idx"])
    C = np.zeros((6, 6))                            # node 0 = frame 0, eps_0 = 0
    rows, V = [np.zeros((6, 6))], [np.zeros((6, 6))]
    c_prev, V_prev = np.zeros((6, 6)), np.zeros((6, 6))
    nmax = 1
    for i in range(1, n):
        f = F[i]
        A = f["H"] / lam + f["Qi"]
        Ainv = np.linalg.inv(A)
        Ainv = 0.5 * (Ainv + Ainv.T)
        Wp = Ainv @ f["Qi"]
        N = C.shape[0] // 6
        Mf = np.zeros((6, 6 * N))
        for nd, Hm in f["Hn"].items():
            Mf[:, 6 * nd:6 * nd + 6] = Ainv @ Hm / lam
        MC = Mf @ C
        c = MC + Wp @ c_prev
        X = Mf @ c_prev.T @ Wp.T
        Vk = MC @ Mf.T + Wp @ V_prev @ Wp.T + X + X.T + Ainv
        Vk = 0.5 * (Vk + Vk.T)
        rows.append(c)
        V.append(Vk)
        if i in kf_set:
            C = np.block([[C, c.T], [c, Vk]])
            c = np.concatenate([c, Vk], 1)
            nmax = max(nmax, C.shape[0] // 6)
        c_prev, V_prev = c, Vk
    ev = np.linalg.eigvalsh(0.5 * (C + C.T)).min() if C.shape[0] else 0.0
    return rows, V, C, dict(n_nodes=C.shape[0] // 6, min_eig_C=float(ev))


def erel_pairs(pre, rows, V, C):
    """E-rel samples: frame, anchor, creation node, d (3), Sigma_G (3x3), e = eps_k - eps_t (6), J (3x6)."""
    eps, kf_idx = pre["eps"], pre["kf_idx"]
    out = []
    for i, fr in enumerate(pre["frames"]):
        for nd, J, a in fr["pairs"]:
            t = kf_idx[nd]
            if t == i:
                continue
            e = eps[i] - eps[t]
            cr = rows[i][:, 6 * nd:6 * nd + 6]
            Cr = V[i] + C[6 * nd:6 * nd + 6, 6 * nd:6 * nd + 6] - cr - cr.T
            out.append(dict(k=i, t=t, a=a, d=J @ e, S=J @ (0.5 * (Cr + Cr.T)) @ J.T, e=e, J=J))
    return out


def gauss3(d, S):
    S = 0.5 * (S + S.T)
    q = float(d @ np.linalg.solve(S, d))
    _, ld = np.linalg.slogdet(2 * np.pi * S)
    return q, 0.5 * (q + ld)


# ------------------------------------------------------------------ reference models T, D, C


_CUM = {}


def path_stats(Th, t, k):
    """Path length and summed rotation angle of the estimated trajectory between tracked frames t and k."""
    key = id(Th)
    if key not in _CUM:
        ds = [0.0] + [float(np.linalg.norm(G.centre(Th[i]) - G.centre(Th[i - 1]))) for i in range(1, len(Th))]
        dp = [0.0] + [G.rot_angle((G.inv_T(Th[i - 1]) @ Th[i])[:3, :3]) for i in range(1, len(Th))]
        _CUM[key] = (np.cumsum(ds), np.cumsum(dp), Th)
    cs, cp, _ = _CUM[key]
    return float(cs[k] - cs[t]), float(cp[k] - cp[t])


def fit_models(pairs, Th):
    E = np.array([p["e"] for p in pairs])
    dk = np.array([p["k"] - p["t"] for p in pairs], dtype=np.float64)
    sp = np.array([path_stats(Th, p["t"], p["k"]) for p in pairs])
    Gam = np.mean(E ** 2 / dk[:, None], 0)
    ok_s, ok_r = sp[:, 0] > 0, sp[:, 1] > 0
    kt2 = float(np.mean(np.sum(E[ok_s, :3] ** 2, 1) / sp[ok_s, 0]) / 3)
    kr2 = float(np.mean(np.sum(E[ok_r, 3:] ** 2, 1) / sp[ok_r, 1]) / 3)
    S0 = E.T @ E / len(E)
    return dict(Gamma=Gam.tolist(), kappa_t2=kt2, kappa_r2=kr2, Sigma0=S0.tolist())


def model_cov(name, M, p, Th, cache):
    if name == "T":
        return np.diag(M["Gamma"]) * (p["k"] - p["t"])
    if name == "D":
        key = (p["t"], p["k"])
        if key not in cache:
            cache[key] = path_stats(Th, p["t"], p["k"])
        s, phi = cache[key]
        return np.diag([M["kappa_t2"] * max(s, 1e-12)] * 3 + [M["kappa_r2"] * max(phi, 1e-12)] * 3)
    return np.array(M["Sigma0"])


def score(pairs, models, Th):
    res = dict(n=len(pairs))
    if not pairs:
        return res
    qs, nl = [], {k: [] for k in ("G", "T", "D", "C")}
    cache = {}
    sd = {k: [] for k in ("G", "T", "D", "C")}
    for p in pairs:
        q, l = gauss3(p["d"], p["S"])
        qs.append(q)
        nl["G"].append(l)
        sd["G"].append(np.sqrt(max(p["S"][2, 2], 0)))
        for k in ("T", "D", "C"):
            S = p["J"] @ model_cov(k, models, p, Th, cache) @ p["J"].T
            nl[k].append(gauss3(p["d"], S)[1])
            sd[k].append(np.sqrt(max(S[2, 2], 0)))
    qs = np.array(qs)
    res.update(nis_mean=float(qs.mean()), nis_median=float(np.median(qs)), pct_out=float(100 * (qs > CHI2_3).mean()),
               nll={k: float(np.mean(v)) for k, v in nl.items()})
    res["_arrays"] = dict(k=np.array([p["k"] for p in pairs]), absd=np.array([abs(p["d"][2]) for p in pairs]),
                          nis=qs, **{f"sd_{k}": np.array(v) for k, v in sd.items()},
                          **{f"nll_{k}": np.array(v) for k, v in nl.items()})
    return res


# ------------------------------------------------------------------ main


def evaluate(pre, rule, photo, anchor, lam=None):
    F = build_variant(pre, rule, photo, anchor)
    nll, nis = local_nll(F, pre["eps"], LAMS)
    il = int(np.argmin(nll))
    lam_fit = float(LAMS[il])
    lam_use = lam_fit if lam is None else lam
    j = int(np.argmin(np.abs(np.log(LAMS) - np.log(lam_use))))
    rows, V, C, cinfo = recursion(F, pre, lam_use)
    pairs = erel_pairs(pre, rows, V, C)
    return dict(lam_fit=lam_fit, lam_edge=bool(il in (0, len(LAMS) - 1)), lam_used=lam_use,
                local_nis_mean=float(nis[j].mean()), local_nis_median=float(np.median(nis[j])),
                local_nll_mean=float(nll[j] / (len(F) - 1)), cinfo=cinfo), pairs


def main():
    os.makedirs(OUT0, exist_ok=True)
    S = dict(variants=[], runs={})
    t00 = time.time()
    # ---------------- snt_near: variants, lambda, reference models
    seq = "snt_near"
    R = {m: load_run(d) for m, d in RUNS[seq].items()}
    P = seq_pass(seq, R)
    pre = precompute_run(R["dec"], P, "dec", seq)
    print(f"snt_near precompute {time.time() - t00:.0f} s, sigma_c {P['sig_c']}", flush=True)
    best = None
    var_pairs = {}
    for rule in RULES:
        for photo in (False, True):
            for anchor in (False, True):
                ev, pairs = evaluate(pre, rule, photo, anchor)
                models = fit_models(pairs, pre["Th"]) if pairs else None
                sc = score(pairs, models, pre["Th"]) if pairs else dict(n=0)
                row = dict(rule=rule, photo=photo, anchor=anchor, **{k: v for k, v in ev.items()},
                           erel={k: v for k, v in sc.items() if k != "_arrays"})
                S["variants"].append(row)
                print(f"  variant {rule:12s} photo={photo!s:5s} anchor={anchor!s:5s}: lambda {ev['lam_fit']:.3g}"
                      f"{' (edge)' if ev['lam_edge'] else ''}, NIS6 {ev['local_nis_mean']:.2f}, E-rel NIS3 "
                      f"{sc.get('nis_mean', float('nan')):.2f}, NLL G {sc.get('nll', {}).get('G', float('nan')):.3f}", flush=True)
                key = sc.get("nll", {}).get("G", np.inf)
                if best is None or key < best[0]:
                    best = (key, rule, photo, anchor, ev["lam_fit"], models)
    _, rule, photo, anchor, lam, models = best
    S["locked"] = dict(rule=rule, photo=photo, anchor=anchor, lam=lam, models=models)
    print(f"LOCKED: rule {rule}, photo {photo}, anchor {anchor}, lambda {lam:.4g}", flush=True)
    # ---------------- every run with the locked variant
    todo = [("snt_near", R, P)]
    for seq2 in ("snt_far", "st_far", "room"):
        todo.append((seq2, None, None))
    for seq2, R2, P2 in todo:
        if R2 is None:
            R2 = {m: load_run(d) for m, d in RUNS[seq2].items()}
            P2 = seq_pass(seq2, R2)
        for m, Rm in R2.items():
            t0 = time.time()
            pre2 = pre if (seq2 == "snt_near" and m == "dec") else precompute_run(Rm, P2, m, seq2)
            anc = anchor and m != "base"
            ev, pairs = evaluate(pre2, rule, photo, anc, lam=lam)
            sc = score(pairs, models, pre2["Th"])
            arr = sc.pop("_arrays", None)
            if arr is not None:
                np.savez(os.path.join(OUT0, f"a12_pairs_{seq2}_{m}.npz"), **arr)
            own = np.mean([fr["n_own"][rule] / max(fr["n_px"], 1) for fr in pre2["frames"][1:]])
            S["runs"][f"{seq2}/{m}"] = dict(dir=Rm["dir"], sigma_c=pre2["sig_c"], n_frames=pre2["n"], n_kf=len(pre2["kf_idx"]),
                                           owned_frac=float(own), **ev, erel=sc)
            print(f"  {seq2}/{m}: lambda fit {ev['lam_fit']:.3g}{' (edge)' if ev['lam_edge'] else ''}, NIS6 "
                  f"{ev['local_nis_mean']:.2f}, E-rel n {sc['n']} NIS3 {sc.get('nis_mean', float('nan')):.2f} "
                  f"out {sc.get('pct_out', float('nan')):.1f} % NLL {sc.get('nll')}, {time.time() - t0:.0f} s", flush=True)
            json.dump(S, open(os.path.join(OUT0, "a12_summary.json"), "w"), indent=1, default=float)
    print(f"done {time.time() - t00:.0f} s")


if __name__ == "__main__":
    main()
