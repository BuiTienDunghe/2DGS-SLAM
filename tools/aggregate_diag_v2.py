"""Plan v2: aggregate D1/D2/D3 on the common evaluation set, decompositions, selection rule §5, figures.

usage: python tools/aggregate_diag_v2.py --diag results_exp/reports/v2/diag --out results_exp/reports/v2/agg
                                         [--configs A,B,...]   (default: every config present in the maps)
Pi* intersection: a Pi* pixel is kept for an event only if the rigid result and EVERY config in the set
still render both layers there (alpha > 0.95), as in P3. All tables read the same intersection.
Selection (TUM only, user decision 2026-10-02): C1 = S1-S4, S6 of plan v1 §8.2 on TUM; C2 median R_X >=
median R_A-1-ref + 5 pp; C3 (Replica) not applicable; C4 wrong-pair proxy <= 10 % in every event;
C5 no event worse than rigid; D3 configs also need an online time <= 120 s.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

ELIGIBLE = ["F1w", "F1t", "F1o", "F2", "F3", "F4", "ALL-1", "ALL-10", "D3-p2p", "D3-p2l", "D3-p2p-10", "D3-all"]
SWITCHES = {"F1w": 1, "F1t": 1, "F1o": 1, "F2": 1, "F3": 1, "F4": 1, "ALL-1": 4, "ALL-10": 4, "ALL-noF3": 3,
            "D3-p2p": 1, "D3-p2l": 1, "D3-p2p-10": 1, "D3-all": 2}
WP = {"ALL-10": 10.0, "D3-p2p-10": 10.0}
BINS = [("<10 mm", 0.0, 0.010), ("10-50 mm", 0.010, 0.050), ("50-100 mm", 0.050, 0.100)]


def load_rows(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[(r["dump"], r["config"])] = r  # last write wins (re-runs of one config)
    return rows


def med(x):
    return float(torch.median(x)) if x.numel() else None


PLAN_SET = ["A-1-ref", "O1", "O1-loose", "F1w", "F1t", "F1o", "F2", "F3", "F4", "ALL-1", "ALL-10",
            "D3-p2p", "D3-p2l", "D3-p2p-10", "D3-all"]


def per_event(store, configs, set_cfgs):
    """-> {cfg: {...}} metrics for one event; also 'rigid'. The intersection is taken over rigid and the
    configs in set_cfgs (the plan's configs); report-only extras (+conv twins, ALL-noF3) are evaluated on
    that set, restricted to their own valid pixels (paired with rigid on the same pixels)."""
    keep, gaps, dD, cov = {}, {c: [] for c in ["rigid"] + configs}, [], {c: [] for c in configs}
    own = {c: [] for c in configs}
    for u in store["pix"]:
        k = store["rigid"][u][1].clone()
        for c in set_cfgs:
            k &= store["configs"][c][u][1]
        keep[u] = k
        gaps["rigid"].append(store["rigid"][u][0][k])
        dD.append(store["dD"][u][k])
        for c in configs:
            gaps[c].append(store["configs"][c][u][0][k])
            cov[c].append(store["configs"][c][u][2][k])
            own[c].append(store["configs"][c][u][1][k])
    g = {c: torch.cat(v) for c, v in gaps.items()}
    dD = torch.cat(dD)
    ownv = {c: torch.cat(v) for c, v in own.items()}
    n_pi = int(sum(v.numel() for v in store["pix"].values()))
    out = {"n_pix": int(dD.numel()), "n_pi": n_pi,
           "alpha_loss": {c: 1.0 - sum(int(store["configs"][c][u][1].sum()) for u in store["pix"]) / max(1, n_pi)
                          for c in configs}}
    for c, x in g.items():
        if c != "rigid" and c not in set_cfgs:
            m = ownv[c]
            x = x[m]
            xr = g["rigid"][m]
            out[c] = {"median": med(x), "rigid_same_px": med(xr), "n": int(m.sum()),
                      "p90": float(torch.quantile(x.float(), 0.9)) if x.numel() else None,
                      "inlier_1cm": float((x < 0.01).float().mean()) if x.numel() else None}
            continue
        out[c] = {"median": med(x), "p90": float(torch.quantile(x.float(), 0.9)) if x.numel() else None,
                  "inlier_1cm": float((x < 0.01).float().mean()) if x.numel() else None}
    def mask_of(c):
        return ownv[c] if (c != "rigid" and c not in set_cfgs) else torch.ones_like(dD, dtype=torch.bool)

    bins = {}
    for name, lo, hi in BINS:
        m = (dD >= lo) & (dD < hi)
        b = {"n": int(m.sum()), "rigid": med(g["rigid"][m])}
        for c in configs:
            mm = m & mask_of(c)
            b[c] = {"n": int(mm.sum()), "rigid": med(g["rigid"][mm]), "cfg": med(g[c][mm])}
        bins[name] = b
    out["bins"] = bins
    covs = {}
    for c in configs:
        cv = torch.cat(cov[c])
        mk = mask_of(c)
        a, b = cv & mk, (~cv) & mk
        covs[c] = {"covered": {"n": int(a.sum()), "rigid": med(g["rigid"][a]), "cfg": med(g[c][a])},
                   "uncovered": {"n": int(b.sum()), "rigid": med(g["rigid"][b]), "cfg": med(g[c][b])}}
    out["coverage"] = covs
    return out


def rel_red(r, x):
    return None if r is None or x is None or r == 0 else (r - x) / r


def red(e, c):
    """R_X of config c in one event (vs rigid on exactly the same pixels)."""
    ref = e[c].get("rigid_same_px", e["rigid"]["median"]) if c != "rigid" else e["rigid"]["median"]
    return rel_red(ref, e[c]["median"])


def criteria(evs, rows, name, ref="A-1-ref"):
    """evs: [(dump, per_event dict)]; rows: {(dump, cfg): row}. Returns dict of (value, pass)."""
    R = [red(e, name) for _, e in evs]
    R = [x for x in R if x is not None]
    Rref = [red(e, ref) for _, e in evs]
    Rref = [x for x in Rref if x is not None]
    out = {}
    medR = float(np.median(R)) if R else None
    out["S1"] = (medR, medR is not None and medR >= 0.20)
    out["S2"] = (float(min(R)) if R else None, bool(R) and min(R) >= -0.10)
    worst = []
    dps = []
    for dump, _ in evs:
        a, b = rows[(dump, name)], rows[(dump, "rigid")]
        if a.get("ate") is not None and b.get("ate") is not None:
            worst.append((a["ate"] - b["ate"]) - max(0.05 * b["ate"], 0.001))
        if a.get("psnr") is not None and b.get("psnr") is not None:
            dps.append(a["psnr"] - b["psnr"])
    out["S3"] = (float(max(worst)) if worst else None, bool(worst) and max(worst) <= 0)
    out["S4"] = (float(np.median(dps)) if dps else None, bool(dps) and float(np.median(dps)) >= -0.3)
    fb = [0.0 if rows[(d, name)].get("accepted") else 1.0 for d, _ in evs]
    out["S6"] = (float(np.mean(fb)) if fb else None, bool(fb) and float(np.mean(fb)) <= 0.5)
    medRref = float(np.median(Rref)) if Rref else None
    out["C2"] = ((medR - medRref) if (medR is not None and medRref is not None) else None,
                 medR is not None and medRref is not None and medR >= medRref + 0.05)
    px = [rows[(d, name)].get("frac_res_gt30mm") for d, _ in evs]
    px = [x for x in px if x is not None]
    out["C4"] = (float(max(px)) if px else None, (not px) or max(px) <= 0.10)
    out["C5"] = (float(min(R)) if R else None, bool(R) and min(R) >= 0.0)
    if name.startswith("D3"):
        tt = [(rows[(d, name)].get("t_stage_s") or {}).get("total") for d, _ in evs]
        tt = [x for x in tt if x is not None]
        out["T120"] = (float(max(tt)) if tt else None, (not tt) or max(tt) <= 120.0)
    return out


def select(table, R_med):
    passed = [n for n in ELIGIBLE if n in table and all(p for _, p in table[n].values())]
    if not passed:
        return None, passed
    best = max(R_med[n] for n in passed)
    near = [n for n in passed if R_med[n] >= best - 0.02]
    near.sort(key=lambda n: (SWITCHES.get(n, 9), -WP.get(n, 1.0), -R_med[n]))
    return near[0], passed


def figures(out, evs, rows, configs, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    labels = [f"{e['_ev']}" for _, e in evs]
    # D1 bars
    d1 = [c for c in ("A-1-ref", "A-1-ref+conv", "O1", "O1-loose") if c in configs]
    if "O1" in configs:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        names = ["rigid"] + d1
        w = 0.8 / len(names)
        for i, n in enumerate(names):
            ax.bar(np.arange(len(evs)) + i * w, [1e3 * e[n]["median"] for _, e in evs], w, label=n)
        ax.set_xticks(np.arange(len(evs)) + 0.4 - w / 2, labels)
        ax.set_ylabel("e_dl median on Pi* [mm]")
        ax.legend(fontsize=8)
        ax.set_title("D1: rigid / A-1 / oracle per TUM event")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "D1_bars.png"), dpi=130)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        bn = [b[0] for b in BINS]
        for i, n in enumerate(d1):
            vals = []
            for b in bn:
                rr = [rel_red(e["bins"][b][n]["rigid"], e["bins"][b][n]["cfg"]) for _, e in evs if e["bins"][b][n]["n"] > 0]
                rr = [x for x in rr if x is not None]
                vals.append(100 * float(np.median(rr)) if rr else np.nan)
            ax.bar(np.arange(len(bn)) + i * 0.8 / len(d1), vals, 0.8 / len(d1), label=n)
        ax.set_xticks(np.arange(len(bn)) + 0.4 - 0.4 / len(d1), bn)
        ax.set_ylabel("median reduction vs rigid [%]")
        ax.set_xlabel("rigid |D_new - D_old| bin")
        ax.axhline(0, c="k", lw=0.6)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "D1_bins.png"), dpi=130)
        plt.close(fig)
    # D2 paired + pairs vs gain (+ D3 points)
    d2 = [c for c in configs if c in SWITCHES and not c.startswith("D3")] + (["A-1-ref"] if "A-1-ref" in configs else [])
    d3 = [c for c in configs if c.startswith("D3") and "+conv" not in c]
    if len(d2) > 1:
        names = ["A-1-ref"] + [c for c in d2 if c != "A-1-ref"]
        fig, ax = plt.subplots(figsize=(1.0 * len(names) + 2, 4))
        for i, n in enumerate(names):
            for _, e in evs:
                r0 = e[n].get("rigid_same_px", e["rigid"]["median"])
                ax.plot([i - 0.25, i + 0.25], [1e3 * r0, 1e3 * e[n]["median"]], "-o", ms=3, c="tab:blue", alpha=0.7)
        ax.set_xticks(range(len(names)), names, rotation=30)
        ax.set_ylabel("e_dl median on Pi* [mm] (left rigid, right config)")
        ax.set_title("D2: paired per TUM event")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "D2_paired.png"), dpi=130)
        plt.close(fig)
        for fname, extra in (("D2_pairs_vs_gain.png", []), ("D3_vs_D2.png", d3)):
            if fname == "D3_vs_D2.png" and not d3:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for n in names + extra:
                xs = [rows[(d, n)].get("corr_capped") or 0 for d, _ in evs]
                ys = [100 * (red(e, n) or 0) for _, e in evs]
                ax.scatter(xs, ys, s=18, marker="^" if n.startswith("D3") else "o", label=n)
            ax.set_xscale("symlog")
            ax.set_xlabel("pairs after cap (per event)")
            ax.set_ylabel("R_X = reduction of e_dl vs rigid [%]")
            ax.axhline(0, c="k", lw=0.6)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, fname), dpi=130)
            plt.close(fig)
    # funnel
    fun = [c for c in ("A-1-ref", "ALL-1") if c in configs]
    if fun:
        steps = ["raw", "alpha", "dist", "normal", "S", "edge", "capped"]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for n in fun:
            vals = []
            for s in steps:
                v = [(rows[(d, n)].get("corr_capped") if s == "capped" else (rows[(d, n)].get("funnel") or {}).get(s)) for d, _ in evs]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else np.nan)
            ax.plot(steps, vals, "-o", label=n)
        ax.set_yscale("log")
        ax.set_ylabel("pairs remaining (mean over TUM events)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "D2_funnel.png"), dpi=130)
        plt.close(fig)
    # D3 H magnitudes
    if d3:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        seen = set()
        for d, _ in evs:
            for k in rows[(d, d3[0])].get("reg_kfs") or []:
                c = "tab:green" if k["passed"] else "tab:red"
                lab = "pass" if k["passed"] else "fail"
                ax.scatter(k["t_mm"], k["rot_deg"], c=c, s=14, label=None if lab in seen else lab)
                seen.add(lab)
        ax.axvline(50, c="k", ls="--", lw=0.8)
        ax.axhline(2, c="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("|t(H_j)| [mm] (camera frame)")
        ax.set_ylabel("rotation of H_j [deg]")
        ax.set_title(f"D3 registration per loop keyframe ({d3[0]}; dashed = Q4)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "D3_H_magnitudes.png"), dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--set", default=None, help="configs defining the intersection (default: the plan's configs present)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_rows(os.path.join(a.diag, "rows.jsonl"))
    evs = []
    configs = None
    for mp in sorted(glob.glob(os.path.join(a.diag, "maps", "*.pt"))):
        store = torch.load(mp, weights_only=False)
        dump = os.path.basename(mp)[:-3]
        cs = a.configs.split(",") if a.configs else sorted(store["configs"])
        configs = cs if configs is None else [c for c in configs if c in cs]
        evs.append((dump, store))
    order = PLAN_SET + sorted(c for c in configs if c not in PLAN_SET)
    configs = [c for c in order if c in configs]
    set_cfgs = a.set.split(",") if a.set else [c for c in PLAN_SET if c in configs]
    res_ev = []
    for dump, store in evs:
        e = per_event(store, configs, set_cfgs)
        r0 = rows[(dump, "rigid")]
        e["_ev"] = f"{r0['event_id']}"
        e["_dump"] = dump
        res_ev.append((dump, e))
    R = {c: [red(e, c) for _, e in res_ev] for c in configs}
    R_med = {c: float(np.median([x for x in v if x is not None])) for c, v in R.items() if any(x is not None for x in v)}
    G = {}
    if "O1" in configs:
        for c in configs:
            gs = []
            for _, e in res_ev:
                den = e["rigid"]["median"] - e["O1"]["median"]
                rx = red(e, c)
                ro = red(e, "O1")
                gs.append(None if abs(den) < 1e-4 or rx is None or not ro else rx / ro)
            G[c] = gs
    table = {c: criteria(res_ev, rows, c) for c in configs if c not in ("A-1-ref",) and not c.startswith("O1")}
    chosen, passed = select(table, R_med)
    summary = {"configs": configs, "events": [e["_dump"] for _, e in res_ev], "R": R, "R_median": R_med, "G": G,
               "criteria": table, "eligible_passed": passed, "chosen": chosen,
               "per_event": {e["_dump"]: e for _, e in res_ev},
               "rows": {f"{d}|{c}": rows[(d, c)] for d, _ in res_ev for c in ["rigid"] + configs if (d, c) in rows}}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    figures(a.out, res_ev, rows, configs, R)
    # console digest
    print("events:", [e["_ev"] for _, e in res_ev], "pixels in intersection:", [e["n_pix"] for _, e in res_ev],
          "of", [e["n_pi"] for _, e in res_ev])
    print(f"{'config':16s} " + " ".join(f"{'ev' + e['_ev']:>16s}" for _, e in res_ev) + f" {'median R':>9s} {'G':>14s}")
    print("intersection set:", set_cfgs)
    for c in ["rigid"] + configs:
        cells = []
        for _, e in res_ev:
            x = e[c]["median"]
            rr = red(e, c)
            cells.append(f"{1e3 * x:7.2f}mm {('' if rr is None else f'{100 * rr:+5.1f}%'):>7s}")
        gtxt = "" if c not in G else ",".join("-" if g is None else f"{g:.2f}" for g in G[c])
        al = "" if c == "rigid" else ",".join(f"{100 * e['alpha_loss'][c]:.0f}" for _, e in res_ev)
        print(f"{c:16s} " + " ".join(f"{x:>16s}" for x in cells) + f" {100 * R_med.get(c, 0):8.1f}% {gtxt:>14s}  alpha-loss% {al}")
    for c, t in table.items():
        print(f"{c:16s}", {k: (None if v[0] is None else round(v[0], 4), v[1]) for k, v in t.items()})
    print("eligible passed:", passed, "-> chosen:", chosen)


if __name__ == "__main__":
    main()
