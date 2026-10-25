"""PLAN A2G-P1 gen §3.5 analysis (report only, never used for a decision): why the gate result is what it is.

1. lambda beyond the plan grid on snt_near (where is the E-local optimum?).
2. Every plane run: E-rel with lambda fitted on that run (E-local) and T / D / C refitted on that run (in-sample
   "oracle" scales) -> does the genealogy STRUCTURE beat the reference models once the scale is right?
3. Figure: snt_far, predicted sd of the offset row of d (G, T, D, C, locked) vs true |d|.
usage: python tools/a2g_gen/stage0_diag.py     -> stage0/a12_diag.json, stage0/sd_vs_absd_snt_far.png
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage0_genealogy as SG  # noqa: E402
from common import OUT0, RUNS, load_run  # noqa: E402


def main():
    S = json.load(open(os.path.join(OUT0, "a12_summary.json")))
    L = S["locked"]
    rule, photo, anchor = L["rule"], L["photo"], L["anchor"]
    D = dict(locked=dict(rule=rule, photo=photo, anchor=anchor, lam=L["lam"]), runs={})
    for seq in ("snt_near", "snt_far", "st_far", "room"):
        R = {m: load_run(d) for m, d in RUNS[seq].items() if m != "base"}
        P = SG.seq_pass(seq, R)
        for m, Rm in R.items():
            pre = SG.precompute_run(Rm, P, m, seq)
            F = SG.build_variant(pre, rule, photo, anchor and m != "base")
            out = {}
            if seq == "snt_near":
                lams = 10.0 ** np.arange(-1.0, 9.0 + 1e-9, 0.1)
                nll, nis = SG.local_nll(F, pre["eps"], lams)
                j = int(np.argmin(nll))
                out["lambda_extended"] = dict(grid="1e-1..1e9", lam_opt=float(lams[j]), nis6_mean=float(nis[j].mean()),
                                              nll_at_1e5=float(nll[np.argmin(np.abs(lams - 1e5))] / (len(F) - 1)),
                                              nll_at_opt=float(nll[j] / (len(F) - 1)))
            nll, nis = SG.local_nll(F, pre["eps"], SG.LAMS)
            lam_run = float(SG.LAMS[int(np.argmin(nll))])
            rows, V, C, _ = SG.recursion(F, pre, lam_run)
            pairs = SG.erel_pairs(pre, rows, V, C)
            models_run = SG.fit_models(pairs, pre["Th"])
            sc = SG.score(pairs, models_run, pre["Th"])
            sc.pop("_arrays", None)
            out["own_scale"] = dict(lam_run=lam_run, **sc)
            D["runs"][f"{seq}/{m}"] = out
            print(f"{seq}/{m}: lambda_run {lam_run:.3g}: NIS3 {sc['nis_mean']:.2f} out {sc['pct_out']:.1f} % NLL {sc['nll']}"
                  + (f" | extended lambda opt {out['lambda_extended']['lam_opt']:.3g}" if "lambda_extended" in out else ""),
                  flush=True)
            json.dump(D, open(os.path.join(OUT0, "a12_diag.json"), "w"), indent=1)
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, m in zip(axs, ("dec", "core")):
        p = os.path.join(OUT0, f"a12_pairs_snt_far_{m}.npz")
        if not os.path.exists(p):
            continue
        a = np.load(p)
        k = a["k"]
        ax.scatter(k, 100 * a["absd"], s=3, c="k", alpha=0.35, label="|d| thật (hàng offset)")
        for name, col in (("G", "C0"), ("T", "C1"), ("D", "C2"), ("C", "C3")):
            o = np.argsort(k)
            ax.plot(k[o], 100 * a[f"sd_{name}"][o], ".", ms=2, color=col, label=f"sd dự đoán {name}")
        ax.set_yscale("log")
        ax.set_title(f"snt_far / {'E1c' if m == 'dec' else 'P1 core'}")
        ax.set_xlabel("frame track")
    axs[0].set_ylabel("cm")
    axs[0].legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT0, "sd_vs_absd_snt_far.png"), dpi=110)
    print("done")


if __name__ == "__main__":
    main()
