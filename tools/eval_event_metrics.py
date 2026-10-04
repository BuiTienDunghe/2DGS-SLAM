"""P1: per-event metrics (M1-M5) of the "pre" state and of the rigid correction, for every dump.

usage: python tools/eval_event_metrics.py --out results_exp/reports/P1 RUN_DIR [RUN_DIR ...]
Writes events.jsonl, P1_metrics.json and figures fig/P1_*.png.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform.dump import EventState, gauss_to_dev, list_dumps, load_dump  # noqa: E402
from deform.metrics import dD_hist_values, evaluate_event, layer_masks  # noqa: E402
from deform.rigid import replay  # noqa: E402


def mesh_path_for(st, root):
    if st.dtype_name != "replica":
        return None
    p = os.path.join(root, "datasets", "replica", f'{st.config["Dataset"]["sequence_name"]}_mesh.ply')
    return p if os.path.exists(p) else None


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(a.out, "fig"), exist_ok=True)
    rows, hist = [], {}
    worst = None
    for rd in a.runs:
        for p in list_dumps(rd):
            t0 = time.time()
            d = load_dump(p)
            st = EventState(d)
            cfg = dcfg.resolve({}, st.config)
            with torch.no_grad():
                G_pre = gauss_to_dev(st.gpre)
                xyz_r, rot_r, _ = replay(d, use_online=False)  # clean fp32 rigid reference
                G_rig = gauss_to_dev(st.gpre, xyz=xyz_r, rot=rot_r)
                # TF32 noise floor of the online baseline: dumped post vs clean replay
                nf = (d["gauss_post"]["xyz"].to(xyz_r.device) - xyz_r).norm(dim=1)
                tf32_floor = {"median_mm": 1e3 * float(nf.median()), "p90_mm": 1e3 * float(torch.quantile(nf[:1000000], 0.9)),
                              "max_mm": 1e3 * float(nf.max())}
                methods = {"pre": (G_pre, st.poses_pre), "rigid": (G_rig, st.poses_pgo)}
                ev = evaluate_event(st, methods, "rigid", cfg, mesh_path=mesh_path_for(st, root),
                                    seed=int(st.meta.get("seed", 0)), extras=True)
                m_old, m_new = layer_masks(st.active)
                if ev["J_L"]:
                    hv = dD_hist_values(st, G_rig, st.poses_pgo, ev["J_L"], m_old, m_new, cfg["corr"]["stride"])
                    hist.setdefault(st.meta["scene"], []).append(hv)
            ev.update({"run": rd, "dump": os.path.basename(p), "scene": st.meta["scene"],
                       "event_id": st.meta["event_id"], "cur_uid": st.meta["cur_uid"],
                       "loop_uid": st.meta["loop_uid"], "n_gauss": st.N,
                       "n_old": int((~st.active).sum()), "n_new": int(st.active.sum()),
                       "delta_t": st.delta_t(), "time_s": time.time() - t0, "tf32_floor": tf32_floor})
            e = (ev["methods"]["rigid"]["e_dl"] or {}).get("median")
            if e is not None and (worst is None or e > worst[0]) and "_maps" in ev:
                uid = sorted(ev["_maps"]["rigid"].keys())[0]
                worst = (e, ev["scene"], ev["event_id"], uid,
                         ev["_maps"]["rigid"][uid]["gap"].cpu().numpy(), ev["_Pi"][uid][0].cpu().numpy(),
                         ev["_maps"]["pre"][uid]["gap"].cpu().numpy())
            rows.append(strip(ev))
            r, pr = ev["methods"]["rigid"], ev["methods"]["pre"]
            print(json.dumps({"scene": ev["scene"], "ev": ev["event_id"], "c": ev["cur_uid"], "h": ev["loop_uid"],
                              "JL": ev["n_loop_kfs"], "Jeval": len(ev["J_eval"]), "pix": ev["n_eval_pix"],
                              "edl_pre": (pr["e_dl"] or {}).get("median"), "edl_rigid": (r["e_dl"] or {}).get("median"),
                              "estep_rigid": (r["e_step"] or {}).get("e_step"), "t": round(ev["time_s"], 1)}),
                  flush=True)
            del G_pre, G_rig
            torch.cuda.empty_cache()
    with open(os.path.join(a.out, "events.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    make_figs(a.out, rows, hist, worst)
    summary = summarize(rows)
    with open(os.path.join(a.out, "P1_metrics.json"), "w") as f:
        json.dump({"summary": summary, "events": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


def summarize(rows):
    out = {}
    for sc in sorted({r["scene"] for r in rows}):
        rr = [r for r in rows if r["scene"] == sc]
        def col(m, k1, k2):
            v = [((r["methods"][m].get(k1) or {}).get(k2)) for r in rr]
            return [x for x in v if x is not None]
        out[sc] = {
            "n_events": len(rr),
            "n_events_with_eval": sum(1 for r in rr if r["J_eval"]),
            "edl_rigid_median_of_events_mm": 1e3 * float(np.median(col("rigid", "e_dl", "median"))) if col("rigid", "e_dl", "median") else None,
            "edl_pre_median_of_events_mm": 1e3 * float(np.median(col("pre", "e_dl", "median"))) if col("pre", "e_dl", "median") else None,
            "estep_rigid_median_of_events_mm": 1e3 * float(np.median(col("rigid", "e_step", "e_step"))) if col("rigid", "e_step", "e_step") else None,
        }
    return out


def make_figs(out, rows, hist, worst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rows) + 2), 3.5))
    lab = [f'{r["scene"].split("/")[-1]}#{r["event_id"]}' for r in rows]
    pre = [1e3 * ((r["methods"]["pre"]["e_dl"] or {}).get("median") or np.nan) for r in rows]
    rig = [1e3 * ((r["methods"]["rigid"]["e_dl"] or {}).get("median") or np.nan) for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, pre, 0.4, label="pre (before fix)")
    ax.bar(x + 0.2, rig, 0.4, label="rigid (baseline)")
    ax.set_xticks(x, lab, rotation=60, fontsize=7)
    ax.set_ylabel("e_dl median [mm]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig", "P1_events_edl.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for sc, hv in hist.items():
        v = np.concatenate(hv) if hv else np.zeros(0)
        if v.size:
            ax.hist(np.clip(v * 1e3, 0, 300), bins=100, histtype="step", label=sc, density=True)
    ax.axvline(50, ls="--", c="k", lw=0.8)
    ax.set_xlabel("|D_new - D_old| on loop keyframes [mm] (clipped 300)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig", "P1_hist_dD.png"), dpi=130)
    plt.close(fig)

    if worst is not None:
        e, sc, ev, uid, gap_r, pi, gap_p = worst
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for axx, g, t in ((axs[0], gap_p, "pre"), (axs[1], gap_r, "rigid")):
            im = axx.imshow(np.clip(g * 1e3, 0, 50), cmap="magma")
            axx.set_title(f"{sc} event {ev} kf {uid}: |gap| {t} [mm]")
            axx.axis("off")
            fig.colorbar(im, ax=axx, fraction=0.03)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig", "P1_heatmap_example.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
