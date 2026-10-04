"""P3: offline deformation grid on the tuning dumps (runs #1, #2), metrics vs rigid, selection rule §8.2.

usage: python tools/run_offline_deform.py --out results_exp/reports/P3 [--freeze configs/deform/selected.yaml]
                                          [--grid main|min] [--override YAML] RUN_DIR [RUN_DIR ...]
Writes events.jsonl, P3_metrics.json, fig/P3_*.png and (with --freeze) the selected config.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform.dump import EventState, gauss_to_dev, list_dumps, load_dump  # noqa: E402
from deform.metrics import evaluate_event  # noqa: E402
from deform.pipeline import correct_map, inp_from_dump, with_pos  # noqa: E402

GRID_MAIN = {  # name: (variant, w_p, w_con)
    "A-1": ("A", 1.0, 1.0), "A-10": ("A", 10.0, 1.0), "A-100": ("A", 100.0, 1.0),
    "B-1": ("B", 1.0, 1.0), "B-10": ("B", 10.0, 1.0), "B-100": ("B", 100.0, 1.0),
    "A-noCon": ("A", 10.0, 0.0), "B-noCon": ("B", 10.0, 0.0),
}
GRID_MIN = {k: GRID_MAIN[k] for k in ("A-10", "B-10", "B-noCon", "A-noCon")}
CANDIDATES = ("A-1", "A-10", "A-100", "B-1", "B-10", "B-100")


def mesh_path_for(st, root):
    if st.dtype_name != "replica":
        return None
    p = os.path.join(root, "datasets", "replica", f'{st.config["Dataset"]["sequence_name"]}_mesh.ply')
    return p if os.path.exists(p) else None


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def run_event(path, grid, root, override, keep_maps):
    d = load_dump(path)
    st = EventState(d)
    inp = inp_from_dump(d, st.frame)
    base = dcfg.resolve(override, st.config)
    base["seed"] = int(st.meta.get("seed", 0))
    cache = {}
    results = {}
    for name, (var, wp, wc) in grid.items():
        cfg = copy.deepcopy(base)
        cfg["energy"]["w_p"], cfg["energy"]["w_con"] = wp, wc
        results[name] = correct_map(inp, cfg, var, cache)
    first = next(iter(results.values()))
    methods = {"pre": (gauss_to_dev(st.gpre), st.poses_pre), "rigid": (first["G_rig"], first["poses_rig"])}
    for name, r in results.items():
        methods[name] = (with_pos(inp["g"], r["xyz"], r["rot"]), r["poses"])
    ev = evaluate_event(st, methods, "rigid", base, mesh_path=mesh_path_for(st, root), seed=base["seed"],
                        extras=keep_maps)
    xyz_r = first["G_rig"]["xyz"]
    row = {"dump": os.path.basename(path), "run": os.path.dirname(os.path.dirname(path)), "scene": st.meta["scene"],
           "event_id": st.meta["event_id"], "cur_uid": st.meta["cur_uid"], "loop_uid": st.meta["loop_uid"],
           "J_L": ev["J_L"], "J_eval": ev["J_eval"], "n_eval_pix": ev["n_eval_pix"], "n_B": ev["n_B"],
           "methods": {}, "logs": {}}
    for k, m in ev["methods"].items():
        row["methods"][k] = {"edl": g(m, "e_dl", "median"), "edl_p90": g(m, "e_dl", "p90"),
                             "edl_inlier": g(m, "e_dl", "inlier_1cm"), "estep": g(m, "e_step", "e_step"),
                             "estep_p90B": g(m, "e_step", "p90_B"), "psnr": g(m, "render", "psnr"),
                             "depth_l1": g(m, "render", "depth_l1"), "ate": m.get("ate_kf"),
                             "acc": g(m, "acc", "acc_median"), "acc_gt1cm": g(m, "acc", "frac_gt_1cm")}
    for name, r in results.items():
        lg = r["log"]
        row["logs"][name] = {"accepted": r["accepted"], "reason": r["reason"],
                             "max_disp_vs_rigid_mm": 1e3 * float((r["xyz"] - xyz_r).norm(dim=1).max()),
                             **{k: lg.get(k) for k in ("n_nodes", "n_filler", "n_edges", "n_loop_kfs", "n_opt_kfs",
                                                       "n_eval_kfs", "corr_raw", "corr_gated", "corr_capped",
                                                       "gate_frac", "E_init", "E_final", "lbfgs_iters",
                                                       "edl_opt_init_mm", "edl_opt_final_mm", "max_node_disp_m",
                                                       "max_node_rot_deg", "pose_sync", "stretch", "t_stage_s",
                                                       "reliability", "delta_t")}}
    maps = None
    if keep_maps and "_maps" in ev:
        maps = {"maps": ev["_maps"], "Pi": ev["_Pi"]}
    del methods, results, cache, inp
    torch.cuda.empty_cache()
    return row, maps


def rel(a, b):
    return None if a is None or b is None or b == 0 else (a - b) / b


def criteria(rows, name, scene_is_replica):
    """§8.2 S1-S6 for one scene and one config. Returns dict of (value, pass)."""
    ev = [r for r in rows if r["J_eval"]]
    out = {}
    rc = [rel(r["methods"][name]["edl"], r["methods"]["rigid"]["edl"]) for r in ev]
    rc = [x for x in rc if x is not None]
    out["S1"] = (float(np.median(rc)) if rc else None, bool(rc) and float(np.median(rc)) <= -0.20)
    out["S2"] = (float(max(rc)) if rc else None, bool(rc) and max(rc) <= 0.10)
    worst = []
    for r in ev:
        a, b = r["methods"][name]["ate"], r["methods"]["rigid"]["ate"]
        if a is not None and b is not None:
            worst.append((a - b) - max(0.05 * b, 0.001))
    out["S3"] = (float(max(worst)) if worst else None, bool(worst) and max(worst) <= 0)
    dp = [r["methods"][name]["psnr"] - r["methods"]["rigid"]["psnr"] for r in ev
          if r["methods"][name]["psnr"] is not None and r["methods"]["rigid"]["psnr"] is not None]
    out["S4"] = (float(np.median(dp)) if dp else None, bool(dp) and float(np.median(dp)) >= -0.3)
    if scene_is_replica:
        ra = [rel(r["methods"][name]["acc"], r["methods"]["rigid"]["acc"]) for r in ev]
        ra = [x for x in ra if x is not None]
        out["S5"] = (float(np.median(ra)) if ra else None, (not ra) or float(np.median(ra)) <= 0.05)
    fb = [0.0 if r["logs"][name]["accepted"] else 1.0 for r in rows]
    out["S6"] = (float(np.mean(fb)) if fb else None, bool(fb) and float(np.mean(fb)) <= 0.5)
    return out


def select(rows):
    scenes = sorted({r["scene"] for r in rows if any(x["J_eval"] for x in rows if x["scene"] == r["scene"])})
    table = {}
    for name in CANDIDATES:
        if name not in rows[0]["methods"]:
            continue
        table[name] = {sc: criteria([r for r in rows if r["scene"] == sc], name, sc.startswith("replica"))
                       for sc in scenes}
    # a config passes only if at least one scene is evaluable and every criterion holds in every evaluable scene
    passed = {n: bool(t) and all(all(p for _, p in crit.values()) for crit in t.values()) for n, t in table.items()}

    def med_edl(name, sc):
        v = [r["methods"][name]["edl"] for r in rows if r["scene"] == sc and r["J_eval"] and r["methods"][name]["edl"] is not None]
        return float(np.median(v)) if v else np.inf

    def pick(names):
        if not names:
            return None
        ranks = {n: 0.0 for n in names}
        for sc in scenes:
            order = sorted(names, key=lambda n: med_edl(n, sc))
            for i, n in enumerate(order):
                ranks[n] += i / max(1, len(scenes))
        return sorted(names, key=lambda n: (ranks[n], -float(n.split("-")[1])))[0]

    rule = None
    chosen = pick([n for n in table if n.startswith("B") and passed[n]])
    if chosen:
        rule = "8.2-1 (V-B passes)"
    else:
        chosen = pick([n for n in table if n.startswith("A") and passed[n]])
        rule = "8.2-2 (V-A passes)" if chosen else None
    if not chosen:
        prio = ["S3", "S2", "S1", "S4", "S5", "S6"]

        def score(n):
            cnt = sum(p for t in table[n].values() for _, p in t.values())
            pr = tuple(sum(t.get(s, (None, True))[1] for t in table[n].values()) for s in prio)
            return (cnt, pr, float(n.split("-")[1]))
        chosen = sorted(table, key=score, reverse=True)[0] if table else None
        rule = "8.2-3 fallback (no config passes every criterion)" if scenes else "none: no evaluable event in any scene"
    return chosen, rule, table, passed, scenes


def make_figs(out, rows, chosen, heat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in rows[0]["methods"] if n not in ("pre", "rigid")]
    for metric, fname, lab in (("edl", "P3_paired_edl.png", "e_dl median [mm]"),
                               ("estep", "P3_paired_estep.png", "e_step [mm]")):
        fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 4))
        for i, n in enumerate(names):
            for r in rows:
                a, b = r["methods"]["rigid"][metric], r["methods"][n][metric]
                if a is None or b is None:
                    continue
                c = "tab:blue" if r["scene"].startswith("tum") else "tab:orange"
                ax.plot([i - 0.25, i + 0.25], [1e3 * a, 1e3 * b], "-o", ms=3, c=c, alpha=0.7)
        ax.set_xticks(range(len(names)), names, rotation=30)
        ax.set_ylabel(lab + "  (left: rigid, right: config)")
        ax.set_title("paired per event (blue TUM, orange Replica)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig", fname), dpi=130)
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, metric in zip(axs, ("edl", "estep")):
        for var in ("A", "B"):
            wps, vals = [], []
            for wp in (1, 10, 100):
                n = f"{var}-{wp}"
                if n not in rows[0]["methods"]:
                    continue
                v = [rel(r["methods"][n][metric], r["methods"]["rigid"][metric]) for r in rows if r["J_eval"]]
                v = [x for x in v if x is not None]
                if v:
                    wps.append(wp)
                    vals.append(100 * float(np.median(v)))
            ax.plot(wps, vals, "-o", label=f"V-{var}")
        ax.set_xscale("log")
        ax.axhline(0, c="k", lw=0.6)
        ax.set_xlabel("w_p")
        ax.set_ylabel(f"median change of {metric} vs rigid [%]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig", "P3_wp_sweep.png"), dpi=130)
    plt.close(fig)

    if heat is not None and chosen in heat["maps"]:
        uid = sorted(heat["Pi"].keys())[0]
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, k in zip(axs, ("rigid", chosen)):
            im = ax.imshow(np.clip(1e3 * heat["maps"][k][uid]["gap"].cpu().numpy(), 0, 30), cmap="magma")
            ax.set_title(f"{heat['scene']} ev {heat['event']} kf {uid}: |gap| {k} [mm]")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig", "P3_heatmap_before_after.png"), dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--grid", default="main", choices=("main", "min"))
    ap.add_argument("--override", default=None, help="YAML with parameter changes (one repair round, §8.2-4)")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(a.out, "fig"), exist_ok=True)
    override = {}
    if a.override:
        with open(a.override) as f:
            override = yaml.safe_load(f) or {}
    grid = GRID_MAIN if a.grid == "main" else GRID_MIN
    rows, heat, heat_e = [], None, -1
    t0 = time.time()
    for rd in a.runs:
        for p in list_dumps(rd):
            row, maps = run_event(p, grid, root, override, keep_maps=True)
            rows.append(row)
            e = row["methods"]["rigid"]["edl"]
            if maps is not None and e is not None and e > heat_e:
                heat_e, heat = e, {**maps, "scene": row["scene"], "event": row["event_id"]}
            short = {n: (None if row["methods"][n]["edl"] is None else round(1e3 * row["methods"][n]["edl"], 2))
                     for n in row["methods"]}
            acc = {n: row["logs"][n]["accepted"] for n in row["logs"]}
            print(json.dumps({"scene": row["scene"], "ev": row["event_id"], "edl_mm": short, "accepted": acc,
                              "t_total_s": round(time.time() - t0)}), flush=True)
    with open(os.path.join(a.out, "events.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if not rows:
        print("NO_DUMPS")
        return
    sanity = max(r["logs"]["A-noCon"]["max_disp_vs_rigid_mm"] for r in rows) if "A-noCon" in grid else None
    chosen, rule, table, passed, scenes = select(rows)
    worse_all = all(
        np.median([rel(r["methods"][chosen]["edl"], r["methods"]["rigid"]["edl"]) for r in rows
                   if r["scene"] == sc and r["J_eval"] and r["methods"][chosen]["edl"] is not None]) > 0
        for sc in scenes) if scenes else False
    res = {"chosen": chosen, "rule": rule, "criteria": table, "passed": passed, "scenes_evaluable": scenes,
           "A_noCon_max_disp_mm": sanity, "G3_block_worse_in_all_scenes": bool(worse_all),
           "n_events": len(rows), "time_s": time.time() - t0, "override": override}
    make_figs(a.out, rows, chosen, heat)
    if a.freeze and chosen:
        var, wp, wc = GRID_MAIN[chosen]
        body = {"mode": "deform", "variant": var, "energy": {"w_p": wp, "w_con": wc}}
        body.update({k: v for k, v in override.items() if k not in ("mode", "variant")})
        if "energy" in override:
            body["energy"] = {**override["energy"], "w_p": wp, "w_con": wc}
        txt = (f"# FROZEN by tools/run_offline_deform.py {time.strftime('%Y-%m-%d %H:%M')} (P3, rule {rule})\n"
               f"# chosen config: {chosen}; tuning set: {', '.join(a.runs)}\n" + yaml.safe_dump(body, sort_keys=False))
        with open(a.freeze, "w") as f:
            f.write(txt)
        res["selected_yaml"] = a.freeze
        res["selected_sha256"] = hashlib.sha256(txt.encode()).hexdigest()
    with open(os.path.join(a.out, "P3_metrics.json"), "w") as f:
        json.dump({"result": res, "events": rows}, f, indent=1)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
