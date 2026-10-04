"""plan v4 K2 (local stability of the hash cap) on one dump.

usage: python tools/k2_probe.py DEFORM_YAML DUMP.pt --out DIR [--n-k2a 100] [--n-sel 5] [--n-unsel 5]
K2a (selection only, no solve): remove one random gated pair, re-run the per-node cap, count the nodes whose
    kept set changed. Rule (user, 2026-10-03): max <= 1 node per trial, else STOP.
K2b (solve): remove n_sel random SELECTED pairs and n_unsel random UNSELECTED pairs (one at a time) from the
    gated set, re-run stages [3]-[9] with the cached reliability, compare node translations with the reference:
    max / p50 / p90 |d t| over nodes, by graph-hop ring from the node that lost the pair, and e_dl / acceptance.
    Rule: <= 1e-4 m is the plan's pass; a miss is a warning and the distribution is reported.
Writes DIR/k2_<dump>.json.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from deform import config as dcfg  # noqa: E402
from deform import pipeline as PL  # noqa: E402
from deform.correspondences import HASH_SALT, cap_pairs  # noqa: E402
from deform.dump import EventState, load_dump  # noqa: E402
from deform.pipeline import correct_map, inp_from_dump  # noqa: E402


def kept_set(pairs):
    return set(zip(pairs["src_uid"].cpu().tolist(), pairs["src_pix"].cpu().tolist()))


def hop_rings(edges, K, src, max_hop=3):
    adj = collections.defaultdict(list)
    for a, b in edges.tolist():
        adj[a].append(b)
        adj[b].append(a)
    hop = np.full(K, max_hop + 1, dtype=np.int64)
    hop[src] = 0
    q = collections.deque([src])
    while q:
        u = q.popleft()
        if hop[u] >= max_hop:
            continue
        for v in adj[u]:
            if hop[v] > hop[u] + 1:
                hop[v] = hop[u] + 1
                q.append(v)
    return hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-k2a", type=int, default=100)
    ap.add_argument("--n-sel", type=int, default=5)
    ap.add_argument("--n-unsel", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    d = load_dump(a.dump)
    st = EventState(d)
    cfg = dcfg.resolve(yaml.safe_load(open(a.yaml)), d["config"])
    cfg["seed"] = int(d["meta"].get("seed", 0))
    inp = inp_from_dump(d, st.frame)
    W = int(inp["intr"]["W"])
    cache = {}
    ref = correct_map(inp, cfg, cfg["variant"], cache=cache)
    ctx = cache[PL.prep_key(cfg, cfg["variant"])]
    gated, gnode = ctx["gated_ids"], ctx["gated_node"]
    G = int(gated.shape[0])
    ref_keep = kept_set(ctx["pairs"])
    node_of = {(int(u), int(p)): int(n) for (u, p), n in zip(gated.tolist(), gnode.tolist())}
    sel_mask = np.array([(int(u), int(p)) in ref_keep for u, p in gated.tolist()])
    print(f"gated {G}, selected {int(sel_mask.sum())}, nodes {ctx['nodes']['g'].shape[0]}, ref accepted {ref['accepted']} "
          f"iters {ref['log'].get('lbfgs_iters')} e_dl {ref['log'].get('edl_opt_init_mm'):.3f}->{ref['log'].get('edl_opt_final_mm'):.3f} mm")
    rng = np.random.default_rng(a.seed)
    rep = {"dump": os.path.basename(a.dump), "yaml": a.yaml, "n_gated": G, "n_selected": int(sel_mask.sum()),
           "n_nodes": int(ctx["nodes"]["g"].shape[0]), "ref": {k: ref["log"].get(k) for k in ("corr_capped", "lbfgs_iters", "edl_opt_init_mm", "edl_opt_final_mm", "max_node_disp_m")}}

    # ------------------------------------------------------------ K2a
    sel_cfg = dict(select=cfg["corr"].get("select", "random"), img_w=W, salt=int(cfg["corr"].get("hash_salt", HASH_SALT)))
    base_pairs = {"src_uid": gated[:, 0].clone(), "src_pix": gated[:, 1].clone(), "P": G}
    if ctx.get("gated_omega") is not None:
        base_pairs["omega"] = ctx["gated_omega"].clone()
    k0 = kept_set(cap_pairs(base_pairs, gnode, cfg["corr"]["max_per_node"], cfg["corr"]["max_total"], cfg["seed"], omega=base_pairs.get("omega"), **sel_cfg))
    assert k0 == ref_keep, "cap replay differs from the pipeline's kept set"
    k2a = []
    for i in rng.choice(G, min(a.n_k2a, G), replace=False):
        keep = torch.ones(G, dtype=torch.bool); keep[int(i)] = False
        q = {k: (v[keep] if torch.is_tensor(v) else v) for k, v in base_pairs.items()}
        q["P"] = G - 1
        kb = kept_set(cap_pairs(q, gnode[keep], cfg["corr"]["max_per_node"], cfg["corr"]["max_total"], cfg["seed"], omega=q.get("omega"), **sel_cfg))
        changed = {node_of[t] for t in k0 ^ kb}
        k2a.append({"removed_selected": bool(sel_mask[int(i)]), "n_nodes_changed": len(changed), "n_pairs_changed": len(k0 ^ kb)})
    worst = max(r["n_nodes_changed"] for r in k2a)
    unsel_bad = sum(1 for r in k2a if not r["removed_selected"] and r["n_nodes_changed"] > 0)
    rep["K2a"] = {"trials": len(k2a), "max_nodes_changed": worst, "n_removed_selected": sum(r["removed_selected"] for r in k2a),
                  "unselected_removals_changing_set": unsel_bad, "pass": worst <= 1 and unsel_bad == 0}
    print(f"K2a: {len(k2a)} trials, max nodes changed {worst}, removed-selected {rep['K2a']['n_removed_selected']}, "
          f"unselected removals that changed the set {unsel_bad} -> {'PASS' if rep['K2a']['pass'] else 'FAIL'}")

    # ------------------------------------------------------------ K2b
    t_ref = ref["node_t"].double()
    edges = ctx["edges"].cpu()
    K = int(ctx["nodes"]["g"].shape[0])
    sel_idx = np.nonzero(sel_mask)[0]
    unsel_idx = np.nonzero(~sel_mask)[0]
    trials = [(int(i), True) for i in rng.choice(sel_idx, min(a.n_sel, len(sel_idx)), replace=False)] + \
             [(int(i), False) for i in rng.choice(unsel_idx, min(a.n_unsel, len(unsel_idx)), replace=False)]
    rows = []
    for i, was_sel in trials:
        uid, pix = int(gated[i, 0]), int(gated[i, 1])
        inp2 = dict(inp)
        inp2["drop_pair"] = (uid, pix)
        c2 = {PL.rel_key(cfg): cache[PL.rel_key(cfg)]}  # reuse the reliability, rebuild everything after it
        r2 = correct_map(inp2, cfg, cfg["variant"], cache=c2)
        ok_shape = r2.get("node_t") is not None and r2["node_t"].shape == t_ref.shape
        row = {"removed_selected": was_sel, "uid": uid, "pix": pix, "node": node_of[(uid, pix)], "accepted": r2["accepted"],
               "reason": r2["reason"], "lbfgs_iters": r2["log"].get("lbfgs_iters"), "corr_capped": r2["log"].get("corr_capped"),
               "edl_opt_final_mm": r2["log"].get("edl_opt_final_mm"), "dropped": r2["log"].get("dropped_pair")}
        if ok_shape:
            dt = (r2["node_t"].double() - t_ref).norm(dim=1).numpy()
            hop = hop_rings(edges, K, node_of[(uid, pix)])
            row.update({"dt_max": float(dt.max()), "dt_p50": float(np.median(dt)), "dt_p90": float(np.quantile(dt, 0.9)),
                        "n_gt_1e-4": int((dt > 1e-4).sum()), "n_gt_1e-6": int((dt > 1e-6).sum()),
                        "dt_max_by_hop": {str(h): float(dt[hop == h].max()) if (hop == h).any() else None for h in range(5)}})
        rows.append(row)
        print(f"K2b drop {'SEL' if was_sel else 'unsel'} uid {uid} pix {pix} node {row['node']}: capped {row['corr_capped']} "
              f"iters {row['lbfgs_iters']} dt max {row.get('dt_max', float('nan')):.2e} p90 {row.get('dt_p90', float('nan')):.2e} "
              f"n>1e-4 {row.get('n_gt_1e-4')} e_dl {row['edl_opt_final_mm']}")
        del r2
        torch.cuda.empty_cache()
    sel_rows = [r for r in rows if r["removed_selected"] and "dt_max" in r]
    unsel_rows = [r for r in rows if not r["removed_selected"] and "dt_max" in r]
    rep["K2b"] = {"rows": rows,
                  "unselected_max_dt": max([r["dt_max"] for r in unsel_rows], default=None),
                  "selected_max_dt": max([r["dt_max"] for r in sel_rows], default=None),
                  "selected_median_of_max_dt": float(np.median([r["dt_max"] for r in sel_rows])) if sel_rows else None,
                  "pass_plan": all(r["dt_max"] <= 1e-4 for r in sel_rows + unsel_rows),
                  "pass_unselected_exact": all(r["dt_max"] <= 1e-6 for r in unsel_rows)}
    print(f"K2b: unselected max dt {rep['K2b']['unselected_max_dt']}, selected max dt {rep['K2b']['selected_max_dt']}, "
          f"plan rule (<=1e-4) {'PASS' if rep['K2b']['pass_plan'] else 'WARN'}")
    with open(os.path.join(a.out, f"k2_{os.path.basename(a.dump).replace('.pt', '')}.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print("K2A_PASS" if rep["K2a"]["pass"] else "K2A_FAIL")


if __name__ == "__main__":
    main()
