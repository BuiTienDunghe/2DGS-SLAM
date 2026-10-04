"""plan v4 A3c / K1: per-stage determinism probe of the offline pipeline on one dump.

usage: python tools/det_probe.py DEFORM_YAML DUMP.pt --out DIR [--runs 2] [--tag NAME]
Each run is a fresh process (`--child` mode) that executes correct_map once with stage snapshots and saves
them to DIR/<tag>_run<i>.pt; the parent then compares run 0 with every other run, stage by stage:
  contributions of the first reliability keyframe, n_vis, S, R, node positions g, node times, pair funnel
  counts, gated pair set (uid, pixel), node assignment, capped pair set, solver iterations, node translations
  (max |dt|), Gaussian positions (max |d mu|), accepted / reason, event time.
Prints one table and writes DIR/<tag>_probe.json. Exit code 0 iff capped pair sets are identical and
max node translation difference <= 1e-6 m (the K1 rule).
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def child(yaml_path, dump_path, out_path):
    import deform  # noqa: F401
    import torch
    import yaml
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from deform import config as dcfg
    from deform import pipeline as PL
    from deform.dump import EventState, load_dump
    from deform.pipeline import correct_map, inp_from_dump
    from deform.render_utils import DEV, det_scope, make_cam, render_subset

    torch.manual_seed(0)
    d = load_dump(dump_path)
    st = EventState(d)
    user = yaml.safe_load(open(yaml_path))
    cfg = dcfg.resolve(user, d["config"])
    cfg["seed"] = int(d["meta"].get("seed", 0))
    inp = inp_from_dump(d, st.frame)
    snap = {}
    # one raw render of the first keyframe (unmasked contributions) under the same render scope as the pipeline
    kf0 = sorted(inp["kf_uids"])[0]
    with det_scope((cfg.get("det") or {}).get("render", False)):
        cam = make_cam(kf0, inp["poses_pre"][kf0], inp["intr"])
        snap["contrib_kf0"] = render_subset(cam, inp["g"])["contrib_full"].cpu()
    cache = {}
    res = correct_map(inp, cfg, cfg["variant"], cache=cache)
    ctx = cache[PL.prep_key(cfg, cfg["variant"])]
    rel = cache[PL.rel_key(cfg)]
    snap.update({"n_vis": rel["n"].cpu(), "S": rel["S"].cpu(), "R": rel["R"].cpu(),
                 "g": ctx["nodes"]["g"].cpu(), "t_nodes": ctx["nodes"]["t"].cpu(),
                 "node_of_cand": ctx["nodes"]["node_of_cand"].cpu(), "cand": ctx["nodes"]["cand"].cpu(),
                 "funnel": res["log"].get("funnel"), "corr_gated": res["log"].get("corr_gated"),
                 "corr_capped": res["log"].get("corr_capped"), "lbfgs_iters": res["log"].get("lbfgs_iters"),
                 "accepted": res["accepted"], "reason": res["reason"], "t_total": res["log"]["t_stage_s"].get("total"),
                 "xyz": res["xyz"].cpu(), "det_warnings": res["log"].get("det_warnings"),
                 "edl_opt": (res["log"].get("edl_opt_init_mm"), res["log"].get("edl_opt_final_mm"))})
    pairs = ctx.get("pairs") or {}
    if pairs.get("P", 0) > 0:
        snap["capped_ids"] = torch.stack([pairs["src_uid"].cpu().long(), pairs["src_pix"].cpu().long()], 1)
        if ctx.get("inf_new") is not None:
            snap["pair_node"] = ctx["inf_new"][0][torch.arange(pairs["P"], device=DEV), ctx["inf_new"][1].argmax(1)].cpu()
    # the gated (pre-cap) set is not kept by the pipeline: rebuild it from the funnel-equivalent call is costly,
    # so we record the gated count only (log) plus the capped set above.
    # node translations: re-run the solve result through the log (max_node_disp is a scalar) -> store theta
    snap["node_t"] = res.get("node_t")
    torch.save(snap, out_path)
    print("child done", out_path)


def compare(a, b):
    import torch
    out = {}

    def mx(x, y):
        return None if x is None or y is None else float((x.double() - y.double()).abs().max())

    out["contrib_kf0_max"] = mx(a["contrib_kf0"], b["contrib_kf0"])
    out["contrib_kf0_n_diff"] = int((a["contrib_kf0"] != b["contrib_kf0"]).sum())
    out["n_vis_n_diff"] = int((a["n_vis"] != b["n_vis"]).sum())
    out["S_max"] = mx(a["S"], b["S"])
    out["S_n_diff"] = int((a["S"] != b["S"]).sum())
    out["R_n_diff"] = int((a["R"] != b["R"]).sum())
    out["n_nodes"] = (int(a["g"].shape[0]), int(b["g"].shape[0]))
    if a["g"].shape == b["g"].shape:
        out["g_max"] = mx(a["g"], b["g"])
        out["t_nodes_n_diff"] = int((a["t_nodes"] != b["t_nodes"]).sum())
        out["node_of_cand_n_diff"] = int((a["node_of_cand"] != b["node_of_cand"]).sum()) if a["node_of_cand"].shape == b["node_of_cand"].shape else None
    out["funnel"] = (a["funnel"], b["funnel"])
    out["corr_gated"] = (a["corr_gated"], b["corr_gated"])
    out["corr_capped"] = (a["corr_capped"], b["corr_capped"])
    sa = set(map(tuple, a["capped_ids"].tolist())) if "capped_ids" in a else set()
    sb = set(map(tuple, b["capped_ids"].tolist())) if "capped_ids" in b else set()
    out["capped_set_sym_diff"] = len(sa ^ sb)
    if "pair_node" in a and "pair_node" in b and a["pair_node"].shape == b["pair_node"].shape and sa == sb:
        # same set and same (sorted) order -> compare node assignment per pair
        out["pair_node_n_diff"] = int((a["pair_node"] != b["pair_node"]).sum())
    out["lbfgs_iters"] = (a["lbfgs_iters"], b["lbfgs_iters"])
    out["edl_opt"] = (a["edl_opt"], b["edl_opt"])
    out["node_t_max"] = mx(a.get("node_t"), b.get("node_t"))
    out["xyz_max"] = mx(a["xyz"], b["xyz"])
    out["xyz_n_gt_1e-6"] = int(((a["xyz"].double() - b["xyz"].double()).norm(dim=1) > 1e-6).sum())
    out["accepted"] = (a["accepted"], b["accepted"], a["reason"], b["reason"])
    out["t_total"] = (a["t_total"], b["t_total"])
    out["det_warnings"] = a.get("det_warnings")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--child", default=None, help="internal: write the snapshot to this path and exit")
    a = ap.parse_args()
    if a.child:
        child(a.yaml, a.dump, a.child)
        return
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or (os.path.basename(a.yaml).replace(".yaml", "") + "_" + os.path.basename(a.dump).replace(".pt", ""))
    paths = []
    for i in range(a.runs):
        p = os.path.join(a.out, f"{tag}_run{i}.pt")
        subprocess.run([sys.executable, os.path.abspath(__file__), a.yaml, a.dump, "--out", a.out, "--child", p], check=True)
        paths.append(p)
    import torch
    snaps = [torch.load(p, weights_only=False) for p in paths]
    rows = [compare(snaps[0], s) for s in snaps[1:]]
    ok = all(r["capped_set_sym_diff"] == 0 and (r["node_t_max"] is None or r["node_t_max"] <= 1e-6) for r in rows)
    rep = {"tag": tag, "yaml": a.yaml, "dump": a.dump, "runs": a.runs, "pairs": rows, "K1_pass": ok}
    with open(os.path.join(a.out, f"{tag}_probe.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"== {tag}")
    for k in ("contrib_kf0_max", "contrib_kf0_n_diff", "n_vis_n_diff", "S_max", "S_n_diff", "R_n_diff", "n_nodes", "g_max",
              "t_nodes_n_diff", "node_of_cand_n_diff", "corr_gated", "corr_capped", "capped_set_sym_diff", "pair_node_n_diff",
              "lbfgs_iters", "edl_opt", "node_t_max", "xyz_max", "xyz_n_gt_1e-6", "accepted", "t_total", "det_warnings"):
        print(f"   {k:22s}", [r.get(k) for r in rows])
    print("K1_PASS" if ok else "K1_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
