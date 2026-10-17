"""PLAN E1c §5 / T15: required files and columns of a run, and n_obs_t (dec_log) vs sigma_min (frame_log).
usage: python tools/e1c_analysis/check_run_files.py RUN_DIR METHOD     (exit 1 on any failure)
"""
import csv
import json
import os
import sys

run, method = sys.argv[1], sys.argv[2]
plane = method in ("ours", "gate", "dec", "dec_edge")
fails, notes = [], []


def need(path):
    if not os.path.exists(os.path.join(run, path)):
        fails.append(f"missing {path}")
        return False
    return True


def header(path):
    with open(os.path.join(run, path), newline="") as f:
        return next(csv.reader(f))


for p in ("traj_est_all.txt", "traj_gt_all.txt", "ape_per_frame.csv", "e1_metrics.json", "frame_log.csv", "timing.csv"):
    need(p)
if need("timing.csv"):
    miss = {"obs_basis_ms", "edge_ms"} - set(header("timing.csv"))
    if miss:
        fails.append(f"timing.csv lacks {sorted(miss)}")
if plane:
    if need("frame_log.csv"):
        miss = {"T_init", "pairs", "active_ids"} - set(header("frame_log.csv"))
        if miss:
            fails.append(f"frame_log.csv lacks {sorted(miss)}")
    for p in ("planes.jsonl", "anchors_kf.jsonl"):
        need(p)
    if os.path.exists(os.path.join(run, "anchors_kf.jsonl")):
        a = json.loads(open(os.path.join(run, "anchors_kf.jsonl")).readline())
        miss = {"id", "n_w", "d_w", "n_obs_kf", "active", "created_kf", "n_samples"} - set(a["anchors"][0])
        if miss:
            fails.append(f"anchors_kf.jsonl lacks {sorted(miss)}")
if method in ("dec", "dec_edge") and need("dec_log.csv"):
    want = {"frame_id", "n_obs_t", "n_obs_r", "gp_removed_norm"} | {f"sig_t{i}" for i in range(3)} | \
           {f"Vt{i}" for i in range(9)} | {f"sum_abs_dxi_t{i}" for i in range(3)} | {f"mean_abs_gr_xi_t{i}" for i in range(3)}
    miss = want - set(header("dec_log.csv"))
    if miss:
        fails.append(f"dec_log.csv lacks {sorted(miss)}")
    fl = {r["frame_id"]: r for r in csv.DictReader(open(os.path.join(run, "frame_log.csv")))}
    bad, n = [], 0
    for r in csv.DictReader(open(os.path.join(run, "dec_log.csv"))):
        f = fl[r["frame_id"]]
        n_act = len([x for x in f["active_ids"].split(";") if x != ""])
        sig = float(f["sigma_min"]) if f["sigma_min"] != "" else 0.0
        n_obs = int(r["n_obs_t"])
        ok = (n_obs == 3) == (n_act >= 3 and sig >= 0.3) and n_obs <= n_act
        n += 1
        if not ok:
            bad.append((r["frame_id"], n_obs, n_act, sig))
    notes.append(f"n_obs_t vs sigma_min: {n - len(bad)}/{n} frames consistent")
    if bad:
        fails.append(f"n_obs_t inconsistent with sigma_min at {bad[:5]}")
if method == "dec_edge" and need("edge_log.csv"):
    want = {"frame_id", "n_bnd", "n_occluding", "n_crease", "n_drop_occluded", "n_edge_match", "edge_loss_final", "med_abs_r"}
    miss = want - set(header("edge_log.csv"))
    if miss:
        fails.append(f"edge_log.csv lacks {sorted(miss)}")
print(f"{run} [{method}]: " + ("PASS" if not fails else "FAIL"))
for x in notes + fails:
    print("  " + x)
sys.exit(1 if fails else 0)
