"""plan v5 step B: print the H0 / H1 / loop-recovery / H2 tables for the report (CPU only).
usage: python tools/summarize_v5B.py [results_exp/reports/v5]
"""
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "results_exp/reports/v5"


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


print("== H0 noise floor")
for s in ("rank", "random"):
    p = os.path.join(R, "h0", f"h0_{s}.json")
    if os.path.exists(p):
        for run, v in json.load(open(p)).items():
            print(f"  {s:6s} {run[-10:]:10s} views {v['n_views']:3d} pix {v['n_pix']:6d} median {f(v['floor_median_mm'], 2)} mm p90 {f(v['floor_p90_mm'])} mm")

print("== loop recovery checks")
p = os.path.join(R, "h1", "loops.json")
if os.path.exists(p):
    for run, d in json.load(open(p)).items():
        for k, v in d.items():
            print(f"  {run[-10:]:10s} {k:12s} grad {v['grad_norm_rest']:8.0f} -> {v['grad_norm_full']:6.1f}  LM drift {v['lm_drift_from_xstar_m']:.1e} m  "
                  f"err(x*) {v['err_at_xstar']:8.2f}  loop residual {1e3 * v['loop_residual_t_m']:.2f} mm / {57.2958 * v['loop_residual_rot_rad']:.4f} deg")

print("== H1")
p = os.path.join(R, "h1", "h1.jsonl")
if os.path.exists(p):
    for l in open(p):
        r = json.loads(l)
        print(f"  {r['run'][-10:]:10s} k={r['k']} {r['pair']:>10s} -> {r['next']:>10s}  rigid@k {f(r.get('rigid_k_mm'))}  post@k {f(r.get('post_k_mm'))}  "
              f"pre@k+1 {f(r.get('pre_k1_mm'))}  real post@k+1 {f(r.get('real_post_k1_mm'))}  err after write replay/logged {f(r.get('err_after_write_k_replay'), 2)}/{f(r.get('err_after_write_k_logged'), 2)}")
        for n, s in (r.get("scenarios") or {}).items():
            ret = "—" if s["retained"] is None else f"{100 * s['retained']:.0f}%"
            print(f"      ({n}) gap {f(s['gap_after_k1_mm'])} mm retained {ret:>5s}  loop kept {str(s['loop_kept']):5s} removed<50 {str(s['loop_removed_lt50']):5s}  "
                  f"err {s['err_no_loop']:.0f}/{s['err_with_loop']:.0f}  pgo {s['pgo_err_before']:.0f}->{s['pgo_err_after']:.0f}  dev vs real {s['max_dev_vs_real_m']:.1e} m  valid {s['valid_frac']:.2f}")

print("== H2")
p = os.path.join(R, "h2", "h2.jsonl")
if os.path.exists(p):
    for l in open(p):
        r = json.loads(l)
        print(f"  {r['run'][-10:]:10s} {r['cur']}<->{r['loop']}  pix {r.get('n_pix')}  rigid {f(r.get('edl_rigid_mm'), 2)}  A {f(r.get('edl_A_mm'), 2)} (R {f(None if r.get('R_A') is None else 100 * r['R_A'])}%)  "
              f"B {f(r.get('edl_B_mm'), 2)} (R {f(None if r.get('R_B') is None else 100 * r['R_B'])}%)  B acc {r.get('B_accepted')} {r.get('B_reason')}  B disp {f(r.get('B_max_node_disp_mm'))} mm  B err {r.get('B_error')}")
