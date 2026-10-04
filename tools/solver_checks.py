"""Plan v3 step A: solver checks V1-V3 on stored problems (pairs, weights, graph) of real loop events.

usage: python tools/solver_checks.py --out results_exp/reports/v3/solver [--configs A-1,F1t] [--mode grad] RUN_DIR...
  V1  converged: ||grad|| <= 1e-6 ||grad_0|| at the solution, iteration cap not reached
  V2  scale invariance: energy x10 (w_con, w_reg, w_p x10) -> max |d mu| < 1e-5 m
  V3  no noise amplification: pair end points and node positions moved by 1e-6 m -> max |d mu| <= 1e-5 m
mu = Gaussian positions phi(x_pre) of the whole map. The same V2/V3 are also run with the legacy solver for
reference. Problems are saved to OUT/problems/<dump>_<config>.pt so the checks can be re-run without the
pipeline (the upstream S / node building is not deterministic, the stored problem is).
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform  # noqa: F401,E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import run_diag_v2 as RD  # noqa: E402
from deform import config as dcfg  # noqa: E402
from deform.dump import EventState, list_dumps, load_dump  # noqa: E402
from deform.field import phi  # noqa: E402
from deform.pipeline import _prepare, inp_from_dump  # noqa: E402
from deform.render_utils import DEV  # noqa: E402
from deform.solver import Problem, solve  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stored_problem(ctx, cfg):
    p = ctx["pairs"]
    keep = ("x_old", "x_new", "n", "omega", "P")
    return {"g": ctx["nodes"]["g"].double(), "R0": ctx["R0"], "t_init": ctx["t_init"], "edges": ctx["edges"],
            "pairs": {k: p[k] for k in keep if k in p}, "inf_old": ctx["inf_old"], "inf_new": ctx["inf_new"],
            "x_pre": ctx["x_pre"], "idx_g": ctx["idx_g"], "w_g": ctx["w_g"],
            "energy": dict(cfg["energy"]), "corr": {"residual": cfg["corr"].get("residual", "p2l")}}


def run_solve(sp, solver_cfg, scale=1.0, perturb=None, seed=0):
    g = sp["g"].clone()
    pairs = dict(sp["pairs"])
    if perturb:
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def jitter(x):
            d = torch.randn(x.shape, generator=gen, dtype=torch.float64).to(x.device)
            return x.double() + perturb * d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        pairs["x_old"], pairs["x_new"] = jitter(pairs["x_old"]), jitter(pairs["x_new"])
        g = jitter(g)
    cfg = {"energy": {k: (v * scale if k in ("w_con", "w_reg", "w_p") else v) for k, v in sp["energy"].items()},
           "corr": sp["corr"], "solver": solver_cfg}
    prob = Problem(g, sp["R0"], sp["t_init"], sp["edges"], pairs, sp["inf_old"], sp["inf_new"], cfg)
    Kn = g.shape[0]
    theta0 = torch.cat([torch.zeros((Kn, 3), dtype=torch.float64, device=DEV), sp["t_init"].double()], 1)
    t0 = time.perf_counter()
    with torch.enable_grad():
        theta, slog = solve(prob, theta0, cfg)
    R, t, _ = prob.unpack(theta)
    mu = phi(sp["x_pre"], sp["idx_g"], sp["w_g"], g, R, t)
    return theta, mu, slog, time.perf_counter() - t0


def dmax(a, b):
    return float((a - b).norm(dim=-1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="A-1,F1t")
    ap.add_argument("--mode", default="grad")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "problems"), exist_ok=True)
    rows = []
    for rd in a.runs:
        for path in list_dumps(rd):
            d = load_dump(path)
            st = EventState(d)
            inp = inp_from_dump(d, st.frame)
            base = dcfg.resolve(load_deform_config(os.path.join(ROOT, "configs", "deform", "selected.yaml")), st.config)
            base["seed"] = int(st.meta.get("seed", 0))
            cache = {}
            for name in a.configs.split(","):
                cfg = base if name == "A-1" else RD.make_cfg(base, name)
                ctx = _prepare(inp, cfg, "A", cache)
                if ctx["fallback"] is not None and ctx.get("pairs", {}).get("P", 0) == 0:
                    rows.append({"dump": os.path.basename(path), "config": name, "skip": ctx["fallback"]})
                    continue
                sp = stored_problem(ctx, cfg)
                torch.save(sp, os.path.join(a.out, "problems", f"{os.path.basename(path)[:-3]}_{name}.pt"))
                new = dict(cfg["solver"], tol_mode=a.mode)
                old = dict(cfg["solver"], tol_mode="legacy")
                th, mu, sl, ts = run_solve(sp, new)
                _, mu10, sl10, _ = run_solve(sp, new, scale=10.0)
                _, mup, slp, _ = run_solve(sp, new, perturb=1e-6, seed=1)
                _, muL, slL, tL = run_solve(sp, old)
                _, muL10, _, _ = run_solve(sp, old, scale=10.0)
                _, muLp, _, _ = run_solve(sp, old, perturb=1e-6, seed=1)
                gi = sl.get("grad", {})
                row = {"dump": os.path.basename(path), "event": st.meta["event_id"], "config": name,
                       "pairs": int(sp["pairs"].get("P", 0)), "nodes": int(sp["g"].shape[0]),
                       "iters": sl["lbfgs_iters"], "grad_ratio": gi.get("grad_ratio"), "hit_max": gi.get("hit_max"),
                       "resets": gi.get("resets"), "solve_s": ts, "E_final": sum(sl["E_final"].values()),
                       "V1": bool(gi.get("converged")) and not gi.get("hit_max") and (gi.get("grad_ratio") or 1) <= 1e-6,
                       "V2_dmu_m": dmax(mu, mu10), "V2": dmax(mu, mu10) < 1e-5,
                       "V3_dmu_m": dmax(mu, mup), "V3": dmax(mu, mup) <= 1e-5,
                       "iters_x10": sl10["lbfgs_iters"], "iters_perturbed": slp["lbfgs_iters"],
                       "legacy": {"iters": slL["lbfgs_iters"], "E_final": sum(slL["E_final"].values()), "solve_s": tL,
                                  "V2_dmu_m": dmax(muL, muL10), "V3_dmu_m": dmax(muL, muLp),
                                  "dmu_vs_new_m": dmax(muL, mu)}}
                rows.append(row)
                print(json.dumps({k: (round(v, 9) if isinstance(v, float) else v) for k, v in row.items() if k != "legacy"}),
                      "| legacy:", json.dumps({k: (round(v, 9) if isinstance(v, float) else v) for k, v in row["legacy"].items()}),
                      flush=True)
            del cache, inp
            torch.cuda.empty_cache()
    ok = all(r.get("V1") and r.get("V2") and r.get("V3") for r in rows if "skip" not in r)
    with open(os.path.join(a.out, "solver_checks.json"), "w") as f:
        json.dump({"mode": a.mode, "rows": rows, "all_pass": ok}, f, indent=1)
    print("SOLVER_CHECKS_PASS" if ok else "SOLVER_CHECKS_FAIL")


if __name__ == "__main__":
    main()
