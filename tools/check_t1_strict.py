"""Plan v2 T1/T7 (strict form): old code (runs/v2_backup) vs new code with the SAME reliability S.

The existing pipeline is not bit-reproducible run to run: S uses per-Gaussian contributions accumulated
with atomicAdd in the rasterizer, so S changes at the 1e-7 level between runs and everything downstream
(node positions, a pixel or two at the alpha threshold, the LBFGS path) moves by ~1e-6 m. This check
removes that source: S is computed once per event, saved, and injected into both code versions through
the pipeline cache, so any difference left is due to the code change.

usage:
  python tools/check_t1_strict.py run  --pkg new|old --rel DIR --out FILE.pt --override YAML RUN_DIR
  python tools/check_t1_strict.py cmp  A.pt B.pt
"""
import argparse
import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = {"A-1": ("A", 1.0, 1.0), "A-10": ("A", 10.0, 1.0), "A-100": ("A", 100.0, 1.0),
        "B-1": ("B", 1.0, 1.0), "B-10": ("B", 10.0, 1.0), "B-100": ("B", 100.0, 1.0),
        "A-noCon": ("A", 10.0, 0.0), "B-noCon": ("B", 10.0, 0.0)}
LOGK = ("corr_raw", "corr_gated", "corr_capped", "n_nodes", "n_filler", "n_edges", "lbfgs_iters")


def run(a):
    if a.pkg == "old":
        sys.path.insert(0, os.path.join(ROOT, "runs", "v2_backup", "oldpkg"))
    sys.path.insert(0, ROOT)
    import deform  # noqa: F401
    import torch
    import yaml
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from deform import config as dcfg
    from deform.dump import EventState, list_dumps, load_dump
    from deform.pipeline import correct_map, inp_from_dump
    print("deform package:", os.path.dirname(deform.__file__), flush=True)
    override = yaml.safe_load(open(a.override)) or {}
    os.makedirs(a.rel, exist_ok=True)
    out = {}
    for p in list_dumps(a.run_dir):
        d = load_dump(p)
        st = EventState(d)
        inp = inp_from_dump(d, st.frame)
        base = dcfg.resolve(override, st.config)
        base["seed"] = int(st.meta.get("seed", 0))
        relp = os.path.join(a.rel, os.path.basename(p))
        cache = {}
        if os.path.exists(relp):
            cache["rel"] = torch.load(relp, map_location="cuda", weights_only=False)
        for name, (var, wp, wc) in GRID.items():
            cfg = copy.deepcopy(base)
            cfg["energy"]["w_p"], cfg["energy"]["w_con"] = wp, wc
            r = correct_map(inp, cfg, var, cache)
            out[(os.path.basename(p), name)] = {
                "xyz": r["xyz"].detach().cpu(), "rot": r["rot"].detach().cpu(),
                "poses": {u: torch.as_tensor(P).double().cpu() for u, P in r["poses"].items()},
                "accepted": r["accepted"], "reason": r["reason"], "log": {k: r["log"].get(k) for k in LOGK}}
        if not os.path.exists(relp):
            torch.save(cache["rel"], relp)
        print(os.path.basename(p), {n: out[(os.path.basename(p), n)]["log"]["corr_capped"] for n in GRID}, flush=True)
    torch.save(out, a.out)


def cmp(a):
    import torch
    A, B = torch.load(a.a, weights_only=False), torch.load(a.b, weights_only=False)
    ok = True
    worst = 0.0
    for k in sorted(A):
        x, y = A[k], B[k]
        dx = float((x["xyz"] - y["xyz"]).norm(dim=1).max())
        dr = float((x["rot"] - y["rot"]).abs().max())
        dp = max(float((x["poses"][u] - y["poses"][u]).abs().max()) for u in x["poses"])
        same = x["accepted"] == y["accepted"] and x["reason"] == y["reason"] and x["log"] == y["log"]
        worst = max(worst, dx)
        flag = "" if (same and dx < 1e-6) else "  <-- DIFF"
        print(f"{k[0]:28s} {k[1]:8s} max|dxyz| {dx:.3e} m  max|drot| {dr:.3e}  max|dpose| {dp:.3e}  logs_equal={same}{flag}")
        ok &= same and dx < 1e-6
    print(f"worst max|dxyz| = {worst:.3e} m")
    print("T1_STRICT_PASS" if ok else "T1_STRICT_FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pkg", choices=("new", "old"), required=True)
    r.add_argument("--rel", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--override", required=True)
    r.add_argument("run_dir")
    c = sub.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        sys.exit(0 if cmp(a) else 1)
