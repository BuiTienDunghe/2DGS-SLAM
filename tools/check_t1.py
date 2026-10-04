"""Plan v2 T1/T7: with every switch at its default, the P3 grid reproduces the frozen results.

usage: python tools/check_t1.py NEW/events.jsonl REF/events.jsonl [NOISE_A/events.jsonl NOISE_B/events.jsonl]
Compares, for the events present in NEW: pair counts (raw / gated / capped), acceptance, node counts
must be identical; e_dl (P3 pixel set) per method must differ by < 1e-6 m (plan threshold). The optional
pair of unchanged-code re-runs gives the run-to-run noise of the existing code for context.
"""
import json
import sys

FIELDS = ("edl", "edl_p90", "estep", "psnr", "depth_l1", "ate")
LOGS = ("corr_raw", "corr_gated", "corr_capped", "accepted", "n_nodes", "n_filler")


def load(p):
    return {json.loads(l)["dump"]: json.loads(l) for l in open(p)}


def diff(new, ref):
    out, exact = {}, []
    for k, r in new.items():
        a = ref[k]
        for m, mv in r["methods"].items():
            for f in FIELDS:
                v, w = mv.get(f), a["methods"][m].get(f)
                if v is None or w is None:
                    if v != w:
                        exact.append((k, m, f, v, w))
                    continue
                out[(m, f)] = max(out.get((m, f), 0.0), abs(v - w))
        for m, lg in r["logs"].items():
            for f in LOGS:
                if lg.get(f) != a["logs"][m].get(f):
                    exact.append((k, m, f, lg.get(f), a["logs"][m].get(f)))
    return out, exact


def main():
    new, ref = load(sys.argv[1]), load(sys.argv[2])
    d, exact = diff(new, ref)
    noise = None
    if len(sys.argv) > 4:
        noise, _ = diff(load(sys.argv[3]), load(sys.argv[4]))
    methods = sorted({m for m, _ in d})
    print(f"{'method':10s} {'max|d edl| m':>14s} {'noise edl m':>14s} {'max|d psnr|':>12s} {'max|d ate|':>12s}")
    for m in methods:
        nz = noise.get((m, "edl")) if noise else None
        print(f"{m:10s} {d.get((m, 'edl'), 0):14.3e} {('-' if nz is None else f'{nz:.3e}'):>14s} "
              f"{d.get((m, 'psnr'), 0):12.3e} {d.get((m, 'ate'), 0):12.3e}")
    for e in exact:
        print("EXACT-FIELD MISMATCH", *e)
    a1 = d.get(("A-1", "edl"), 0.0)
    counts_ok = not [e for e in exact if e[1] == "A-1"]
    ok = counts_ok and a1 < 1e-6
    print(f"T1: A-1 pair counts/acceptance identical={counts_ok}; max |d e_dl| A-1 = {a1:.3e} m (< 1e-6 m: {a1 < 1e-6})")
    print("T1_PASS" if ok else "T1_FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
