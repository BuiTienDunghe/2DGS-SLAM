"""Local-loop frequency on a finished run, from the verbose frontend log (run.log) only. Analysis only.

usage: python tools/loop_stats_from_log.py --out DIR RUN_DIR [RUN_DIR ...]
Parses the `Track:` / `Loop:` lines the frontend prints with -l (utils/slam_frontend.py: try_loop_closure,
detect_loop_by_revisit, detect_loop_by_featquery, is_this_loop_necessary, reloc_with_mast3r). rich wraps long lines
at the terminal width, so continuation lines are joined to the record they belong to; tqdm fragments are split off.
Per run:
  1 frames with a loop check (= tracked frames; loop_closure_check_every = 1)
  2 revisit: frames passing the geometric gate (observed_ratio > loop_overlap_ratio), throttled / passed on
  3 featquery: frames with a candidate above the score threshold, throttled / passed on
  4 MASt3R relocalisation: rejections per step and accepted loops, per path
  5 revisit bursts: gate-passing frames at most GAP tracked frames apart; start, end, candidate keyframes,
    max observed_ratio, accepted or not
  6 what the log holds about observed_ratio (only values > 0.85 x threshold are logged)
Writes DIR/loop_stats.json and prints tables.
"""
import argparse
import collections
import json
import os
import re

TAG = re.compile(r"(?=(?:Track|Loop|Map|Eval): )")
START = re.compile(r"^(Track|Loop|Map|Eval): ")


def records(path):
    raw = open(path, errors="replace").read().replace("\r", "\n")
    out = []
    for line in raw.split("\n"):
        for p in TAG.split(line):
            p = p.rstrip()
            if not p.strip():
                continue
            if START.match(p):
                out.append(p)
            elif p.startswith("SLAM:") or p.startswith("[run_one]") or "%|" in p:
                out.append("OTHER " + p)
            elif out and not out[-1].startswith("OTHER"):
                out[-1] += " " + p.strip()  # rich wrap continuation
    return out


def parse(rd, gap=2):
    recs = records(os.path.join(rd, "run.log"))
    frames, idx_of = [], {}
    cur = -1
    F = collections.defaultdict(dict)  # per frame facts
    reloc = []  # one dict per relocalisation call
    open_reloc = None
    for r in recs:
        m = re.match(r"Track: track frame=(\d+) iters=(\d+)/(\d+)", r)
        if m:
            n = int(m.group(1))
            if n > cur:  # a re-localisation tracking of an old keyframe has a smaller uid
                cur = n
                idx_of[n] = len(frames)
                frames.append(n)
            continue
        if not r.startswith("Loop: "):
            continue
        s = r[6:]
        f = F[cur]
        m = re.search(r"revisit: PASS geometric gate . observed_ratio=([\d.]+).*candidate\s+kf=(\d+)", s)
        if m:
            f["rev_ratio"], f["rev_kf"] = float(m.group(1)), int(m.group(2))
            continue
        m = re.search(r"revisit: FAIL geometric gate \(near miss\) . observed_ratio=([\d.]+)", s)
        if m:
            f["near_ratio"] = float(m.group(1))
            continue
        m = re.search(r"(revisit|featquery): candidate kf=(\d+) throttle (PASS|FAIL)", s)
        if m:
            f[m.group(1)[:3] + "_throttle"] = m.group(3)
            if m.group(1) == "featquery":
                f["fq_kf"] = int(m.group(2))
            continue
        m = re.search(r"featquery PASS kf=(\d+) score=([\d.]+)", s)
        if m:
            f["fq_pass_kf"], f["fq_score"] = int(m.group(1)), float(m.group(2))
            continue
        m = re.search(r"mast3r reloc \((revisit|featquery)\): query_frame=(\d+) . loop_kf=(\d+)", s)
        if m:
            open_reloc = {"path": m.group(1), "frame": int(m.group(2)), "loop_kf": int(m.group(3)), "result": "unknown"}
            reloc.append(open_reloc)
            continue
        m = re.search(r"mast3r \((revisit|featquery)\): REJECT at (predict_2view confidence|depth-consistency overlap|loop tracking verification)", s)
        if m and open_reloc is not None:
            open_reloc["result"] = {"predict_2view confidence": "reject_conf", "depth-consistency overlap": "reject_overlap",
                                    "loop tracking verification": "reject_tracking"}[m.group(2)]
            v = re.search(r"(mean_conf|overlap_ratio|depth_avg_error)=([\d.]+)", s)
            if v:
                open_reloc[v.group(1)] = float(v.group(2))
            continue
        m = re.search(r"(revisit|featquery): ACCEPT loop . relocalized kf=(\d+)", s)
        if m:
            f["accept"] = (m.group(1), int(m.group(2)))
            if open_reloc is not None:
                open_reloc["result"] = "accepted"
            continue
    n = len(frames)
    rev = sorted(fr for fr, d in F.items() if "rev_ratio" in d)
    fq = sorted(fr for fr, d in F.items() if "fq_pass_kf" in d)
    near = sorted(fr for fr, d in F.items() if "near_ratio" in d)

    def bursts(fs):
        out, curb = [], []
        for fr in fs:
            if curb and idx_of[fr] - idx_of[curb[-1]] > gap:
                out.append(curb)
                curb = []
            curb.append(fr)
        if curb:
            out.append(curb)
        return out

    def describe(b, key):
        kfs = collections.Counter(F[fr].get("rev_kf") for fr in b if F[fr].get("rev_kf") is not None)
        acc = [(fr,) + F[fr]["accept"] for fr in b if "accept" in F[fr]]
        thr = collections.Counter(F[fr].get("rev_throttle") for fr in b if "rev_throttle" in F[fr])
        rj = collections.Counter(r["result"] for r in reloc if r["path"] == "revisit" and r["frame"] in set(b))
        return {"start": b[0], "end": b[-1], "n_frames": len(b), "cand_kf": [k for k, _ in kfs.most_common(3)],
                "cand_kf_range": [min(kfs), max(kfs)] if kfs else None,
                "max_ratio": max(F[fr].get(key, F[fr].get("rev_ratio", 0.0)) for fr in b),
                "throttle_pass": thr.get("PASS", 0), "throttle_fail": thr.get("FAIL", 0), "reloc": dict(rj),
                "accepted": [{"frame": a[0], "path": a[1], "loop_kf": a[2]} for a in acc]}

    rb = [describe(b, "rev_ratio") for b in bursts(rev)]
    # every frame whose observed_ratio is known to exceed 0.85 x threshold (gate passes + near misses)
    nb = [describe(b, "near_ratio") for b in bursts(sorted(set(rev) | set(near)))]
    res = {"run": os.path.basename(rd), "n_tracked_frames": n, "first_frame": frames[0] if frames else None, "last_frame": frames[-1] if frames else None,
           "revisit": {"gate_pass": len(rev), "throttle_fail": sum(1 for fr in rev if F[fr].get("rev_throttle") == "FAIL"),
                       "throttle_pass": sum(1 for fr in rev if F[fr].get("rev_throttle") == "PASS"), "near_miss_frames": len(near)},
           "featquery": {"score_pass": len(fq), "throttle_fail": sum(1 for fr in fq if F[fr].get("fea_throttle") == "FAIL"),
                         "throttle_pass": sum(1 for fr in fq if F[fr].get("fea_throttle") == "PASS")},
           "reloc": {p: dict(collections.Counter(r["result"] for r in reloc if r["path"] == p)) for p in ("revisit", "featquery")},
           "accepted": [{"frame": fr, "path": d["accept"][0], "loop_kf": d["accept"][1]} for fr, d in sorted(F.items()) if "accept" in d],
           "revisit_bursts": rb, "bursts_ratio_gt_0p425": nb,
           "ratio_values_logged": len(rev) + len(near)}
    # backend view
    for name in ("loop_attempts.jsonl", "loop_events.jsonl"):
        p = os.path.join(rd, name)
        res[name] = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allres = []
    for rd in a.runs:
        r = parse(rd, a.gap)
        allres.append(r)
        att = r["loop_attempts.jsonl"] or []
        ev = r["loop_events.jsonl"] or []
        rv, fq = r["revisit"], r["featquery"]
        print(f"== {r['run']}  tracked frames {r['n_tracked_frames']} ({r['first_frame']}..{r['last_frame']})")
        print(f"   revisit: gate pass {rv['gate_pass']} frames (throttled {rv['throttle_fail']}, passed on {rv['throttle_pass']}); near-miss frames {rv['near_miss_frames']}")
        print(f"   featquery: score pass {fq['score_pass']} frames (throttled {fq['throttle_fail']}, passed on {fq['throttle_pass']})")
        print(f"   reloc revisit {r['reloc']['revisit']}  featquery {r['reloc']['featquery']}")
        print(f"   accepted loops: {[(x['frame'], x['loop_kf'], x['path']) for x in r['accepted']]}")
        print(f"   backend attempts {len(att)}: " + ", ".join(f"{x['cur_uid']}<->{x['loop_uid']} pgo={x.get('pgo_ran')}" for x in att))
        print(f"   backend PGO events {len(ev)}: " + ", ".join(f"{x['cur_uid']}<->{x['loop_uid']}" for x in ev))
        print(f"   revisit bursts ({len(r['revisit_bursts'])}):")
        for b in r["revisit_bursts"]:
            acc = ", ".join(f"{x['frame']}<->{x['loop_kf']} via {x['path']}" for x in b["accepted"]) or "no"
            print(f"      {b['start']:5d}-{b['end']:5d} ({b['n_frames']:3d} fr) cand kf {b['cand_kf']} range {b['cand_kf_range']} max ratio {b['max_ratio']:.3f} "
                  f"throttle pass/fail {b['throttle_pass']}/{b['throttle_fail']} reloc {b['reloc']} accepted: {acc}")
        print(f"   bursts with observed_ratio > 0.425 (pass + near miss): {len(r['bursts_ratio_gt_0p425'])}; ratio values in the log: {r['ratio_values_logged']} of {r['n_tracked_frames']} frames")
    json.dump(allres, open(os.path.join(a.out, "loop_stats.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
