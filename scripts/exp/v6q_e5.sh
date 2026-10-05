#!/usr/bin/env bash
# plan v6 quick: the E5 run (baseline rigid, seed 0, keyframe-0 fix + loop logs + revisit dumps), then the analyses
# that need it (E3 on its dumps, E1 on its maps, the revisit-gap analysis). One SLAM run, nothing else on the GPU.
cd "$(dirname "$0")/../.."
bash scripts/exp/run_one_wt.sh configs/tum/v6q/fr1_room_E5.yaml 0 E5 -l -r 26000 --dump-loops
source tools/exp_env.sh
R=results_exp/tum/room
E5=$(ls -td $R/*_E5 2>/dev/null | head -1)
O=results_exp/reports/v6q
echo "== E5 run folder: $E5 $(date -Is)"
if [ -z "$E5" ] || [ ! -f "$E5/metrics.csv" ]; then echo "V6Q_E5_FAILED (no metrics.csv)"; exit 1; fi
echo "== E3 on E5 $(date -Is)"
python -u tools/e3_floor_event.py --out $O/e3_e5 --revisit --final $E5 2>&1 | grep --line-buffered -v "Warn\|^  warn"
echo "== E1 on E5 $(date -Is)"
python -u tools/e1_refine_split.py --out $O/e1_e5 $E5 2>&1 | grep --line-buffered -v "Warn\|^  warn"
FLOOR=$(python - <<PY
import json, statistics as S
v = []
for p in ("$O/e3/e3.jsonl", "$O/e3_e5/e3.jsonl"):
    try:
        for l in open(p):
            r = json.loads(l)
            if "floor" in r and r["state"] != "final" and r["floor"]["rank"]["median_mm"] is not None:
                v.append(r["floor"]["rank"]["median_mm"])
    except FileNotFoundError:
        pass
print(f"{S.median(v):.2f}" if v else "7.0")
PY
)
echo "== E5 revisit gaps (event-time floor, keyframe-wise split, median over states: $FLOOR mm) $(date -Is)"
python -u tools/e5_revisit_gap.py --out $O/e5 --floor-mm $FLOOR --baseline $R/20261002013837_rigid_s0 $E5 2>&1 | grep --line-buffered -v "Warn\|^  warn"
echo "V6Q_E5_CHAIN_DONE $(date -Is)"
