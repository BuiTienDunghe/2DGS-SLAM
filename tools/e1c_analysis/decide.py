"""PLAN E1c/E1d decision reports (§10.1) with the user's revised rules of 18/10/2026.
usage:
  python tools/e1c_analysis/decide.py c   SEEDS          e.g. "0" or "0,1"  -> results_exp/e1c/DECISION_C_seeds<..>.md + .json
  python tools/e1c_analysis/decide.py d   SEEDS          (dec_edge vs dec)  -> results_exp/e1d/DECISION_D_seeds<..>.md + .json
Every number is read from seg_metrics.json / frame_log.csv / ape_per_frame.csv / e1_metrics.json.
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np

R = os.path.expanduser("~/2DGS-SLAM/results_exp")
B = 14.64                                   # ATE-all base seed 0 (cm), fixed by the user
TH = dict(C1_S1_raw_end=8.6, C2_S3_ate=3.4, C2_S5_rpe=1.53, C2_S5_path=1.73, C3_S2_ate=1.9, C3_S4_ate=3.2)
REUSED = {"base": f"{R}/e1c/reused/base_seed0", "ours": f"{R}/e1c/reused/ours_seed0"}


def sm(path):
    return json.load(open(os.path.join(path, "seg_metrics.json")))


def vn(x, f="{:.2f}"):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x).replace(".", ",")


def c_checks(m):
    S = m["segments"]
    c1 = S["S1"]["raw_end"] <= TH["C1_S1_raw_end"]
    c2p = dict(S3_ate=S["S3"]["ate_local_cm"] <= TH["C2_S3_ate"], S5_rpe=S["S5"]["rpe_trans_cm_d1"] <= TH["C2_S5_rpe"],
               S5_path=S["S5"]["path_ratio"] <= TH["C2_S5_path"])
    c3p = dict(S2=S["S2"]["ate_local_cm"] <= TH["C3_S2_ate"], S4=S["S4"]["ate_local_cm"] <= TH["C3_S4_ate"])
    return dict(C1=c1, C2=all(c2p.values()), C2_parts=c2p, C3=all(c3p.values()), C3_parts=c3p)


def failed(run_dir, m):
    em = json.load(open(os.path.join(run_dir, "e1_metrics.json")))
    return bool(em.get("has_nan_pose")) or m["ate_all_cm"] > 50 or em.get("frac_frames_ape_gt_30cm", 0) > 0.2


def dup_jitter(run_dir):
    """Share of APE jumps > 3 cm in S5 that fall on frames whose `pairs` anchor ids changed (PLAN §7 C2)."""
    fl = {int(r["frame_id"]): r for r in csv.DictReader(open(os.path.join(run_dir, "frame_log.csv")))}
    ape = [(int(r["frame_id"]), float(r["ape_m"])) for r in csv.DictReader(open(os.path.join(run_dir, "ape_per_frame.csv")))]
    if "pairs" not in next(iter(fl.values())):
        return None
    ids = lambda f: frozenset(p.split(":")[1] for p in fl[f]["pairs"].split(";") if p)  # noqa: E731
    jumps, hit = 0, 0
    for (f0, a0), (f1, a1) in zip(ape[:-1], ape[1:]):
        if 481 <= f1 <= 790 and abs(a1 - a0) > 0.03:
            jumps += 1
            hit += int(ids(f1) != ids(f0))
    return dict(jumps=jumps, on_pair_change=hit, frac=(hit / jumps) if jumps else None)


def seg_table(cols):
    L = ["| Đoạn | Chỉ số | " + " | ".join(c for c, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
    rows = [("ATE cục bộ (cm)", "ate_local_cm", "{:.2f}"), ("RPE dịch Δ1 (cm)", "rpe_trans_cm_d1", "{:.2f}"),
            ("RPE dịch Δ15 trượt (cm)", "rpe_trans_cm_d15_slide", "{:.2f}"), ("Tỉ lệ quãng đường", "path_ratio", "{:.2f}"),
            ("Không căn đầu→cuối (cm)", None, None), ("e_free / e_obs V_scene (cm)", "ef", None),
            ("e_free / e_obs V_track (cm)", "eft", None)]
    for s in ("S1", "S2", "S3", "S4", "S5"):
        for name, key, f in rows:
            vals = []
            for _, m in cols:
                g = m["segments"][s]
                if key is None:
                    vals.append(f"{vn(g.get('raw_start'), '{:.1f}')} → {vn(g.get('raw_end'), '{:.1f}')}")
                elif key == "ef":
                    vals.append(f"{vn(g.get('e_free_rms_cm'))} / {vn(g.get('e_obs_rms_cm'))}")
                elif key == "eft":
                    vals.append(f"{vn(g.get('e_free_rms_cm_track'))} / {vn(g.get('e_obs_rms_cm_track'))}"
                                if g.get("e_free_rms_cm_track") is not None else "—")
                else:
                    vals.append(vn(g.get(key), f))
            L.append(f"| {s} | {name} | " + " | ".join(vals) + " |")
    return L


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


mode, seeds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
if mode == "c":
    runs = {k: f"{R}/e1c/fr3_snt_far/dec/seed{k}" for k in seeds}
    ms = {k: sm(runs[k]) for k in seeds}
    per = {}
    for k in seeds:
        ch = c_checks(ms[k])
        Rk = ms[k]["ate_all_cm"] / B
        fail = failed(runs[k], ms[k])
        n_miss = sum(not ch[c] for c in ("C1", "C2", "C3"))
        if fail or Rk > 1.1:
            cls = "Tệ hơn"
        elif Rk <= 0.5 and n_miss == 0:
            cls = "Vượt trội"
        elif Rk <= 0.9 or (Rk <= 0.5 and n_miss == 1):
            cls = "Tốt hơn, chưa chắc"
        else:
            cls = "Ngang base"
        dj = dup_jitter(runs[k])
        dup = (ch["C1"] and ch["C3"] and not ch["C2"] and ch["C2_parts"]["S3_ate"] and dj and dj["frac"] is not None
               and dj["frac"] >= 0.5)
        per[k] = dict(R=Rk, fail=fail, checks=ch, cls=cls, dup_jitter=dj, dup_flag=bool(dup))
    if len(seeds) == 1:
        k = seeds[0]
        cls = per[k]["cls"]
        if cls == "Vượt trội":
            verdict, action = "Vượt trội", "chốt E1c (tag e1c-locked), sang D0"
        elif cls == "Tệ hơn":
            verdict, action = "Tệ hơn", "chạy lại T10–T13; nếu đạt: bỏ D0–D3, sang F"
        else:
            verdict, action = cls, "chạy dec seed 1"
    else:
        Rm = float(np.mean([per[k]["R"] for k in seeds]))
        all_c = all(per[k]["checks"][c] for k in seeds for c in ("C1", "C2", "C3"))
        if any(per[k]["cls"] == "Tệ hơn" for k in seeds):
            verdict, action = "Tệ hơn (một seed)", "chạy lại T10–T13; nếu đạt: bỏ D0–D3, sang F"
        elif Rm <= 0.7 and all_c:
            verdict, action = "Chốt", "chốt E1c (tag e1c-locked), sang D0"
        else:
            verdict, action = "Chưa rõ", "ghi chẩn đoán D, vẫn sang D0"
    diag = None
    if verdict not in ("Vượt trội", "Chốt") and "Tệ hơn" not in verdict:
        c2_major = sum(per[k]["checks"]["C2"] for k in seeds) > len(seeds) / 2
        ef = [ms[k]["e_free_obs_S3uS5"]["scene"] for k in seeds]
        ratio = [e["e_free_rms_cm"] / e["e_obs_rms_cm"] for e in ef]
        diag = dict(C2_majority=bool(c2_major), e_free_over_e_obs=ratio,
                    conclusion="E1c chưa đủ, cần E1d" if (c2_major and all(r >= 2 for r in ratio)) else
                    "điều kiện chẩn đoán D không đạt (theo luật mới vẫn sang D0)")
    out_dir, tag = f"{R}/e1c", "C"
    cols = [("base s0", sm(REUSED["base"])), ("E1 s0", sm(REUSED["ours"]))] + [(f"E1c s{k}", ms[k]) for k in seeds]
else:
    runs = {k: f"{R}/e1d/fr3_snt_far/dec_edge/seed{k}" for k in seeds}
    ms = {k: sm(runs[k]) for k in seeds}
    dec_runs = sorted(d for d in os.listdir(f"{R}/e1c/fr3_snt_far/dec") if d.startswith("seed")
                      and os.path.exists(f"{R}/e1c/fr3_snt_far/dec/{d}/seg_metrics.json"))
    dms = {d: sm(f"{R}/e1c/fr3_snt_far/dec/{d}") for d in dec_runs}
    E_bar = float(np.mean([m["ate_all_cm"] for m in dms.values()]))
    ref = dms["seed0"]
    per = {}
    for k in seeds:
        m = ms[k]
        ef_d = m["e_free_obs_S3uS5"]["scene"]["e_free_rms_cm"]
        ef_c = float(np.mean([x["e_free_obs_S3uS5"]["scene"]["e_free_rms_cm"] for x in dms.values()]))
        S, Sc = m["segments"], ref["segments"]
        d1 = ef_d <= 0.5 * ef_c
        d2 = S["S2"]["ate_local_cm"] <= 1.2 * Sc["S2"]["ate_local_cm"] + 0.3 and S["S4"]["ate_local_cm"] <= 1.2 * Sc["S4"]["ate_local_cm"] + 0.3
        d3 = m["ate_all_cm"] <= E_bar
        d4 = S["S3"]["ate_local_cm"] <= 1.8 and S["S5"]["rpe_trans_cm_d1"] <= 1.02
        ratio_free = ef_d / ef_c
        if m["ate_all_cm"] > 1.1 * E_bar or not d2:
            cls = "Gây hại"
        elif ratio_free > 0.8:
            cls = "Không tác dụng"
        elif d1 and d2 and d3 and d4 and m["ate_all_cm"] <= 0.7 * E_bar:
            cls = "Vượt trội"
        else:
            cls = "Chưa chắc"
        per[k] = dict(E_bar=E_bar, e_free=ef_d, e_free_dec=ef_c, D1=d1, D2=d2, D3=d3, D4=d4, cls=cls,
                      ratio_E=m["ate_all_cm"] / E_bar, ratio_free=ratio_free)
    if len(seeds) == 1:
        cls = per[seeds[0]]["cls"]
        verdict = cls
        action = {"Vượt trội": "chốt E1d (tag e1d-locked), sang F", "Chưa chắc": "chạy dec_edge seed 1 (và dec seed 1 nếu chưa có)",
                  "Không tác dụng": "chạy lại T17–T18 một lần, sang F (dec_edge là kết quả âm)",
                  "Gây hại": "chạy lại T17–T18 một lần, sang F (dec_edge là kết quả âm)"}[cls]
    else:
        verdict = "Chốt" if all(per[k]["cls"] == "Vượt trội" for k in seeds) else "Kết cục sau 2 seed: " + ", ".join(per[k]["cls"] for k in seeds)
        action = "sang F"
    diag = None
    out_dir, tag = f"{R}/e1d", "D"
    cols = [("base s0", sm(REUSED["base"])), ("E1 s0", sm(REUSED["ours"]))] + \
           [(f"E1c {d.replace('seed', 's')}", dms[d]) for d in dec_runs] + [(f"E1d s{k}", ms[k]) for k in seeds]

os.makedirs(out_dir, exist_ok=True)
name = f"DECISION_{tag}_seeds{'_'.join(map(str, seeds))}"
L = [f"# E1{'c' if mode == 'c' else 'd'} — quyết định sau seed {', '.join(map(str, seeds))}\n",
     f"Commit `{git_head()}` · branch hiện tại · RTX 5060 Ti · run: " + ", ".join(f"`{v}`" for v in runs.values()) + "\n",
     f"## 1. Verdict: **{verdict}** → {action}\n", "## 2. Bảng quyết định\n", "| Điều kiện | Ngưỡng | Giá trị | Đạt? |", "|---|---|---|---|"]
if mode == "c":
    for k in seeds:
        m, p = ms[k], per[k]
        S = m["segments"]
        L += [f"| seed {k}: R = ATE-all / B | ≤ 0,5 vượt trội; ≤ 1,1 không tệ hơn | {vn(m['ate_all_cm'])} / {vn(B)} = {vn(p['R'], '{:.3f}')} | {p['cls']} |",
              f"| seed {k}: C1 không căn cuối S1 | ≤ 8,6 cm | {vn(S['S1']['raw_end'], '{:.1f}')} | {'✓' if p['checks']['C1'] else '✗'} |",
              f"| seed {k}: C2 ATE cục bộ S3 | ≤ 3,4 cm | {vn(S['S3']['ate_local_cm'])} | {'✓' if p['checks']['C2_parts']['S3_ate'] else '✗'} |",
              f"| seed {k}: C2 RPE dịch Δ1 S5 | ≤ 1,53 cm | {vn(S['S5']['rpe_trans_cm_d1'])} | {'✓' if p['checks']['C2_parts']['S5_rpe'] else '✗'} |",
              f"| seed {k}: C2 tỉ lệ quãng đường S5 | ≤ 1,73 | {vn(S['S5']['path_ratio'])} | {'✓' if p['checks']['C2_parts']['S5_path'] else '✗'} |",
              f"| seed {k}: C3 ATE cục bộ S2 | ≤ 1,9 cm | {vn(S['S2']['ate_local_cm'])} | {'✓' if p['checks']['C3_parts']['S2'] else '✗'} |",
              f"| seed {k}: C3 ATE cục bộ S4 | ≤ 3,2 cm | {vn(S['S4']['ate_local_cm'])} | {'✓' if p['checks']['C3_parts']['S4'] else '✗'} |"]
        dj = p["dup_jitter"]
        if dj:
            L.append(f"| seed {k}: giật S5 trùng lúc đổi anchor | ≥ 50 % (chỉ khi C2 fail riêng ở S5) | {dj['on_pair_change']}/{dj['jumps']} bước nhảy APE > 3 cm | {'có' if p['dup_flag'] else 'không'} |")
else:
    for k in seeds:
        p = per[k]
        L += [f"| seed {k}: D1 e_free S3∪S5 (V_scene) | ≤ 0,5 × E1c ({vn(p['e_free_dec'])}) | {vn(p['e_free'])} (×{vn(p['ratio_free'], '{:.2f}')}) | {'✓' if p['D1'] else '✗'} |",
              f"| seed {k}: D2 ATE cục bộ S2, S4 | ≤ 1,2 × E1c + 0,3 cm | xem bảng 3 | {'✓' if p['D2'] else '✗'} |",
              f"| seed {k}: D3 ATE-all | ≤ Ē = {vn(p['E_bar'])} | {vn(ms[k]['ate_all_cm'])} (×{vn(p['ratio_E'], '{:.2f}')}) | {'✓' if p['D3'] else '✗'} |",
              f"| seed {k}: D4 ATE cục bộ S3 ≤ 1,8 và RPE Δ1 S5 ≤ 1,02 | | {vn(ms[k]['segments']['S3']['ate_local_cm'])} / {vn(ms[k]['segments']['S5']['rpe_trans_cm_d1'])} | {'✓' if p['D4'] else '✗'} |",
              f"| seed {k}: kết cục | | | {p['cls']} |"]
if diag:
    L += ["", f"**Chẩn đoán D**: C2 đạt ở đa số seed: {'có' if diag['C2_majority'] else 'không'}; e_free/e_obs trên S3∪S5 (V_scene): "
          + ", ".join(vn(x) for x in diag["e_free_over_e_obs"]) + f" → {diag['conclusion']}."]
L += ["", "Số seed: " + ", ".join(f"{c} = 1 lần chạy" for c, _ in cols) + ". Với 1 lần chạy, so sánh ghi \"trên 1 lần chạy\".", "",
      "## 3. Bảng đoạn S1–S5\n"] + seg_table(cols)
L += ["", "## 4. Toàn chuỗi\n", "| Run | ATE-all (cm) | ATE-kf (cm) | RPE dịch Δ1 (cm) | RPE Δ15 trượt (cm) | e_free/e_obs S3∪S5 V_scene | V_track |", "|---|---|---|---|---|---|---|"]
for c, m in cols:
    w, e = m["whole"], m.get("e_free_obs_S3uS5") or {}
    sc, tr = e.get("scene"), e.get("track")
    L.append(f"| {c} | {vn(m['ate_all_cm'])} | {vn(m['ate_kf_cm'])} | {vn(w['rpe_trans_cm_d1'])} | {vn(w['rpe_trans_cm_d15_slide'])} | "
             f"{vn(sc['e_free_rms_cm']) + ' / ' + vn(sc['e_obs_rms_cm']) if sc else '—'} | {vn(tr['e_free_rms_cm']) + ' / ' + vn(tr['e_obs_rms_cm']) if tr else '—'} |")
open(os.path.join(out_dir, name + ".md"), "w").write("\n".join(L) + "\n")
json.dump(dict(verdict=verdict, action=action, per_seed={str(k): v for k, v in per.items()}, diag=diag),
          open(os.path.join(out_dir, name + ".json"), "w"), indent=1, default=str)
print(f"{name}: {verdict} -> {action}")
