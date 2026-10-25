"""PLAN A2G-P1 gen §3.2 decisions a/b/c from the A3 pair tables (stage0_sensor.py output).

a. room: mean NIS of the outer band (> 250 px) > 1.5 x inner band (< 150 px) -> undistort depth (and A3 on room
   is recomputed with undistorted depth: run stage0_sensor.py --seqs room --undistort-depth --tag ud first).
b. snt_near: NIS increases with chi2_fit/dof -> R_s <- R_s max(1, chi2_fit/dof). "Increases" = Spearman rank
   correlation > 0 with p < 1 - alpha = 0.01 (the alpha of every plane-world test; no new constant).
c. snt_near, after b: lambda_s = mean NIS / 3; set to 1 if in [0.8, 1.25]. Locked.
usage: python tools/a2g_gen/stage0_a3_decide.py   -> stage0/a3_summary.json, stage0/a3.md
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT0  # noqa: E402

from utils.pw.uncertainty import ALPHA, CHI2_3  # noqa: E402

SEQS = ["snt_near", "snt_far", "st_far", "room"]
RAD = [(0, 150, "< 150 px"), (150, 250, "150–250 px"), (250, 1e9, "> 250 px")]
ZB = [(0, 1, "< 1 m"), (1, 2, "1–2 m"), (2, 3, "2–3 m"), (3, 1e9, "≥ 3 m")]
FX = [(0, 1, "< 1"), (1, 2, "1–2"), (2, 5, "2–5"), (5, 1e18, "≥ 5")]
NC = [(0, 10, "< 10"), (10, 50, "10–50"), (50, 200, "50–200"), (200, 1e9, "≥ 200")]


def f(x, nd=2):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def load(seq, tag):
    p = os.path.join(OUT0, f"a3_{seq}_{tag}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def nis_of(d, fitx=False, lam=1.0):
    fa = np.where(np.isfinite(d["fa"]), np.maximum(1.0, d["fa"]), 1.0) if fitx else np.ones(len(d["fa"]))
    fb = np.where(np.isfinite(d["fb"]), np.maximum(1.0, d["fb"]), 1.0) if fitx else np.ones(len(d["fb"]))
    S = lam * (fa[:, None, None] * d["SA"] + fb[:, None, None] * d["SB"])
    r = d["r"]
    return np.einsum("ni,ni->n", r, np.linalg.solve(S, r[..., None])[..., 0])


def stat(v):
    return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else None, median=float(np.median(v)) if len(v) else None,
                pct_out=float(100 * (v > CHI2_3).mean()) if len(v) else None)


def binned(d, nisv):
    fx = np.fmax(np.nan_to_num(d["fa"], nan=0.0), np.nan_to_num(d["fb"], nan=0.0))
    out = {}
    for key, x, B in (("rad", d["rad"], RAD), ("z", d["z"], ZB), ("fitx", fx, FX), ("ncells", d["nc"], NC)):
        out[key] = [dict(label=lab, **stat(nisv[(x >= lo) & (x < hi)])) for lo, hi, lab in B]
    return out


def main():
    S = {}
    raw = {s: load(s, "raw") for s in SEQS}
    # a (room, raw)
    rb = binned(raw["room"], raw["room"]["nis"])["rad"]
    ratio_a = rb[2]["mean"] / rb[0]["mean"]
    dec_a = bool(ratio_a > 1.5)
    S["a"] = dict(inner=rb[0], outer=rb[2], ratio=ratio_a, undistort_depth=dec_a)
    ud = load("room", "ud")
    if dec_a and ud is None:
        print("decision a = undistort: run stage0_sensor.py --seqs room --undistort-depth --tag ud, then rerun me")
    # b (snt_near)
    d = raw["snt_near"]
    fx = np.fmax(np.nan_to_num(d["fa"], nan=0.0), np.nan_to_num(d["fb"], nan=0.0))
    rho, p = spearmanr(fx, d["nis"])
    dec_b = bool(rho > 0 and p < 1 - ALPHA)
    nis_b = nis_of(d, fitx=dec_b)
    S["b"] = dict(spearman_rho=float(rho), p=float(p), fit_excess=dec_b, mean_after=float(nis_b.mean()))
    # c (snt_near after b)
    lam = float(nis_b.mean() / 3.0)
    lam_locked = 1.0 if 0.8 <= lam <= 1.25 else lam
    S["c"] = dict(lambda_s_raw=lam, lambda_s=lam_locked)
    S["sensor_locked"] = dict(undistort_depth=dec_a, fit_excess=dec_b, lambda_s=lam_locked)
    # tables: raw and final for every sequence (room final with undistorted depth when a says so)
    S["seq"] = {}
    for s in SEQS:
        dr = raw[s]
        dfin = ud if (s == "room" and dec_a and ud is not None) else dr
        nf = nis_of(dfin, fitx=dec_b, lam=lam_locked)
        S["seq"][s] = dict(n_frames=int(dr["n_frames"]), n_seg=int(dr["n_seg"]), raw=stat(dr["nis"]), raw_bins=binned(dr, dr["nis"]),
                           final=stat(nf), final_bins=binned(dfin, nf), final_depth="undistorted" if dfin is ud else "as loaded")
    if ud is not None:
        b = binned(ud, ud["nis"])["rad"]
        S["room_ud_raw"] = dict(all=stat(ud["nis"]), rad=b, ratio=b[2]["mean"] / b[0]["mean"])
    json.dump(S, open(os.path.join(OUT0, "a3_summary.json"), "w"), indent=1)

    L = ["### A3 — mô hình cảm biến (pose GT, σ₀ = 0)\n",
         "Cặp \"cùng mặt\": hai segment cách nhau ≤ 10 frame track, góc world ≤ 3°, hai offset ≤ 3 cm, ≥ 1 ô chung lưới 5 cm "
         f"(chỉ để gán nhãn). NIS₃ của r(s_a | s_b) với S = J_a R_a J_aᵀ + J_b R_b J_bᵀ; kỳ vọng trung bình 3, % > χ²₃(0,99) = {f(CHI2_3)} là 1 %.\n",
         "| Chuỗi | Frame | Segment | Cặp | NIS TB (thô) | NIS trung vị (thô) | % > χ² (thô) | NIS TB (sau khóa) | NIS trung vị (sau khóa) | % > χ² (sau khóa) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SEQS:
        q = S["seq"][s]
        L.append(f"| {s} | {q['n_frames']} | {q['n_seg']} | {q['raw']['n']} | {f(q['raw']['mean'], 1)} | {f(q['raw']['median'], 1)} | "
                 f"{f(q['raw']['pct_out'], 1)} | {f(q['final']['mean'])} | {f(q['final']['median'])} | {f(q['final']['pct_out'], 1)} |")
    L.append("")
    for key, title in (("rad", "khoảng cách tâm segment tới tâm ảnh (lớn hơn của hai segment)"),
                       ("z", "độ sâu (trung bình hai tâm)"), ("fitx", "độ dư của fit χ²_fit/dof (lớn hơn của hai segment)"),
                       ("ncells", "số ô CAPE của segment nhỏ hơn (chẩn đoán thêm, ngoài plan)")):
        L.append(f"**NIS thô theo {title}** (TB / % > χ², số cặp):\n")
        labs = [b["label"] for b in S["seq"]["snt_near"]["raw_bins"][key]]
        L.append("| Chuỗi | " + " | ".join(labs) + " |")
        L.append("|---|" + "---|" * len(labs))
        for s in SEQS:
            L.append(f"| {s} | " + " | ".join(f"{f(b['mean'], 1)} / {f(b['pct_out'], 0)} % ({b['n']})"
                                                for b in S["seq"][s]["raw_bins"][key]) + " |")
        L.append("")
    L.append("**Quyết định** (theo thứ tự của plan):\n")
    L.append(f"- **a (room, thô):** NIS TB dải ngoài {f(S['a']['outer']['mean'], 1)} / dải trong {f(S['a']['inner']['mean'], 1)} = "
             f"{f(ratio_a)} lần {'> 1,5: **bật undistort depth**' if dec_a else '≤ 1,5: không undistort'}.")
    if "room_ud_raw" in S:
        u = S["room_ud_raw"]
        L.append(f"  Tính lại trên room với depth đã undistort: NIS TB {f(u['all']['mean'], 1)}, tỉ lệ dải ngoài/trong {f(u['ratio'])}.")
    L.append(f"- **b (snt_near):** tương quan hạng Spearman giữa NIS và χ²_fit/dof = {f(rho, 3)} (p = {p:.1e}) → "
             f"{'**bật** hệ số fit-excess R_s ← R_s·max(1, χ²_fit/dof)' if dec_b else 'không bật hệ số fit-excess'}; NIS TB sau b = {f(nis_b.mean(), 1)}.")
    L.append(f"- **c (snt_near):** λ_s = NIS TB / 3 = {f(lam, 1)}" + (" → nằm trong [0,8; 1,25], đặt 1." if lam_locked == 1.0 else
                                                                     f" → khóa **λ_s = {f(lam_locked, 1)}** (R_s nhân λ_s)."))
    L.append(f"- **Khối `PlaneWorld.sensor` khóa:** `undistort_depth: {str(dec_a).lower()}`, `fit_excess: {str(dec_b).lower()}`, "
             f"`lambda_s: {lam_locked:.4g}`.\n")
    open(os.path.join(OUT0, "a3.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
