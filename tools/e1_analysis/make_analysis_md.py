"""Build results_exp/e1/analysis/ANALYSIS.md from q123.json, q4.json and logs/ (every number read from files).
usage: python tools/e1_analysis/make_analysis_md.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, SEGMENTS, vn  # noqa: E402

q = json.load(open(os.path.join(OUT, "q123.json")))
q4 = json.load(open(os.path.join(OUT, "q4.json")))
Q1, R, Q2, Q3 = q["q1"], q["q1_raw"], q["q2"], q["q3"]
lam = q["lambda_plane"]
S = [s[0] for s in SEGMENTS]


def bo(seg, key, fmt="{:.2f}"):
    return f"{vn(Q1[seg]['base_' + key], fmt)} / {vn(Q1[seg]['ours_' + key], fmt)}"


L = []
L.append("# Phân tích bổ sung E1 — fr3_snt_far, seed 0 (chỉ đọc kết quả, không chạy lại SLAM)\n")
L.append("Nguồn: `results_exp/e1/fr3_snt_far/{base,ours}/seed0/` (không sửa). Code: `tools/e1_analysis/`. "
         "Đoạn: " + ", ".join(f"{n} = {a}–{b}" for n, a, b in SEGMENTS) + " (chỉ số dataset).\n")

# ------------------------------------------------------------------ 0. verdict
s1, s3, s5, s2, s4 = Q1["S1"], Q1["S3"], Q1["S5"], Q1["S2"], Q1["S4"]
r1, r3, r4 = R["S1"], R["S3"], R["S4"]
L.append("## 0. Kết luận: lỗi nằm ở đâu\n")
L.append("| Giả thuyết | Kết luận | Bằng chứng (đọc từ file) |")
L.append("|---|---|---|")
L.append(
    f"| `plane_loss` vẫn bật ở đoạn suy biến (σ_min < 0,3) | **Đúng, đây là nơi sinh ra gần hết lỗi; nhưng suy biến thôi chưa đủ để hỏng** | "
    f"S3 (σ_min trung vị {vn(s3['sigma_min_median_ours'], '{:.3f}')}, {vn(s3['pct_init_plane_ours'], '{:.0f}')} % khởi tạo bằng plane): sai số không căn của ours tăng "
    f"{vn(r3['ours']['start'], '{:.1f}')} → {vn(r3['ours']['end'], '{:.1f}')} cm trong {s3['n_tracked']} frame, base chỉ "
    f"{vn(r3['base']['start'], '{:.1f}')} → {vn(r3['base']['end'], '{:.1f}')} cm; RPE dịch Δ15 base/ours {bo('S3', 'rpe_trans_cm_d15')} cm. "
    f"S5 (σ_min {vn(s5['sigma_min_median_ours'], '{:.3f}')}): RPE dịch Δ1 {bo('S5', 'rpe_trans_cm_d1')} cm, tỉ lệ quãng đường "
    f"{bo('S5', 'path_ratio')} (pose giật qua lại). Ngược lại ở S2/S4 (σ_min ≈ 1, plane khởi tạo "
    f"{vn(s2['pct_init_plane_ours'], '{:.0f}')} %/{vn(s4['pct_init_plane_ours'], '{:.0f}')} % frame), ours **tốt hơn** base: "
    f"ATE cục bộ {bo('S2', 'ate_local_cm')} và {bo('S4', 'ate_local_cm')} cm. Phản ví dụ: S1 cũng suy biến "
    f"(σ_min {vn(s1['sigma_min_median_ours'], '{:.3f}')}, plane_loss bật) mà ours tốt hơn (cuối S1 lệch {vn(r1['ours']['end'], '{:.1f}')} so với "
    f"{vn(r1['base']['end'], '{:.1f}')} cm); khác biệt là anchor ở S1 sinh từ pose GT của frame 0 |")
L.append(
    f"| Trọng số plane quá mạnh (λ = {vn(lam, '{:.1f}')}, σ_n 0,02 rad, σ_d 1 cm) | **Có góp phần; run này không tách riêng được** | "
    f"λ·loss_plane / loss_render trung vị {vn(Q3['all']['ratio_med'])}, phân vị 90 {vn(Q3['all']['ratio_p90'])} (cả chuỗi). "
    f"Ở S3 loss render của ours trung vị {vn(Q3['S3']['render_med'], '{:.3f}')} so với base {vn(Q3['S3']['base_render_med'], '{:.3f}')}: "
    f"plane kéo pose ngược với bằng chứng ảnh, và ours chỉ đi {vn(s3['ours_path_ratio'])} quãng đường thật (base {vn(s3['base_path_ratio'])}): "
    f"pose bị ghì lại. Cơ chế khả dĩ (chưa chứng minh): với 2 hướng mặt, plane không ràng buộc hướng còn lại, nhưng gradient plane lớn "
    f"làm Adam co bước theo mọi thành phần của τ, nên loss render không kéo kịp pose theo hướng tự do. Cùng trọng số lại cho kết quả tốt ở S2/S4. "
    f"Muốn tách vai trò của trọng số cần quét λ (Phase 4), chưa chạy |")
L.append(
    f"| Anchor không được sửa theo pose đã hiệu chỉnh | **Không đúng theo nghĩa đen; dạng gần với nó là nghi phạm chưa kiểm được** | "
    f"Với loop/PGO tắt, backend không tối ưu pose (`slam_backend.py` chỉ đổi pose trong nhánh PGO), nên không có pose "
    f"\"đã hiệu chỉnh\" nào để sửa anchor theo. Dạng gần: anchor sinh từ pose đã trôi rồi đóng băng. Có hai dấu hiệu: (a) phản ví dụ S1 ở dòng 1; "
    f"(b) ở S4 sai số không căn của ours đứng yên {vn(r4['ours']['min'], '{:.1f}')}–{vn(r4['ours']['max'], '{:.1f}')} cm, tức khởi tạo bằng plane "
    f"giữ nguyên độ lệch đã có (không thêm, không bớt). Kiểm tra trực tiếp ở mức anchor (Q4) dừng vì replay chỉ khớp "
    f"{vn(100 * q4['match_rate'], '{:.1f}')} % < 95 % |")
L.append("")
L.append(
    f"**Đính chính báo cáo trước** (`BAO_CAO_E1.md` §3): nhận định \"ours bị khoá vào anchor sai, APE đứng ở ~24 cm trong S2\" là sai. "
    f"Mức 24 cm trên F2 đến từ phép căn SE(3) toàn chuỗi, bị kéo bởi lỗi lớn ở S3/S5. Không căn chỉnh thì cuối S2 ours lệch "
    f"{vn(R['S2']['ours']['end'], '{:.1f}')} cm, base {vn(R['S2']['base']['end'], '{:.1f}')} cm.\n")
L.append(
    f"**Q2**: không thấy tracking bằng render dừng sớm rồi đánh giá thấp chuyển động. iters_done trung vị 120 ở mọi đoạn; "
    f"frame dừng sớm: base {vn(Q2['base']['all']['pct_lt_120'], '{:.1f}')} %, ours {vn(Q2['ours']['all']['pct_lt_120'], '{:.1f}')} %. "
    f"Spearman(iters_done, r_k) cả chuỗi: base {vn(Q2['base']['all']['spearman_iters_vs_r'])}, ours {vn(Q2['ours']['all']['spearman_iters_vs_r'])}.\n")
n15 = [Q1[s]["ours_n_pairs_d15"] for s in S]
L.append(
    "**Tóm lại**: lỗi sinh ra khi `plane_loss` chạy trong cấu hình suy biến với anchor không còn khớp pose thật "
    f"(S3: +{vn(r3['ours']['change'], '{:.1f}')} cm không căn; S5: pose giật). Khởi tạo bằng plane khi đủ 3 hướng mặt có ích về cục bộ "
    "nhưng giữ nguyên độ lệch có sẵn. Hướng sửa đúng với số liệu này: tắt `plane_loss` khi σ_min < ngưỡng (dùng cùng cổng với "
    "`solve_pose`), rồi đo lại; quét λ để biết trọng số còn gây hại khi đã có cổng không.\n")
L.append(f"Giới hạn: 1 seed; RPE Δ15 theo cách ghép cặp của evo chỉ có {min(n15)}–{max(n15)} cặp mỗi đoạn "
         "(cột cửa sổ trượt ở bảng 1b có nhiều cặp hơn).\n")

# ------------------------------------------------------------------ Q1
L.append("## 1. Q1 — theo đoạn\n")
L.append("Mỗi ô ghi base / ours. RPE: RMSE theo định nghĩa evo (cặp i, i+Δ với i = 0, Δ, 2Δ, …, cả hai frame trong đoạn). "
         "ATE cục bộ: căn SE(3) riêng từng đoạn. Tỉ lệ quãng đường = Σ‖Δt_est‖ / Σ‖Δt_GT‖.\n")
L.append("| Đoạn | σ_min trung vị (ours) | RPE dịch Δ1 (cm) | RPE quay Δ1 (°) | RPE dịch Δ15 (cm) | RPE quay Δ15 (°) | ATE cục bộ (cm) | Tỉ lệ quãng đường |")
L.append("|---|---|---|---|---|---|---|---|")
for s in S:
    L.append(f"| {s} | {vn(Q1[s]['sigma_min_median_ours'], '{:.3f}')} | {bo(s, 'rpe_trans_cm_d1')} | {bo(s, 'rpe_rot_deg_d1', '{:.3f}')} | "
             f"{bo(s, 'rpe_trans_cm_d15')} | {bo(s, 'rpe_rot_deg_d15', '{:.3f}')} | {bo(s, 'ate_local_cm')} | {bo(s, 'path_ratio')} |")
L.append("\n**1b. Bổ sung**\n")
L.append("| Đoạn | Frame track | % frame khởi tạo bằng plane (ours) | r_k trung vị | r_k trung bình | Số cặp Δ1 / Δ15 | RPE dịch Δ15 cửa sổ trượt (cm) | Số cặp trượt |")
L.append("|---|---|---|---|---|---|---|---|")
for s in S:
    v = Q1[s]
    L.append(f"| {s} | {v['n_tracked']} | {vn(v['pct_init_plane_ours'], '{:.1f}')} | {bo(s, 'r_median')} | {bo(s, 'r_mean')} | "
             f"{v['ours_n_pairs_d1']} / {v['ours_n_pairs_d15']} | {bo(s, 'rpe_trans_cm_d15_slide')} | {v['ours_n_pairs_d15_slide']} |")
L.append("\n**1c. Sai số vị trí không căn chỉnh** (run bắt đầu đúng ở pose GT của frame 0, nên est và GT chung hệ toạ độ)\n")
L.append("| Đoạn | base: đầu → cuối (cm) | base: min–max | ours: đầu → cuối (cm) | ours: min–max |")
L.append("|---|---|---|---|---|")
for s in S:
    b, o = R[s]["base"], R[s]["ours"]
    L.append(f"| {s} | {vn(b['start'], '{:.1f}')} → {vn(b['end'], '{:.1f}')} | {vn(b['min'], '{:.1f}')}–{vn(b['max'], '{:.1f}')} | "
             f"{vn(o['start'], '{:.1f}')} → {vn(o['end'], '{:.1f}')} | {vn(o['min'], '{:.1f}')}–{vn(o['max'], '{:.1f}')} |")
L.append("\n![F6](figs/F6_rpe_segments.png)\n")
L.append("![F10](figs/F10_raw_error.png)\n")

# ------------------------------------------------------------------ Q2
L.append("## 2. Q2 — tracking có dừng sớm và đánh giá thấp chuyển động không\n")
L.append("| Đoạn | iters_done trung vị (base / ours) | % frame < 120 vòng (base / ours) | Spearman(iters_done, r_k) base (p) | ours (p) |")
L.append("|---|---|---|---|---|")
for s in S + ["all"]:
    b, o = Q2["base"][s], Q2["ours"][s]
    L.append(f"| {'cả chuỗi' if s == 'all' else s} | {vn(b['iters_median'], '{:.0f}')} / {vn(o['iters_median'], '{:.0f}')} | "
             f"{vn(b['pct_lt_120'], '{:.1f}')} / {vn(o['pct_lt_120'], '{:.1f}')} | {vn(b['spearman_iters_vs_r'])} ({vn(b['spearman_p'], '{:.3f}')}) | "
             f"{vn(o['spearman_iters_vs_r'])} ({vn(o['spearman_p'], '{:.3f}')}) |")
L.append("\nSpearman của base ở S1 không xác định vì mọi frame đều chạy đủ 120 vòng.\n")
L.append("![F7](figs/F7_motion_ratio_iters.png)\n")

# ------------------------------------------------------------------ Q3
L.append(f"## 3. Q3 — lực của plane_loss so với loss render (ours, λ_plane = {vn(lam, '{:.1f}')})\n")
L.append("| Đoạn | loss_plane trung vị / p90 | loss_render ours trung vị / p90 | loss_render base trung vị / p90 | λ·plane/render trung vị / p90 |")
L.append("|---|---|---|---|---|")
for s in S + ["all"]:
    v = Q3[s]
    L.append(f"| {'cả chuỗi' if s == 'all' else s} | {vn(v['plane_med'], '{:.3f}')} / {vn(v['plane_p90'], '{:.3f}')} | "
             f"{vn(v['render_med'], '{:.3f}')} / {vn(v['render_p90'], '{:.3f}')} | {vn(v['base_render_med'], '{:.3f}')} / {vn(v['base_render_p90'], '{:.3f}')} | "
             f"{vn(v['ratio_med'])} / {vn(v['ratio_p90'])} |")
L.append("\nloss_plane_final và loss_render_final là giá trị ở vòng tối ưu cuối của mỗi frame (`frame_log.csv`).\n")
L.append("![F8](figs/F8_loss_balance.png)\n")

# ------------------------------------------------------------------ Q4
L.append("## 4. Q4 — replay anchor ngoài SLAM: **dừng** (khớp < 95 %)\n")
fr = q4["field_rates"]
L.append(f"| Kiểm tra | Tỉ lệ khớp với `frame_log.csv` |")
L.append("|---|---|")
L.append(f"| Cả 4 trường (n_planes, n_matched_active, σ_min, init_source) | **{vn(100 * q4['match_rate'], '{:.1f}')} %** ({q4['n_frames']} frame) |")
for k, v in fr.items():
    L.append(f"| {k} | {vn(100 * v, '{:.1f}')} % |")
L.append(f"\nFrame lệch đầu tiên: {q4['first_mismatch']}. Theo quy tắc, các mục 1–6 của Q4 và hình F9 **không được tính**.\n")
L.append("Vì sao lệch (log `logs/debug_replay_*.log`):\n")
L.append(f"- Tách plane khớp 100 %, nên lệch không nằm ở CAPE-lite. Chạy replay trên GPU với pose ép float32 (giống run) cho đúng cùng tỉ lệ, nên cũng không do sai số số học CPU/GPU.")
L.append(f"- Tới keyframe 84 số anchor của replay và run trùng nhau. Ở frame {q4['first_mismatch']} cả hai có 6 anchor (5 active), nhưng run ghép một plane vào anchor chưa active còn replay ghép vào anchor active. "
         "Trong replay, plane đó có hai ứng viên là anchor 3 (active) và anchor 5 (chưa active, sinh ở keyframe 84, lệch 0,9° và ~1 cm so với cùng mặt), tức là một **anchor trùng** "
         "(số liệu trong `logs/debug_replay_counts.log` và `logs/debug_replay_frame86.log` ở mục 6).")
L.append("- Run không ghi id anchor được ghép mỗi frame, cũng không ghi tham số anchor, nên không xác định được trạng thái anchor nào của run khác replay.\n")
L.append("Thiếu để hoàn tất Q4: log id anchor được ghép mỗi frame và bảng anchor (n_w, d_w, mẫu) ở mỗi keyframe trong run. "
         "Cần một run mới có ghi thêm các log này (không chạy trong phiên này, theo quy tắc).\n")

# ------------------------------------------------------------------ checks
L.append("## 5. Kiểm tra cài đặt RPE so với evo (cả chuỗi)\n")
L.append("| Run, Δ | RPE dịch tự tính / evo (cm) | RPE quay tự tính / evo (°) |")
L.append("|---|---|---|")
for k, v in q["check"].items():
    L.append(f"| {k} | {vn(v['mine_trans_cm'], '{:.4f}')} / {vn(v['evo_trans_cm'], '{:.4f}')} | {vn(v['mine_rot_deg'], '{:.4f}')} / {vn(v['evo_rot_deg'], '{:.4f}')} |")
L.append("")

# ------------------------------------------------------------------ logs
L.append("## 6. Lệnh đã chạy và output\n")
for p in sorted(glob.glob(os.path.join(OUT, "logs", "*.log"))):
    txt = open(p).read()
    lines = [ln for ln in txt.splitlines() if "Warning" not in ln and "warnings.warn" not in ln]
    if len(lines) > 120:
        lines = lines[:60] + [f"... ({len(lines) - 120} dòng lược, bản đầy đủ: logs/{os.path.basename(p)}) ..."] + lines[-60:]
    L.append(f"**logs/{os.path.basename(p)}**\n")
    L.append("```")
    L.extend(lines)
    L.append("```\n")
open(os.path.join(OUT, "ANALYSIS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:40]))
