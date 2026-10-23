"""Append the GT merge check and the final verdict to REPORT_P1_QUICK.md (the generator marks (b) pending)."""
import json

OUT = "results_exp/a2g_p1q"
rep = open(f"{OUT}/REPORT_P1_QUICK.md").read()
i = rep.index("## Quyết định")
head, tail = rep[:i], rep[i:]
L = ["## Kiểm gộp bằng GT (thay cho soát tay)\n",
     "Mỗi lần gộp: hai anchor lấy ở keyframe tạo ra chúng, đưa về world bằng pose GT của keyframe đó; lệch > 3° hoặc "
     "> 3 cm là gộp nhầm thật. Hai anchor cùng một mặt thật lệch nhau 0,7–1,8° và 0,3–1,9 cm (nhiễu lúc tạo).\n",
     "| Chuỗi | Keyframe | Giữ (tạo ở) | Gộp (tạo ở) | Lệch theo pose ước lượng | Lệch theo pose GT | Kết luận |",
     "|---|---|---|---|---|---|---|"]
n_false = {}
for s in ("snt_far", "st_far"):
    rows = json.load(open(f"{OUT}/{s}/p1/seed0/merge_gt_check.json"))
    n_false[s] = (sum(r["gt_false"] for r in rows), len(rows))
    for r in rows:
        L.append(f"| {s} | {r['k']} | {r['keep']} ({r['created'][0]}) | {r['drop']} ({r['created'][1]}) | "
                 f"{r['est_angle_deg']:.2f}°, {r['est_dist_cm']:+.2f} cm | {r['gt_angle_deg']:.2f}°, {r['gt_dist_cm']:+.2f} cm | "
                 f"{'**gộp nhầm**' if r['gt_false'] else 'cùng mặt'} |".replace(".", ","))
L.append("")
L.append(f"Gộp nhầm thật: snt_far {n_false['snt_far'][0]}/{n_false['snt_far'][1]}, "
         f"st_far {n_false['st_far'][0]}/{n_false['st_far'][1]}. Mọi lần nhầm đều là gộp một anchor tạo muộn vào "
         "anchor tạo sớm (cách nhau 300–550 frame), góc thật 1,9–11,2°: phép thử K5 cho qua vì C(t_j, t_k) giữa hai thời "
         "điểm tạo đã lớn (κ_r · Δφ với Δφ cỡ 100–200°), nên cổng gộp rộng tới vài độ.\n")
tail = tail.replace("| (b) | trượt | ", "| (b) | trượt | kiểm bằng GT: "
                    f"{n_false['snt_far'][0] + n_false['st_far'][0]} lần gộp nhầm thật; ")
k = tail.index("**Chưa kết luận**")
tail = tail[:k] + (
    "**Trượt** (theo bảng quyết định: có gộp nhầm thật). Theo plan: dừng, làm phần chẩn đoán mục 10 trước khi sửa.\n\n"
    "Đọc kết quả:\n\n"
    "1. Mục tiêu chính của Phần 1 đạt về bản trùng: đổi ID loại bản trùng trên st_far giảm 99 → 1, cặp gần trùng ở cuối "
    "18 → 1 (snt_far 24 → 1); số anchor primary cuối 30 → 6 (st_far).\n"
    "2. Tracking không tệ đi mà tốt hơn E1c: ATE mọi frame 3,80 → 3,20 cm (snt_far), 10,57 → 3,54 cm (st_far). "
    "1 seed, nhiễu khoảng 15 %; chênh ở st_far lớn hơn nhiều mức nhiễu.\n"
    "3. Cái giá: phép gộp K5 quá rộng với anchor cách xa nhau theo thời gian, gộp nhầm các mặt lệch 2–11°; "
    "NIS EKF trung bình 2,18 (snt_far, đạt) và 1,85 (st_far, trượt nhẹ, chỉ 7 lần cập nhật). Trên st_far JC bỏ cặp "
    "ở 44,8 % frame (snt_far 4,6 %), có thể là hệ quả của các anchor bị gộp nhầm.\n\n"
    "Bước tiếp theo đề xuất (mục 10, chưa làm): replay offline với pose GT để tách lỗi gộp do luật K5 hay do drift; "
    "thử chặn gộp theo góc tuyệt đối hoặc tính C(t_j, t_k) theo quãng đường giữa hai lần quan sát gần nhất thay vì giữa "
    "hai thời điểm tạo; rồi chạy lại riêng chuỗi bị trượt.\n") + tail[k + len("**Chưa kết luận**: có lần nghi gộp nhầm chưa soát tay; các tiêu chí khác xem bảng.") :]
open(f"{OUT}/REPORT_P1_QUICK.md", "w").write(head + "\n".join(L) + "\n" + tail)
print("finalized")
