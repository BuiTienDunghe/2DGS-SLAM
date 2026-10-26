# DIAG_E1C — chẩn đoán plane world của E1c trên MonoGS (PLAN MONOGS-PW §5)

Sinh bằng `tools/monogs/pw/diag_e1c.py` trên `results_monogs/B/{snt_far,snt_near}/e1c`. Mặt đo đổi về hệ camera bằng `traj_all`, rồi sang world bằng GT; nhãn mặt thật theo mục 8.1. Chỉ để diễn giải, không đổi thiết kế.

| Chuỗi | Anchor mới (sau keyframe đầu) | R1: mảnh khác cùng frame | R2: trong cổng, không ghép (overlap) | R3: ngoài cổng (trôi) | Mới thật | % ghép nhầm | Đổi ID đồng thời | Đổi ID lật |
|---|---|---|---|---|---|---|---|---|
| snt_far | 19 | 4 | 4 | 0 | 11 | 37,3 | 49 | 378 |
| snt_near | 27 | 1 | 4 | 2 | 20 | 34,2 | 96 | 261 |

**RPE (dịch, cm/frame) ở frame có và không có sự kiện:**

| Chuỗi | Có lật ID | Không lật ID | Số frame lật | Khởi tạo bằng plane | Không khởi tạo bằng plane | Số frame plane |
|---|---|---|---|---|---|---|
| snt_far | 1,17 | 0,66 | 307 | 0,80 | 0,92 | 424 |
| snt_near | 2,38 | 0,81 | 204 | 1,21 | 1,06 | 384 |
