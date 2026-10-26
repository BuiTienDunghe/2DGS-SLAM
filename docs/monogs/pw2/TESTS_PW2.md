# TESTS_PW2 — test và smoke (PLAN MONOGS-PW2 §5)

Sinh bằng `tools/monogs/pw2/tests_doc_pw2.py`. T8–T11 chạy bằng `python tests/test_pw2.py` (trong `monogs/`) với bộ nhiễu đã khóa trong `pw2_calib.yaml`.

| Test | Kết quả | Số liệu chính |
|---|---|---|
| T8 hiệu chỉnh | đạt | (a) σ ước lượng 0,497° / thật 0,5°; bộ ước lượng R1 cho 0,336° (0,672σ); (b) b trung vị 0,00204 / thật 0,002 |
| T9 ghép `e_first` | đạt | (a) chọn ứng viên [[0, 1]] (e active 4,17, e ứng viên 0,2); (b) ứng viên thành active, số anchor [2, 2]; (c) cả hai inlier → chọn active [[0, 0]]; (d) chế độ `cost` giống R1 trên 10 frame thật: có |
| T10 cơ sở | đạt | (a) không cặp inlier: lệch so với Adam của base 0 (100 bước, gradient ngẫu nhiên); (b) cơ sở dựng từ 3 cặp inlier, trùng cơ sở E1c của đúng các cặp đó |
| T11 PGO | đạt | (a) residual vector hóa lệch 2,73e-14; (b) T4 R1: 11,6 → 2,98 cm; (c) factor sai: w IRLS 0,0582, bị bỏ [24]; (d) odometry không đổi trọng số: có; (e) 70 keyframe, 280 factor: 0,345 s |
| T12 `kf_force_every` | đạt | base, N = 11, snt_far 0–200: 19 keyframe, khoảng cách lớn nhất 11 frame |
| R1 tái lập | đạt | `full` snt_far 0–100: ATE keyframe 1,11 cm, smoke 1 của R1: 0,998 cm (lệch 11 %) |

## Smoke `pw2` (snt_far, `--range 0 200`)

| Chỉ số | Giá trị |
|---|---|
| Chạy hết | có (200 frame, 260 s) |
| NaN | không |
| Log thiếu | không (pgo_poses.jsonl chỉ có dòng khi có PGO) |
| % cặp bị kernel loại | 1,69 % |
| Ứng viên tạo / xóa / kích hoạt | 4 / 0 / 1 |
| "Đủ" | 71,4 % |
| Keyframe theo lý do | {'init': 1, 'mono': 7} |
| ATE keyframe | 1,01 cm |

**Ghi chú test:**
- **T10 (a)** kiểm trên dữ liệu tổng hợp, không trên 10 frame thật. Khi không có cặp inlier, `DecoupledPoseStep` có cơ sở đơn vị và không hướng nào nhận gradient plane. Test so 100 bước với gradient ngẫu nhiên giữa bước đó và `torch.optim.Adam` (đường cập nhật của `base`): hai đường trùng nhau tuyệt đối. Kết luận áp được cho frame thật, vì đường cập nhật chỉ phụ thuộc gradient render.
- **T11 (b):** lần chạy đầu dùng seed drift khác với T4 của R1 (sai số giảm 5,8 → 3,9 cm, chưa tới 50 %). Đã sửa để chạy đúng bài toán T4 của R1 (seed 7), như plan yêu cầu.
- **T8:** lần chạy đầu lỗi import (thiếu đường dẫn `tools/monogs`); đã sửa.

