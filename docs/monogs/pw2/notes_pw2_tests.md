**Ghi chú test:**
- **T10 (a)** kiểm trên dữ liệu tổng hợp, không trên 10 frame thật. Khi không có cặp inlier, `DecoupledPoseStep` có cơ sở đơn vị và không hướng nào nhận gradient plane. Test so 100 bước với gradient ngẫu nhiên giữa bước đó và `torch.optim.Adam` (đường cập nhật của `base`): hai đường trùng nhau tuyệt đối. Kết luận áp được cho frame thật, vì đường cập nhật chỉ phụ thuộc gradient render.
- **T11 (b):** lần chạy đầu dùng seed drift khác với T4 của R1 (sai số giảm 5,8 → 3,9 cm, chưa tới 50 %). Đã sửa để chạy đúng bài toán T4 của R1 (seed 7), như plan yêu cầu.
- **T8:** lần chạy đầu lỗi import (thiếu đường dẫn `tools/monogs`); đã sửa.
