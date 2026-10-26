1. **Bỏ hạn chót 21:00 của mục 0.1.** Lượt này bắt đầu lúc 18:25, nên lịch 17:00–21:00 không còn áp được. Người dùng chọn chạy đủ, không cắt run. Không bỏ run nào vì hết giờ.
2. **PGO dùng `scipy.optimize.least_squares`, không dùng GTSAM.** Bản GTSAM trong env không có `OrientedPlane3Factor`. `CustomFactor` cần tự viết Jacobian cho một kiểu landmark mặt phẳng. Plan cho phép đường dự phòng này (mục 6). Cài đặt:
   - pose trái `T_w2c = Exp(ξ)·T0`, keyframe đầu cố định (thay prior σ = 1e-6);
   - mặt phẳng `n = normalize(n0 + B·δθ)`, `d = d0 + δd`;
   - Huber `f_scale = 1,345` áp trên **từng thành phần** phần dư đã chia σ (`loss='huber'` của scipy), không phải trên chuẩn của cả factor;
   - phần dư mặt phẳng 4 chiều [m − R·n_w (3); d] chia [σ_n, σ_n, σ_n, σ_d(z)], đúng phần dư E1c ở mục 2.1, thay cho 3 chiều của `OrientedPlane3Factor`.
3. **Giới hạn số lần đánh giá của solver:** `max_nfev = 100 × max(1, số biến / 50)`. Khi chạm giới hạn (status 0), coi là đã dừng theo giới hạn vòng của plan, không coi là lỗi. T4c lúc đầu bị hủy vì lý do này.
4. **κ không chuyển được giữa hai chuỗi validation** (22 911 so với 516, hơn 40 lần). Theo plan, vẫn dùng một κ chung (2 916) và không đổi số.
5. **σ_odo lấy từ hai run base hiệu chỉnh** (frame 0–300), đúng như phương án "lượt này" của plan: 3,96 cm / 0,77°.
6. **Mô hình nhiễu chỉ gồm nhiễu đo** (đo bằng pose GT): σ_n = 0,24°, σ_d ≈ 0,76 cm. Plan cố ý không dùng mô hình drift. Hệ quả quan sát được trên smoke và run: phần dư của các cặp **đúng** do sai số pose tracking thường ở e = 3–5, nên kernel loại cả nhiều cặp đúng, và ứng viên ít khi được kích hoạt. Không chỉnh theo chuỗi thử; xem mục 6 (gợi ý).
7. **Nhãn mặt thật dùng chung cho mọi run** (mục 8.1), gom theo thứ tự frame trên các mặt đo đã đổi sang world bằng GT. Với E1c, mặt đo trong log ở world ước lượng, nên được đổi về camera bằng `traj_all` trước.
8. **Đổi ID theo nhãn GT** (mục 8.2) đếm trên cặp được ghi sổ (E1c: mọi cặp ghép; PW: cặp inlier). Bảng 8.6 dùng `eval_runs.py` không sửa logic, chỉ thêm `--max-frame`.
9. **Hai cờ CLI chỉ dùng cho hiệu chỉnh và smoke:** `--stiffness-every N` (đo độ cứng mỗi N frame) và `--pgo-force-every N` (ép PGO mỗi N keyframe, smoke 2). Mặc định cả hai tắt.
10. **T6 (b):** ngưỡng 1 mm áp trên pixel hợp lệ theo quy ước MonoGS (opacity > 0,95 ở cả hai lần render). Số trên mọi pixel có depth vẫn được ghi kèm.
11. **sgs:** khi có đúng 2 mặt không song song, dịch dọc giao tuyến không xác định được (T5). Không thêm ràng buộc nào ngoài plan. Lượt này không chạy run `sgs` (không có trong bảng 7.3).
12. **Ghi `traj_all_corr.txt` cho mọi method** chạy ở lượt này. Các run cũ ở `results_monogs/B` không có file này, nên cột "ATE mọi frame hiệu chỉnh" của chúng để trống.
