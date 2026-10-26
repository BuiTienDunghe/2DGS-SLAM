1. **Watchdog 10 phút** (`WATCHDOG_IDLE_S=600`) cho mọi run lượt này, thay vì 30 phút. Ở R1, một run treo lúc thoát làm mất 30 phút. Điều này an toàn vì PGO giờ chỉ mất dưới 1 giây.
2. **N của `base_kf`** tính từ số frame thật của run 1 (789) chia số keyframe (49), làm tròn: 16, đúng như plan.
3. **T10 (a)** kiểm trên gradient tổng hợp, không trên 10 frame thật (xem `TESTS_PW2.md`).
4. **σ_eff dùng một bộ (a, b) cho mọi khoảng cách frame,** đúng plan, dù bảng theo khoảng cách cho thấy phần dư tăng theo khoảng cách.
5. **Kết luận lượt:** D1 không đạt nên theo bảng mục 8 không kết luận về D2–D4. Report vẫn ghi số và luật của D2–D4 để tham khảo, cùng diễn giải vì sao D1 trượt (ghép nhầm, bản trùng; không phải thiếu cặp).
6. **Độ chính xác closure** không xác định (không closure nào được nhận) nên D3 không đạt theo luật, dù ATE `pw2` tốt hơn `pw2_nopgo` trên cả hai chuỗi snt.
7. **Run cũ** (base, e1c ở `results_monogs/B`) không có `traj_all_corr.txt`, nên ATE mọi frame và RPE của chúng tính trên quỹ đạo thô (đánh dấu * trong bảng), như plan cho phép.
8. **Hai lần chạy test bị lỗi của chính test** (T8 thiếu đường dẫn import, T11b dùng seed khác T4 R1) đã sửa 1 lần mỗi lỗi.
