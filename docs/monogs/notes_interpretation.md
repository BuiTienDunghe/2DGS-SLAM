Theo quy tắc đặt trước ở mục 6.2 (n = 15 %, 1 seed):

1. **MonoGS tái lập được trên fr1/desk.** Đơn luồng 1,43 cm và song song 1,46 cm, so với 1,50 cm của paper, nên r xấp xỉ 1. Hai lần chạy cùng seed vẫn khác nhau vì `atomicAdd` của rasterizer, nhưng ATE chỉ chênh 8 %. Trên các chuỗi này, base MonoGS cho 10,55 cm (snt_far) và 40,36 cm (snt_near), ATE mọi frame. Các số đó nằm trong khoảng mà các paper khác công bố (10,39–16,63 và 26,91–55,59 cm).
2. **E1c giúp MonoGS rõ rệt, nhưng ít hơn trên 2DGS-SLAM.**
   - ATE keyframe giảm 36–41 % (snt_far) và 65–68 % (snt_near), vượt ngưỡng 2n = 30 % ở cả bốn trường hợp, nên đều xếp "E1c tốt hơn base".
   - Tiêu chí "tương tự 2DGS-SLAM" (giảm ATE mọi frame ≥ 50 % trên cả hai chuỗi) **không đạt**: snt_far chỉ giảm 46–48 %, trong khi trên 2DGS-SLAM là −74 % / −84 %.
   - Lưu ý: base MonoGS trên snt_far đã tốt hơn base 2DGS-SLAM (10,55 so với 14,64 cm), nên còn ít chỗ để cải thiện.
   - Theo ATE keyframe (giao thức của các paper), E1c trên MonoGS cho 6,24 cm (snt_far) và 10,93 cm (snt_near). StructureGS-SLAM công bố 7,47 và 16,36 cm. Đây chỉ là tham chiếu: khác máy, khác số run, và tên chuỗi trong các paper chưa chắc trùng.
3. **E1c làm quỹ đạo giật hơn.** RPE Δ1 tăng 1,7–1,9 lần (snt_far) và 1,5–2,1 lần (snt_near), vượt ngưỡng 1,1 lần. Bước pose của plane (khởi tạo dạng đóng ở khoảng 36–54 % frame, cộng gradient plane) đẩy pose theo từng frame. Trong khi đó, mapping và local BA của MonoGS chỉ chỉnh keyframe. E1c cũng làm tăng số keyframe trên snt_near (50 → 84–122).
4. **λ_p = 0,1 và λ_p = 1 cho kết quả trong nhiễu** (−7,7 % trên snt_far, +7,9 % trên snt_near), nên **không có bằng chứng** rằng trọng số plane đang quá lớn. e1c_s còn giật hơn e1c (RPE Δ1 cao hơn), nên độ giật không đến từ độ lớn λ_p. Nhiều khả năng nó đến từ khởi tạo dạng đóng và cách ghép anchor.
5. **Plane world chưa ổn định.** Cuối run có nhiều cặp anchor gần trùng (18–46) và hàng trăm lần đổi ID giữa hai frame liên tiếp. Đây là cùng kiểu lỗi đã thấy với E1c trên 2DGS-SLAM: anchor được trung bình hóa ở keyframe, không có gộp bản trùng.
