**Diễn giải (1 seed):**
1. **Plane world không còn "đói".** Sửa A (σ_eff) và B (`e_first`) có tác dụng đúng như chẩn đoán:
   - chỉ 2,6–4,1 % cặp đúng nhãn bị kernel loại (R1: 37–57 %);
   - "đủ" đạt 50–59 % frame, ngang E1c (47–66 %);
   - T1 chỉ sinh 0–9 keyframe (R1: 15–57).

   D1 trượt ở **ghép nhầm** (snt_far 38,2 so với E1c 37,3 %; st_far 29,8 so với 26,4 %) và **bản trùng** trên st_far (2 so với 1); trên snt_near thì đạt. Vì thế câu "plane world vẫn đói" của bảng mục 8 không đúng với dữ liệu, dù về mặt luật D1 không đạt. Lý do: σ_eff lớn hơn R1 khoảng 3 lần, nên kernel nhận lại cả cặp sai. Kiểm offline đã báo trước điều này: 35–55 % anchor sai nhãn trong cổng có e ≤ 3.
2. **Lợi trên snt đến từ đồ thị mặt phẳng, không đến từ closure.**
   - `pw2` hơn hẳn `pw2_nopgo`: 2,50 so với 5,98 cm (snt_far) và 6,78 so với 18,96 cm (snt_near).
   - **Không closure nào được nhận:** 5 ứng viên tier 2 đều sai nhãn, đều bị kiểm χ² bỏ đúng. Phần đóng góp nằm ở chỗ mỗi lần PGO tối ưu lại pose của mọi keyframe cùng mọi quan sát mặt phẳng, như một bước BA toàn cục bằng mặt phẳng.
   - Theo luật, D3 không đạt (độ chính xác closure không xác định).
   - Keyframe thêm **không** giải thích được lợi này: `base_kf` (51 keyframe) cho 5,54 cm, so với 2,50 cm của `pw2` (D4 đạt).
3. **st_far tệ hơn E1c** (2,15 so với 1,16 cm). Ở cảnh có texture, không có PGO nào chạy (không có ứng viên closure), và ghép nhầm tăng (29,8 %). Trên snt_near, `pw2` ngang E1c (6,78 so với 7,51 cm, trong nhiễu).
4. **Giật (D5):** RPE (hiệu chỉnh) của `pw2` thấp hơn E1c trên st_far và snt_near, cao hơn trên snt_far (1,59 so với 1,39). Ở cả ba chuỗi đều cao hơn base.

