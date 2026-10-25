### Test (mục 5.4)

| Test | Kết quả | Số liệu |
|---|---|---|
| T1 đổi pose | đạt | sai lớn nhất 0.0e+00 (100 pose ngẫu nhiên) |
| T2 quy ước gradient pose | đạt | bản đồ đã tối ưu (500 vòng, 75 % pixel opacity > 0,95): cos 0,997, dấu thành phần lớn trùng; δ = 10⁻³: cos 0,997. Chẩn đoán, bản đồ mới (opacity khoảng 0,5): cos -0,852 |
| T3 ManualAdam | đạt | sai tương đối lớn nhất 0.0e+00 so với torch.optim.Adam (200 bước) |
| T4 test E1c cũ | trượt (test_T3_solve_pose) | 9/10 đạt |

**T2.** Lần chạy đầu dựng bản đồ thẳng từ đám mây điểm của một frame (opacity khoảng 0,5) và cho cos −0,75. Chẩn đoán: với loss màu bỏ thừa số opacity, gradient rasterizer khớp sai phân trung tâm (cos 0,998), và các loss trơn (trung bình depth, trung bình màu có trọng số) khớp với cos 0,92–0,99 theo nhiễu trái `SE3_exp(τ) @ T_w2c`, không khớp theo nhiễu phải. Vậy quy ước τ của rasterizer trùng với loss plane. Phần lệch đến từ thừa số opacity trong loss tracking của MonoGS: rasterizer không đạo hàm theo nhánh opacity, mà nhánh đó lấn át khi opacity thấp. Test được sửa (1 lần) để đo ở trạng thái thật của hệ: bản đồ đã tối ưu như `initialize_map`.

**T4.** `test_T3_solve_pose` trượt **cả trên code gốc của nhánh `exp/e1c-decouple` trong env `2dgs-slam`** (cùng thông báo), nên không do việc port. `T3: clean worst 2.41e-06 deg / 6.35e-16 m; noisy rot median 0.470 max 0.806 deg, centre median 0.746 max 2.459 cm, trials over limit 7/20`. Ngưỡng của test (mọi thử nghiệm có nhiễu dưới 1° và 1 cm) chặt hơn mức nhiễu nó tự sinh; plan yêu cầu giữ nguyên logic E1c, nên không sửa.

### Smoke E1c (snt_far, `--range 0 200`, `--method e1c`)

| Chỉ số | Giá trị |
|---|---|
| Chạy hết | có (200 frame, 9 keyframe, 164 s) |
| NaN trong quỹ đạo | không |
| Nguồn khởi tạo | plane 140, prev 59, rejected 0 |
| ATE keyframe (cm) | 1.00 |
| ms/frame tracking / plane | 519 / 15.9 |

