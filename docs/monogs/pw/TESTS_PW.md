# TESTS_PW — test và smoke (PLAN MONOGS-PW §6, §7.2)

Sinh bằng `tools/monogs/pw/tests_doc_pw.py` từ `tests_pw.json` (chạy `python tests/test_pw.py` trong `monogs/`) và log smoke. Mô hình nhiễu dùng khi test: bộ số đã khóa trong `pw_calib.yaml`.

| Test | Kết quả | Số liệu chính |
|---|---|---|
| T1 | đạt | (a) 2 mảnh → anchor [[0, 0], [1, 0]]; (b) mặt song song 6 cm: e = 7,71, w = 0,0173; (c) phần xa của mặt lớn: ghép, inlier; (d) khởi tạo bền bỏ 1 cặp, sai 1,33e-14 cm (E1c: 2,88 cm); (e) chế độ e1c giống code cũ: ghép có, lệch loss 0 |
| T2 | đạt | (a) sai refresh 0; (b) n_obs_kf 2; (c) còn sống sau 0/1/2 keyframe: [True, True, False]; (d) {'0': 1, '1': 1} |
| T3 | đạt | (a) tier 2 = [[0, 0]]; (b) hai anchor cũ trong cổng: []; (c) {'passed': True, 'old': True, 'window_obs': False, 'gap6': False} |
| T4 | đạt | (a) phần dư lớn nhất 0; (b) sai số TB 11,6 → 2,91 cm; (c) factor bị bỏ [24] |
| T5 | trượt | sai dịch 3,68 cm, sai quay 0,0634°; thành phần dịch thật dọc giao tuyến hai mặt 3,98 cm |
| T6 | đạt | (a) sai xyz 3,75e-07, sai quay 3,87e-07; (b) depth lệch lớn nhất 0,198 mm trên 291591 pixel hợp lệ (p99 0,00334 mm; mọi pixel có depth: lớn nhất 0,844 mm) |
| T7 | đạt | (a) R, T khôi phục từng bit: có; (b) Δ_k nhỏ nhất 0,000137; (c) cân bằng 0,02 / đích 0,02; (d) κ ước lượng 36,8 / thật 37.0; (e) loss 0,355088106 / công thức 0,355088106 |

**Ghi chú:**
- **T4:** lần chạy đầu dùng drift nhân trái lên w2c (drift quay theo hướng camera). Khi đó chính quỹ đạo thật có cost odometry 108, cao hơn nghiệm tối ưu 10,7, nên PGO không thể đưa về quỹ đạo thật. Test được sửa theo mô hình drift odometry chuẩn: mỗi chuyển động tương đối mang cùng một sai lệch 0,5 cm / 0,2° trong hệ thân. Solver không đổi.
- **T4c, solver:** `least_squares` dừng vì chạm giới hạn số lần đánh giá (status 0), mà lúc đầu code coi đó là lỗi. Plan đặt "tối đa 100 vòng", nên đã sửa để chấp nhận status 0.
- **T5 trượt:** với **đúng 2 mặt**, dịch dọc giao tuyến hai mặt không quan sát được, cả bằng nghiệm dạng đóng (nghiệm chuẩn nhỏ nhất) lẫn bằng ICP điểm–mặt. Sai số đo được gần bằng thành phần dịch thật dọc giao tuyến. Đây là giới hạn của closure kiểu StructureGS khi chỉ có 2 mặt; không sửa.
- **T6:** lần đầu so mọi pixel có depth và lệch 1,3 mm ở viền vật. Đã sửa 1 lần: chỉ so pixel hợp lệ theo quy ước MonoGS (opacity > 0,95 ở cả hai lần render). Kèm số trên mọi pixel để đối chiếu.
- **T7:** hai lỗi của chính test: thiếu `compute_grad_mask` cho camera, và sai dấu bước của `ManualAdam`. `ManualAdam` trả về chính bước cập nhật.

## Smoke (mục 7.2)

**Smoke 1: `full`, snt_far, `--range 0 100`**

| Chỉ số | Giá trị |
|---|---|
| Chạy hết | có (100 frame, 77 s, 769 ms/frame) |
| NaN trong quỹ đạo | không |
| Log thiếu | không |
| Keyframe theo lý do | init: 1, mono: 3 |
| % cặp ứng viên bị kernel loại | 20 % |
| Trung vị phần chia s_k | 0,182 |
| Số lần PGO | 0 (—) |
| Độ dịch keyframe lớn nhất sau mỗi PGO (cm) | — |
| ATE keyframe (cm) | 0,998 |

**Smoke 2: `full`, snt_far, `--range 0 200`, ép PGO mỗi 5 keyframe**

| Chỉ số | Giá trị |
|---|---|
| Chạy hết | có (200 frame, 213 s, 1063 ms/frame) |
| NaN trong quỹ đạo | không |
| Log thiếu | không |
| Keyframe theo lý do | init: 1, mono: 7, T1: 4 |
| % cặp ứng viên bị kernel loại | 26,9 % |
| Trung vị phần chia s_k | 0,107 |
| Số lần PGO | 2 (applied, applied) |
| Độ dịch keyframe lớn nhất sau mỗi PGO (cm) | 2,02, 2,03 |
| ATE keyframe (cm) | 1,16 |

