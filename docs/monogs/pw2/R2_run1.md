# R2 — run 1 (pw2, snt_far đủ chuỗi), mốc P1.5

Sinh bằng `tools/monogs/pw2/make_report_pw2.py --run1`.

### fr3/structure_notexture_far (đủ chuỗi)

| Method | ATE keyframe | ATE mọi frame (hiệu chỉnh*) | RPE Δ1 hiệu chỉnh* / thô | Chưa căn cuối / lớn nhất | Keyframe (mono/T1/T2) | "Đủ" % | % cặp bị kernel loại | % cặp đúng nhãn bị loại | Ứng viên tạo / xóa / kích hoạt | Anchor / mặt thật | Bản trùng | % ghép nhầm | Đổi ID đồng thời / lật | PGO áp | ms/frame | s tổng |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 9,75 | 10,55* | 0,801* / 0,801 | 42,7 / 47,4 | 34 | — | — | — | — / — / — | — | — | — | — / — | — | 453 | 669 |
| e1c | 6,24 | 5,66* | 1,394* / 1,394 | 13,9 / 16,8 | 41 | 53,8 | — | — | — / — / — | 1,55 | 6 | 37,3 | 49 / 378 | — | 555 | 819 |
| full_R1 | 3,51 | 3,69 | 1,144 / 2,239 | 3,9 / 14,4 | 72 (8/57/6) | 8,8 | 58,7 | 38,7 | 218 / 189 / 4 | 1,00 | 0 | 15,8 | 0 / 23 | 5 | 688 | 1739 |
| pw2 | 2,50 | 3,45 | 1,587 / 2,403 | 10,5 / 17,7 | 49 (37/5/6) | 50,0 | 10,3 | 4,1 | 31 / 16 / 10 | 1,29 | 2 | 38,2 | 11 / 121 | 2 | 965 | 1436 |
| pw2_nopgo | không có run | | | | | | | | | | | | | | | | |
| base_kf | không có run | | | | | | | | | | | | | | | | |

\* run không có `traj_all_corr.txt` (run cũ, không PGO): dùng quỹ đạo thô.

**snt_far / pw2** — độ chính xác closure (closure được nhận, có nhãn): — % (0)

| Frame k | Trạng thái | Tier 1 / tier 2 | Closure (anchor, tier, đúng nhãn, bị bỏ, w IRLS) | ATE tập keyframe trước → sau (cm) | Dịch lớn nhất (cm) | s |
|---|---|---|---|---|---|---|
| 689 | applied | 0 / 1 | a21 t2 sai BỎ w=0,12 | 5,72 → 3,36 | 7,97 | 0,6 |
| 722 | applied | 0 / 1 | a21 t2 sai BỎ w=0,08 | 3,59 → 2,47 | 6,84 | 0,7 |

