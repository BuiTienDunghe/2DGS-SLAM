1. **Worktree riêng** `~/2DGS-SLAM-monogs` thay cho `git checkout` trong `~/2DGS-SLAM`. Cây chính đang được một phiên khác dùng (nhánh `deform-v5-test`). Nhánh vẫn là `exp/monogs-e1c`, tạo từ `fork/exp/e1c-decouple` (c80ff00).
2. **Push lên remote `fork`** (github.com/BuiTienDunghe/2DGS-SLAM), như plan cho phép.
3. **Sửa build simple-knn** (`#include <cfloat>`): lần build đầu lỗi `FLT_MAX`; sửa 1 lần, đúng như mục 2.2 cho phép.
4. **Bọc phần vẽ hình của `evaluate_evo`** trong `try`: smoke đầu tiên crash ở cuối run vì evo 1.11 + matplotlib 3.10. Đây là sửa tương thích ngoài danh sách mục 2.3; nó không đổi số ATE.
5. **`pip install` mục 2.2 nâng `wandb` (0.31.0) và `protobuf` (7.36.2)** trong env `monogs`. Cài thêm `pypdf` để đọc paper, vì máy không có `pdftotext`. Env `2dgs-slam` không bị đụng.
6. **Số paper lấy từ bản arXiv tải được** (2312.06741, bản PDF hiện hành): Table 1, fr1/desk RGB-D = 1,50 cm. Paper ghi: ATE trên keyframe, trung bình 3 run, multi-process, RTX 4090.
7. **Cờ tất định để tìm nguồn lệch** (`MONOGS_DETERMINISTIC=1`, mặc định tắt) được thêm vào `utils/repro.py` cho lần thử 1 ở mục 4.2. Run chính không bật cờ này.
8. **sp_r1 chạy ở `cf3148d`, sp_r2 và mt_r1 ở `19dc87f`.** Hook E1c được commit khi sp_r1 đang chạy. Với `--method base`, hook không chạy, nên đường chạy base giống hệt.
9. **T2 sửa 1 lần:** bản đồ được tối ưu 500 vòng (như `initialize_map`) trước khi so gradient. Lý do: với bản đồ mới (opacity khoảng 0,5), loss tracking của MonoGS có thừa số opacity mà rasterizer không đạo hàm theo; chẩn đoán ghi ở TESTS.md.
10. **T4:** `test_T3_solve_pose` trượt ở cả nhánh nguồn trong env `2dgs-slam`; không sửa, vì plan yêu cầu giữ nguyên logic E1c.
11. **Ngoài hai nhãn "E1c tốt hơn base" / "trong nhiễu" của plan,** report thêm nhãn "tệ hơn base (ngoài nhiễu)" khi ATE_e1c > (1 + 2n) × ATE_base. Quy tắc "tốt hơn" được áp cho ATE keyframe (chỉ số chính) và ghi thêm cho ATE mọi frame.
12. **Đo ms tracking có thêm `torch.cuda.synchronize()`** sau mỗi lần tracking (chỉ để đo thời gian; tracking vốn đã đồng bộ ở `get_median_depth`).
13. **ID anchor trong `e1c_frames.jsonl`** là cặp ghép lúc tracking (trước khi ghép lại ở keyframe). Cặp gần trùng được tính trên tập anchor cuối run.
14. **Run cuối (snt_near / e1c_s) treo lúc thoát.** Run đã ghi đủ quỹ đạo và `timing.json` lúc 22:13, rồi không tự thoát (tiến trình backend không dừng). Tôi dừng tiến trình theo PID sau khoảng 5 phút, nên `runs.tsv` ghi rc = 143. Kết quả của run đầy đủ. Các run khác thoát bình thường.
