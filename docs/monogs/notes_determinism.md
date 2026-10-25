**Tìm nguồn lệch còn lại (mục 4.2, 2 lần thử):**

1. **`torch.use_deterministic_algorithms(True, warn_only=True)`** (biến môi trường `MONOGS_DETERMINISTIC=1`, mặc định tắt), chạy lặp `--range 0 30` hai lần: hai run vẫn lệch ngay ở frame 1 (1,14 mm), và torch không báo cảnh báo thao tác bất định nào. Hai run không bật cờ cũng lệch như vậy (1,15 mm ở frame 1). Vậy nguồn lệch không nằm trong các op của PyTorch.
2. **Rà các lời gọi RNG** trong `monogs/` (RGB-D):
   - `torch.randperm` của backend đã được seed;
   - `random.randint` chỉ có trong tinh chỉnh màu, vốn không chạy;
   - `torch.randn_like` và `np.random.randn` chỉ có trong nhánh đơn mắt (monocular);
   - Open3D `random_down_sample` chạy trong backend, đã seed bằng `seed + 1`.
   Không còn RNG chưa seed.

**Kết luận:** nguồn còn lại là `atomicAdd` trong backward của rasterizer (12 chỗ trong `diff-gaussian-rasterization/cuda_rasterizer/backward.cu`). Thứ tự cộng số thực thay đổi giữa các lần chạy, nên ngay bản đồ khởi tạo (1 050 vòng mapping ở frame 0) đã khác, và frame 1 lệch khoảng 1 mm. Plan cho phép để nguyên, nên không sửa. Trên cả chuỗi, lệch nhỏ này làm danh sách keyframe tách ra (chỉ 30 keyframe trùng). ATE keyframe chênh 8,1 %, mức nhiễu n dùng ở phần B là max(8,1 %, 15 %) = 15 %.

**Code của các run phần A:** sp_r1 chạy ở `cf3148d` (trước khi port E1c); sp_r2 và mt_r1 chạy ở `19dc87f` (đã có hook E1c, tắt khi chạy base; đường chạy base giống hệt). Hai cây chỉ khác các file tool chưa commit, không đụng code chạy.
