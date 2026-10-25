# MonoGS-E1c — cài đặt và Phần A (mốc P1)

Sinh từ dữ liệu (`tools/monogs/make_report.py --stage A`).

## Môi trường

| Mục | Giá trị |
|---|---|
| GPU, driver, VRAM | NVIDIA GeForce RTX 5060 Ti, 591.86, 16311 MiB |
| CPU | AMD Ryzen 7 7700 8-Core Processor |
| CUDA toolkit | Cuda compilation tools, release 12.8, V12.8.93 |
| PyTorch / CUDA / cuDNN | 2.7.1+cu128 / 12.8 / 90701 |
| Open3D | 0.19.0 |
| NumPy | 1.23.5 |
| evo / matplotlib | 1.11.0 / 3.10.9 |
| Python | Python 3.10.21 |
| WSL kernel | 6.18.33.2-microsoft-standard-WSL2 |
| MonoGS | 6c9254c (muskie82/MonoGS, vendored unmodified in commit 749c5bd) |
| Submodule diff-gaussian-rasterization | 43e21bf |
| Submodule simple-knn | 44f7642 |
| TF32 mặc định (matmul / cuDNN) | False / True |

| Chuỗi | Đường dẫn | Dòng trong rgb.txt |
|---|---|---|
| rgbd_dataset_freiburg1_desk | `/home/wslphu/2DGS-SLAM/datasets/tum/rgbd_dataset_freiburg1_desk` | 613 |
| rgbd_dataset_freiburg3_structure_notexture_far | `/home/wslphu/2DGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_structure_notexture_far` | 814 |
| rgbd_dataset_freiburg3_structure_notexture_near | `/home/wslphu/2DGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_structure_notexture_near` | 1082 |

**Sửa build, tương thích và tái lập** (`git diff 749c5bd -- monogs/` liệt kê đúng các chỗ này):

| Sửa | Loại | Lý do |
|---|---|---|
| `submodules/simple-knn/simple_knn.cu`: thêm `#include <cfloat>` | Build | CUDA 12.8 báo `FLT_MAX` chưa khai báo (lần build đầu thất bại); cùng bản vá đã có trong 2DGS-SLAM |
| `utils/dataset.py`: `np.unicode_` → `np.str_` | Tương thích | NumPy ≥ 2 bỏ `np.unicode_` (env hiện có NumPy 1.23, sửa phòng xa như plan yêu cầu). Không đổi hành vi |
| `slam.py`: `from gui import slam_gui` chỉ khi `use_gui` | Tương thích | Chạy headless (slam_gui cần glfw / OpenGL). Không đổi hành vi |
| `utils/eval_utils.py`: bọc phần vẽ hình của `evaluate_evo` trong `try` | Tương thích | evo 1.11 `traj_colormap` lỗi với matplotlib 3.10 ("Unable to determine Axes to steal space for Colorbar"), làm crash `eval_ate` ở cuối run (smoke đầu tiên). ATE được tính trước bước vẽ, nên không đổi số |
| `utils/slam_backend.py`: đọc `Training.single_thread` (chỉ khi thiếu mới đọc `Dataset.single_thread`); log và so cờ ở cả hai tiến trình | Tái lập (§3.1) | Bản gốc: frontend đọc `Training`, backend đọc `Dataset` |
| `slam.py` / `utils/repro.py`: `--seed` (mặc định 0) seed `random`, `numpy`, `torch` (+ CUDA), `open3d.utility.random.seed`; `cudnn.deterministic = True`, `benchmark = False`; backend seed lại với `seed + 1` | Tái lập (§3.2) | Bản gốc không seed; Open3D `random_down_sample` và `torch.randperm` của backend ngẫu nhiên |
| `--save-dir`, `--range`, `--method`; ghi `traj_all.txt`, `traj_kf.txt`, `traj_gt_all.txt`, `kf_indices.txt`, `timing.json`, `ortho.csv` | CLI, chỉ ghi (§3.3) | |
| Frontend: đo ms tracking (có `torch.cuda.synchronize()` sau tracking), log cờ | Chỉ ghi | |
| Chạy với `WANDB_MODE=disabled` | Không gửi dữ liệu ra ngoài | |

Env `monogs` = clone của `2dgs-slam` (torch không đổi), cài thêm hai extension của MonoGS và các gói ở mục 2.2 của plan. Lệnh `pip install wandb` nâng `wandb` lên 0.31.0 và `protobuf` lên 7.36.2 trong env `monogs` (env `2dgs-slam` không bị đụng). Cài thêm `pypdf` để đọc paper (máy không có `pdftotext`).

### Phần A — MonoGS trên fr1/desk

Số paper: **1,50 cm** (MonoGS, arXiv 2312.06741, Table 1, RGB-D, không loop closure; ATE RMSE trên keyframe, trung bình 3 run, chế độ song song, RTX 4090). Ở đây mỗi run là 1 lần chạy, RTX 5060 Ti.

| Run | Chế độ | ATE keyframe (cm) | ATE mọi frame (cm) | Số keyframe | Số frame | ms/frame tracking | s/frame tổng | r so với paper | Đánh giá |
|---|---|---|---|---|---|---|---|---|---|
| sp_r1 | đơn luồng, seed 0 | 1,43 | 1,43 | 109 | 592 | 525 | 2,52 | 0,95 | khớp |
| sp_r2 | đơn luồng, seed 0, chạy lại | 1,54 | 1,47 | 108 | 592 | 524 | 2,47 | 1,03 | khớp |
| mt_r1 | song song, seed 0 | 1,46 | 1,45 | 136 | 592 | 917 | 0,97 | 0,97 | khớp |

**sp_r1 so với sp_r2 (cùng code, cùng seed 0, chạy lại):**

| Chỉ số | Giá trị |
|---|---|
| Danh sách keyframe trùng nhau | không (109 / 108 keyframe, chung 30) |
| Lệch tâm camera lớn nhất ở keyframe chung, không căn (cm) | 0,729 |
| Lệch tâm camera lớn nhất trên mọi frame, không căn (cm) | 1,674 |
| Frame đầu tiên lệch > 1 mm | 1 |
| ATE keyframe r1 / r2 (cm) | 1,43 / 1,54 |
| ATE mọi frame r1 / r2 (cm) | 1,43 / 1,47 |
| Mức nhiễu tương đối n = \|ATE r1 − ATE r2\| / ATE r1 (keyframe) | 8,1 % |


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

