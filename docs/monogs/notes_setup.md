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
