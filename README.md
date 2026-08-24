# Lab 2 — Decision Tree

Scaffold dùng chung cho các thí nghiệm Decision Tree. Mục tiêu là để mọi thành viên chạy cùng một quy ước về môi trường, chia dữ liệu, thư mục đầu ra và format kết quả.

## Bắt đầu nhanh

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Kiểm tra import:

```bash
python -c "from decision_tree_lab2.config import RANDOM_STATE; print(RANDOM_STATE)"
```

## Cấu trúc project

```text
.
├── configs/                 # Cấu hình thí nghiệm dạng YAML/JSON nếu cần
├── data/
│   ├── raw/                 # Dữ liệu gốc, không chỉnh sửa
│   └── processed/           # Dữ liệu đã tiền xử lý
├── figures/                 # Hình xuất từ notebook/script
├── models/                  # Model đã train, nếu cần lưu
├── notebooks/               # Notebook khám phá/phân tích
├── results/                 # Kết quả dạng JSON và bảng tổng hợp
├── src/decision_tree_lab2/  # Python package dùng chung
└── tests/                   # Kiểm thử các contract dùng chung
```

Các thư mục đầu ra đã được tạo sẵn bằng `.gitkeep`; file lớn hoặc dữ liệu riêng không nên commit vào repo.

## Quy ước dùng chung

- Python: 3.11.x.
- Tất cả tên file/thư mục dùng `snake_case`, viết thường; không dùng dấu cách hoặc tiếng Việt có dấu.
- Tên notebook có dạng `NN_<topic>.ipynb`, ví dụ `01_data_overview.ipynb`.
- Tên script có dạng `<verb>_<topic>.py`, ví dụ `train_decision_tree.py`.
- Tên figure có dạng `<experiment_id>__<figure_name>.png` và/hoặc `.pdf`.
- Tên result có dạng `results/<experiment_id>.json`.
- Không ghi đè kết quả của thí nghiệm khác; dùng `experiment_id` mới cho mỗi cấu hình/model.
- Import code từ `decision_tree_lab2`, không copy lại logic split hoặc đường dẫn trong notebook.

Chi tiết nằm trong [docs/CONVENTIONS.md](docs/CONVENTIONS.md).

## Reproducibility và train/test split

Giá trị mặc định dùng chung:

```text
RANDOM_STATE = 42
TEST_SIZE = 0.20
STRATIFY_SPLIT = True
```

Dùng `split_train_test` trong `decision_tree_lab2.split`. Chỉ chia train/test một lần ở đầu pipeline; mọi model/experiment phải dùng đúng các tập đã chia đó. Không fit preprocessing trên test set.

Nếu bài toán không phù hợp với stratification, phải ghi rõ lý do trong `notes` của result và gọi hàm với `stratify=False`.

## Nơi lưu figures/results

- Figures: `figures/`
- Metrics và metadata: `results/`
- Model artifact: `models/`
- Dữ liệu gốc: `data/raw/`
- Dữ liệu xử lý: `data/processed/`

Có thể gọi `ensure_project_dirs()` trước khi ghi artifact để tạo các thư mục cần thiết.

## Format kết quả

Mỗi experiment ghi một file JSON theo schema version `1.0`:

```json
{
  "schema_version": "1.0",
  "experiment_id": "dt_baseline",
  "dataset": "dataset_name",
  "model": "DecisionTreeClassifier",
  "split": {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": true
  },
  "metrics": {
    "accuracy": 0.0,
    "f1_macro": 0.0
  },
  "artifacts": {
    "figure_paths": [],
    "model_path": null
  },
  "notes": null,
  "created_at_utc": "2026-01-01T00:00:00+00:00"
}
```

Các metric phải là số JSON hợp lệ; không dùng `NaN`/`Infinity`. Tên metric dùng `snake_case`. Helper `build_result` và `save_result` trong `decision_tree_lab2.results` giúp tạo/ghi đúng format.

## Kiểm thử

```bash
pytest -q
```
