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
├── scripts/                 # Script tải dữ liệu/chạy pipeline
├── figures/                 # Hình xuất từ notebook/script
├── models/                  # Model đã train, nếu cần lưu
├── notebooks/               # Notebook khám phá/phân tích
├── results/                 # Kết quả dạng JSON và bảng tổng hợp
├── src/decision_tree_lab2/  # Python package dùng chung
└── tests/                   # Kiểm thử các contract dùng chung
```

Các thư mục đầu ra đã được tạo sẵn bằng `.gitkeep`; file lớn hoặc dữ liệu riêng không nên commit vào repo.

Dataset sources and the reproducible download command are documented in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Chuẩn bị Letter Recognition (thẻ 02–03)

Sau khi tải dữ liệu, chạy pipeline audit, làm sạch và tạo split dùng chung:

```bash
python scripts/prepare_letter_recognition.py
```

Pipeline giữ nguyên `data/raw/`, loại các dòng trùng hoàn toàn, rồi tạo stratified
train/test split với `test_size=0.20` và `random_state=42`. Các model phải dùng
`data/processed/letter_recognition/train.csv` và `test.csv` để bảo đảm cùng split.

Thống kê, quyết định preprocessing và mô tả dataset dùng cho report nằm trong
[docs/LETTER_RECOGNITION_DATASET.md](docs/LETTER_RECOGNITION_DATASET.md).

Tạo figures, bảng EDA và nhận xét dùng cho report/slide:

```bash
python scripts/run_letter_eda.py
```

Pipeline tạo class distribution, phân bố 16 features, correlation heatmap và
class–feature profiles. Phần nhận xét được tổng hợp tại
[docs/LETTER_RECOGNITION_EDA.md](docs/LETTER_RECOGNITION_EDA.md).

## Notebook Decision Tree baseline cho Kaggle

- [Letter Recognition baseline](notebooks/01_letter_decision_tree_baseline.ipynb)
- [Handwritten Digits baseline](notebooks/02_digits_decision_tree_baseline.ipynb)
- [Covertype scalability](notebooks/03_covertype_decision_tree_scalability.ipynb)

Ba notebook dùng `DecisionTreeClassifier`, cùng protocol `test_size=0.20`,
`random_state=42`, `stratify=True` và tự lưu figures, model, result JSON. Trên Kaggle
CPU là accelerator khuyến nghị. Notebook tự ghi timing và thông tin phần cứng; nếu Kaggle
được bật GPU thì tên GPU/VRAM/driver cũng được lưu, nhưng Decision Tree vẫn chạy trên CPU.
Riêng Covertype dùng thêm 5-fold cross-validation để đánh giá độ ổn định, thời gian và
đánh đổi giữa hiệu năng với độ phức tạp của cây.
Xem hướng dẫn upload và Add Input trong [notebooks/README.md](notebooks/README.md).

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
