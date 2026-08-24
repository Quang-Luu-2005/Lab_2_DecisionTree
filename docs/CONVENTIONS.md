# Quy ước project — Lab 2 Decision Tree

## 1. Phạm vi

Đây là quy ước bắt buộc cho code, notebook và artifact của Lab 2. Nếu một thí nghiệm cần khác quy ước, lý do phải được ghi trong `notes` của file result.

## 2. Tên file và thư mục

| Đối tượng | Quy ước | Ví dụ |
|---|---|---|
| Python module/script | `snake_case` | `train_decision_tree.py` |
| Notebook | `NN_<topic>.ipynb` | `01_data_overview.ipynb` |
| Figure | `<experiment_id>__<name>.<ext>` | `dt_baseline__confusion_matrix.png` |
| Result | `<experiment_id>.json` | `dt_baseline.json` |
| Model | `<experiment_id>.<ext>` | `dt_baseline.joblib` |
| Thư mục | chữ thường, `snake_case` | `data/processed/` |

Không dùng dấu cách, ký tự đặc biệt hoặc tên phụ thuộc máy cá nhân.

## 3. Reproducibility

Mọi code có tính ngẫu nhiên phải nhận `random_state` từ `decision_tree_lab2.config`. Giá trị mặc định của project là `42`. Nếu override, phải ghi giá trị thực tế vào `split.random_state` hoặc metadata tương ứng trong result.

## 4. Train/test split chung

```python
from decision_tree_lab2.split import split_train_test

X_train, X_test, y_train, y_test = split_train_test(X, y)
```

Mặc định:

- `test_size=0.20`
- `random_state=42`
- `stratify=True` cho classification

Quy tắc:

1. Split sau khi xác định `X` và `y`, trước khi fit model hoặc preprocessing có học tham số.
2. Tất cả model so sánh trong cùng experiment dùng cùng indices train/test.
3. Không chọn hyperparameter bằng test set.
4. Nếu stratify không dùng được, đặt `stratify=False` và ghi lý do.

## 5. Artifact

| Loại | Vị trí |
|---|---|
| Dữ liệu gốc | `data/raw/` |
| Dữ liệu processed | `data/processed/` |
| Figures | `figures/` |
| Model | `models/` |
| Kết quả và metadata | `results/` |

Các đường dẫn ghi trong JSON là đường dẫn tương đối từ project root để kết quả có thể đọc trên máy khác.

## 6. Result contract

Bắt buộc có các field:

- `schema_version`
- `experiment_id`
- `dataset`
- `model`
- `split`
- `metrics`
- `artifacts`
- `notes`
- `created_at_utc`

`metrics` dùng tên `snake_case`, giá trị là `int`/`float` hữu hạn. `artifacts.figure_paths` là list đường dẫn tương đối; artifact chưa tạo dùng `null` hoặc list rỗng.
