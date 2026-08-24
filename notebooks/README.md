# Kaggle notebooks

Các notebook được chia thành ba nhóm:

```text
notebooks/
├── decision_tree/
│   ├── 01_letter_decision_tree_baseline.ipynb
│   ├── 02_digits_decision_tree_baseline.ipynb
│   └── 03_covertype_decision_tree_scalability.ipynb
├── benchmark_models/
│   ├── 04_random_forest_letter_digits.ipynb
│   ├── 05_svm_letter_digits.ipynb
│   └── 06_knn_letter_digits.ipynb
└── model_comparison/
    └── 07_letter_digits_model_comparison.ipynb
```

## Thứ tự chạy

1. Chạy ba notebook trong `decision_tree/`.
2. Chạy Random Forest, SVM và KNN trong `benchmark_models/`.
3. Tải các ZIP kết quả về hoặc gom chúng thành một Kaggle Dataset.
4. Gắn năm ZIP/JSON của Letter, Digits, Random Forest, SVM và KNN bằng **Add Input** rồi
   chạy notebook `07_letter_digits_model_comparison.ipynb`.

Covertype được giữ riêng cho thí nghiệm scalability của Decision Tree, không đưa vào SVM/KNN
vì 581.012 mẫu làm thay đổi đáng kể chi phí và mục tiêu của benchmark.

## Input chung

- Letter Recognition: gắn `data/processed/letter_recognition/train.csv` và `test.csv`.
  Các benchmark tự tìm cặp file theo schema 16 features và target `letter`.
- Digits: dùng `sklearn.datasets.load_digits()` nên không cần Internet.
- Covertype: ưu tiên `covertype.csv`; nếu không có thì bật Internet để notebook gọi
  `fetch_covtype()`.

Tất cả thí nghiệm dùng `test_size=0.20`, `random_state=42`, `stratify=True`. SVM và KNN
fit `StandardScaler` chỉ trên train thông qua pipeline; Decision Tree và Random Forest không
cần scaling.

## Cấu hình Kaggle

Chọn **Accelerator: None (CPU)**. Các estimator scikit-learn trong bộ notebook này không
dùng GPU. Notebook vẫn ghi môi trường, CPU và GPU nhìn thấy được vào result JSON để mô tả
chính xác cấu hình chạy.

## Output

Mỗi notebook model tạo:

- confusion matrix trong `figures/`;
- model `.joblib` trong `models/`;
- metrics JSON và bảng CSV trong `results/`;
- một file `__outputs.zip` tải ở cell cuối.

Notebook so sánh tạo bảng tổng hợp cho bốn model, biểu đồ accuracy/macro-F1, train-test gap,
runtime và `letter_digits_model_comparison__outputs.zip`. Kết quả test chỉ dùng để báo cáo,
không dùng để chọn siêu tham số.

Ba notebook benchmark và notebook so sánh được tái tạo bằng
`python scripts/generate_benchmark_notebooks.py`; notebook Decision Tree được quản lý trực tiếp.
