# Kaggle notebooks

Các notebook được chia thành hai nhóm:

```text
notebooks/
├── decision_tree/
│   ├── 01_letter_decision_tree_baseline.ipynb
│   ├── 02_digits_decision_tree_baseline.ipynb
│   ├── 03_covertype_decision_tree_scalability.ipynb
│   └── 04_hierarchical_shrinkage_three_dataset_benchmark.ipynb
└── benchmark_models/
    └── 05_three_dataset_three_model_benchmark.ipynb
```

## Thứ tự chạy

1. Chạy bốn notebook trong `decision_tree/` nếu cần các kết quả chi tiết của Decision Tree
   exact và Hierarchical Shrinkage.
2. Chọn **GPU Accelerator** trên Kaggle rồi chạy `05_three_dataset_three_model_benchmark.ipynb`.
   Notebook tự train và so sánh cả bốn model trên cả ba dataset trong cùng một lần chạy, sau đó
   tạo duy nhất một ZIP kết quả để tải về.

## Input chung

- Letter Recognition: gắn `data/processed/letter_recognition/train.csv` và `test.csv`.
  Các benchmark tự tìm cặp file theo schema 16 features và target `letter`.
- Digits: dùng `sklearn.datasets.load_digits()` nên không cần Internet.
- Covertype: ưu tiên `covertype.csv`; nếu không có thì bật Internet để notebook gọi
  `fetch_covtype()`. Notebook 05 dùng toàn bộ dữ liệu, không lấy mẫu rút gọn.

Tất cả thí nghiệm dùng `test_size=0.20`, `random_state=42`, `stratify=True`. SVM và KNN
fit `StandardScaler` chỉ trên train bằng cuML; Decision Tree và Random Forest không cần scaling.

## Cấu hình Kaggle

Notebook `01`–`04` dùng scikit-learn/imodels cho Decision Tree exact và Hierarchical
Shrinkage, nên chọn **Accelerator: None (CPU)**. Notebook `05` là bản benchmark tổng hợp:
chọn **GPU Accelerator** (T4 hoặc P100 tuỳ quota), Decision Tree chạy CPU exact còn Random
Forest/SVM/KNN dùng trực tiếp RAPIDS/cuML và CuPy trên GPU; không có CPU fallback cho ba model
GPU. Giữ Internet off nếu image Kaggle đã có RAPIDS. Output ghi tên GPU, CUDA runtime, VRAM
khả dụng, timing và metric của cả bốn model vào cùng JSON/CSV/ZIP.

## Output

Các notebook Decision Tree tạo:

- confusion matrix trong `figures/`;
- model `.joblib` trong `models/`;
- metrics JSON và bảng CSV trong `results/`;
- một file `__outputs.zip` tải ở cell cuối.

Notebook benchmark tổng hợp không lưu `.joblib` vì Random Forest/KNN trên toàn bộ dữ liệu có
thể làm output rất lớn; ZIP `gpu_three_dataset_four_model_comparison__outputs.zip` có đủ
confusion matrices, metrics, timing, comparison plots và metadata phần cứng.

Notebook `04_hierarchical_shrinkage_three_dataset_benchmark.ipynb` dùng `imodels==3.0.0`,
nhận `HS_DATASET=letter_recognition|handwritten_digits|covertype|all`, và chạy cùng protocol
trên cả ba dataset. HS chọn `lambda` bằng stratified CV trên train; nó thay prediction values
nhưng kiểm tra depth/leaves của underlying tree để chứng minh topology không đổi. Output có
summary E0–E4, CV selection, lambda curves, confusion matrices, complexity plot và ZIP.

Notebook `05` tạo bảng tổng hợp bốn model trên ba dataset, biểu đồ accuracy/macro-F1,
train-test gap, runtime và ZIP kết quả. Với KNN cần đọc cả `training_seconds` và
`prediction_seconds`: KNN là lazy learner nên fit nhanh nhưng suy luận có thể rất chậm. Kết
quả test chỉ dùng để báo cáo, không dùng để chọn siêu tham số.

Notebook benchmark được tái tạo bằng
`python scripts/generate_gpu_benchmark_notebook.py` hoặc lệnh tương thích
`python scripts/generate_benchmark_notebooks.py`. Notebook HS được tái tạo bằng
`python scripts/generate_hs_benchmark_notebook.py`.
