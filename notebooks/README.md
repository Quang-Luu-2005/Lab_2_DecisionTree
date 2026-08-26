# Kaggle notebooks

Các notebook được chia thành ba nhóm:

```text
notebooks/
├── decision_tree/
│   ├── 01_letter_decision_tree_baseline.ipynb
│   ├── 02_digits_decision_tree_baseline.ipynb
│   ├── 03_covertype_decision_tree_scalability.ipynb
│   └── 04_hierarchical_shrinkage_three_dataset_benchmark.ipynb
├── benchmark_models/
│   └── 05_three_dataset_three_model_benchmark.ipynb
└── model_comparison/
    └── 06_three_dataset_model_comparison.ipynb
```

## Thứ tự chạy

1. Chạy bốn notebook trong `decision_tree/`. Đây là các thí nghiệm Decision Tree exact và
   Hierarchical Shrinkage; chúng chạy CPU để giữ đúng model cần nghiên cứu.
2. Chọn **GPU Accelerator** trên Kaggle rồi chạy `05_three_dataset_three_model_benchmark.ipynb`
   để benchmark Random Forest, SVM và KNN bằng RAPIDS/cuML trên cả ba dataset. Covertype dùng
   toàn bộ 581.012 mẫu; notebook kiểm tra CUDA/cuML và dừng nếu không có GPU, không âm thầm
   chuyển sang CPU.
3. Tải các ZIP kết quả về hoặc gom chúng thành một Kaggle Dataset.
4. Gắn bốn ZIP/JSON đầu vào bằng **Add Input** rồi chạy
   `06_three_dataset_model_comparison.ipynb`.

## Input chung

- Letter Recognition: gắn `data/processed/letter_recognition/train.csv` và `test.csv`.
  Các benchmark tự tìm cặp file theo schema 16 features và target `letter`.
- Digits: dùng `sklearn.datasets.load_digits()` nên không cần Internet.
- Covertype: ưu tiên `covertype.csv`; nếu không có thì bật Internet để notebook gọi
  `fetch_covtype()`. Notebook 05 dùng toàn bộ dữ liệu, không lấy mẫu rút gọn.

Tất cả thí nghiệm dùng `test_size=0.20`, `random_state=42`, `stratify=True`. SVM và KNN
fit `StandardScaler` chỉ trên train thông qua pipeline; Decision Tree và Random Forest không
cần scaling.

## Cấu hình Kaggle

Notebook `01`–`04` dùng scikit-learn/imodels cho Decision Tree exact và Hierarchical
Shrinkage, nên chọn **Accelerator: None (CPU)**. Notebook `05` là bản benchmark GPU-only:
chọn **GPU Accelerator** (T4 hoặc P100 tuỳ quota), dùng trực tiếp RAPIDS/cuML và CuPy; không
thêm CPU fallback. Giữ Internet off nếu image Kaggle đã có RAPIDS. Kết quả của notebook `05`
ghi tên GPU, CUDA runtime, VRAM khả dụng, thời gian truyền dữ liệu/fit/predict và metric vào
JSON/ZIP. Notebook `06` chỉ đọc các output, không huấn luyện model.

## Output

Các notebook Decision Tree tạo:

- confusion matrix trong `figures/`;
- model `.joblib` trong `models/`;
- metrics JSON và bảng CSV trong `results/`;
- một file `__outputs.zip` tải ở cell cuối.

Notebook benchmark GPU không lưu `.joblib` vì Random Forest/KNN trên toàn bộ dữ liệu có thể
làm output rất lớn; ZIP `gpu_three_dataset_three_model_benchmark__outputs.zip` vẫn có đủ
confusion matrices, metrics, timing và metadata phần cứng. Notebook `06` đọc file
`gpu_three_dataset_three_model_benchmark.json` từ ZIP này.

Notebook `04_hierarchical_shrinkage_three_dataset_benchmark.ipynb` dùng `imodels==3.0.0`,
nhận `HS_DATASET=letter_recognition|handwritten_digits|covertype|all`, và chạy cùng protocol
trên cả ba dataset. HS chọn `lambda` bằng stratified CV trên train; nó thay prediction values
nhưng kiểm tra depth/leaves của underlying tree để chứng minh topology không đổi. Output có
summary E0–E4, CV selection, lambda curves, confusion matrices, complexity plot và ZIP.

Notebook so sánh tạo bảng tổng hợp bốn model trên ba dataset, biểu đồ accuracy/macro-F1,
train-test gap, runtime và `three_dataset_model_comparison__outputs.zip`. Với KNN cần đọc cả
`training_seconds` và `prediction_seconds`: KNN là lazy learner nên fit nhanh nhưng suy luận
có thể rất chậm. Kết quả test chỉ dùng để báo cáo, không dùng để chọn siêu tham số.

Notebook GPU benchmark được tái tạo bằng
`python scripts/generate_gpu_benchmark_notebook.py`; lệnh
`python scripts/generate_benchmark_notebooks.py` cũng gọi generator này và tái tạo notebook
so sánh. Notebook HS được tái tạo bằng `python scripts/generate_hs_benchmark_notebook.py`.
