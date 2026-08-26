# Kaggle notebooks

Các notebook được chia thành ba nhóm:

```text
notebooks/
├── decision_tree/
│   ├── 01_letter_decision_tree_baseline.ipynb
│   ├── 02_digits_decision_tree_baseline.ipynb
│   ├── 03_covertype_decision_tree_scalability.ipynb
│   └── 04_letter_decision_tree_improvements.ipynb
├── benchmark_models/
│   └── 07_three_dataset_three_model_benchmark.ipynb
└── model_comparison/
    └── 08_three_dataset_model_comparison.ipynb
```

## Thứ tự chạy

1. Chạy bốn notebook trong `decision_tree/`.
2. Chạy `07_three_dataset_three_model_benchmark.ipynb` để benchmark đủ ba model Random Forest,
   SVM và KNN trên cả ba dataset. Covertype dùng toàn bộ 581.012 mẫu. Notebook lưu checkpoint
   sau mỗi model; KNN/SVM có thể chạy rất lâu.
3. Tải các ZIP kết quả về hoặc gom chúng thành một Kaggle Dataset.
4. Gắn bốn ZIP/JSON đầu vào bằng **Add Input** rồi chạy
   `08_three_dataset_model_comparison.ipynb`.

## Input chung

- Letter Recognition: gắn `data/processed/letter_recognition/train.csv` và `test.csv`.
  Các benchmark tự tìm cặp file theo schema 16 features và target `letter`.
- Digits: dùng `sklearn.datasets.load_digits()` nên không cần Internet.
- Covertype: ưu tiên `covertype.csv`; nếu không có thì bật Internet để notebook gọi
  `fetch_covtype()`. Notebook 07 dùng toàn bộ dữ liệu, không lấy mẫu rút gọn.

Tất cả thí nghiệm dùng `test_size=0.20`, `random_state=42`, `stratify=True`. SVM và KNN
fit `StandardScaler` chỉ trên train thông qua pipeline; Decision Tree và Random Forest không
cần scaling.

## Cấu hình Kaggle

Chọn **Accelerator: None (CPU)**. Các estimator scikit-learn trong bộ notebook này không
dùng GPU. Notebook vẫn ghi môi trường, CPU và GPU nhìn thấy được vào result JSON để mô tả
chính xác cấu hình chạy. Nếu bật GPU Kaggle, GPU có thể được phát hiện nhưng không tham gia
huấn luyện bốn model này.

## Output

Các notebook Decision Tree tạo:

- confusion matrix trong `figures/`;
- model `.joblib` trong `models/`;
- metrics JSON và bảng CSV trong `results/`;
- một file `__outputs.zip` tải ở cell cuối.

Notebook benchmark không lưu `.joblib` vì Random Forest/KNN trên toàn bộ dữ liệu có thể làm
output rất lớn; ZIP vẫn có đủ confusion matrices, metrics, timing và metadata phần cứng.

Notebook `04_letter_decision_tree_improvements.ipynb` thử 36 candidate qua ba nhánh
`max_depth`, `min_samples_split`/`min_samples_leaf` và `ccp_alpha`; cấu hình cuối được chọn
bằng 5-fold CV trên train. ZIP của notebook chứa curves, heatmap, pruning path, rules và cây
được chọn.

Notebook so sánh tạo bảng tổng hợp bốn model trên ba dataset, biểu đồ accuracy/macro-F1,
train-test gap, runtime và `three_dataset_model_comparison__outputs.zip`. Với KNN cần đọc cả
`training_seconds` và `prediction_seconds`: KNN là lazy learner nên fit nhanh nhưng suy luận
có thể rất chậm. Kết quả test chỉ dùng để báo cáo, không dùng để chọn siêu tham số.

Notebook benchmark và notebook so sánh được tái tạo bằng
`python scripts/generate_benchmark_notebooks.py`; notebook Decision Tree được quản lý trực tiếp.
