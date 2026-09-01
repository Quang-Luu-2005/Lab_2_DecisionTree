# Group 7 - Lab 2 Decision Tree

Thư mục này được đóng gói theo cấu trúc nộp bài: mã nguồn, dữ liệu cần chạy,
`requirements.txt` và hướng dẫn. Không cần tải thêm dataset cho các notebook CPU.

## Cấu trúc

```text
Group 7 - Code/
├── source code/
│   ├── 01_letter_decision_tree_baseline.ipynb
│   ├── 02_digits_decision_tree_baseline.ipynb
│   ├── 03_covertype_decision_tree_scalability.ipynb
│   ├── 04_hierarchical_shrinkage_three_dataset_benchmark.ipynb
│   ├── 05_three_dataset_four_model_benchmark.ipynb
│   ├── custom_decision_tree/
│   └── run_end_to_end_check.py
├── dataset/
│   ├── DATASET_DESCRIPTION.md
│   ├── letter_recognition/
│   ├── handwritten_digits/
│   └── covertype/
├── requirements.txt
└── README.md
```

## Nội dung từng notebook

1. `01_letter_decision_tree_baseline.ipynb`
   - Làm sạch duplicate và sử dụng split Letter Recognition chuẩn.
   - Huấn luyện Decision Tree baseline; báo cáo Accuracy, Error rate,
     Precision/Recall/F1 macro, confusion matrix, feature importance và cây.

2. `02_digits_decision_tree_baseline.ipynb`
   - Baseline trên bộ Handwritten Digits 8 x 8.
   - Phân tích confusion matrix, mẫu dự đoán sai, feature importance và cấu trúc cây.

3. `03_covertype_decision_tree_scalability.ipynb`
   - Thực nghiệm chính trên toàn bộ 581.012 mẫu Covertype.
   - So sánh hold-out/CV, kiểm tra khả năng mở rộng và phân tích lỗi theo lớp.

4. `04_hierarchical_shrinkage_three_dataset_benchmark.ipynb`
   - Chạy E0-E4 trên cả ba dataset.
   - So sánh baseline, pre-pruning, Cost-Complexity Pruning, Hierarchical
     Shrinkage và CCP kết hợp HS; lambda được chọn bằng CV trên train.

5. `05_three_dataset_four_model_benchmark.ipynb`
   - So sánh Decision Tree, Random Forest, SVM-RBF và KNN trên ba dataset.
   - Notebook này dành cho Kaggle GPU có RAPIDS/cuML; không có CPU fallback để
     tránh báo sai thời gian GPU.

Ngoài notebook, `custom_decision_tree/` chứa Decision Tree tự cài đặt từ đầu và
script đối chiếu với sklearn. `run_end_to_end_check.py` kiểm tra toàn bộ chuỗi
`load -> validate/preprocess -> split -> fit -> predict -> metrics` cho 3/3 dataset.

## Cài đặt và chạy

Mở terminal tại `Group 7 - Code`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook
```

Sau đó mở notebook trong thư mục `source code` và chạy theo thứ tự 01 đến 05.
Các notebook tự dò file trong `dataset/` khi Jupyter được khởi động từ thư mục
`Group 7 - Code`.

Chạy phần custom và kiểm tra end-to-end:

```powershell
python "source code\custom_decision_tree\compare_custom_tree.py"
python "source code\run_end_to_end_check.py"
```

Output mới của hai script được ghi vào `generated_results/`; thư mục này chỉ
được tạo khi chạy và không phải thành phần bắt buộc của submission.

## Protocol chung

- `random_state=42`, test 20%, stratified split.
- Test set chỉ dùng để báo cáo; hyperparameter được chọn trên train/CV.
- Decision Tree của sklearn và Hierarchical Shrinkage chạy trên CPU.
- Notebook so sánh bốn mô hình sử dụng Kaggle GPU cho Random Forest, SVM và KNN.
