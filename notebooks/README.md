# Kaggle notebooks

Ba notebook có thể upload trực tiếp lên Kaggle:

- `01_letter_decision_tree_baseline.ipynb`: Letter Recognition, 16 engineered features,
  26 classes.
- `02_digits_decision_tree_baseline.ipynb`: scikit-learn Handwritten Digits, ảnh 8x8,
  64 raw-pixel features, 10 classes.
- `03_covertype_decision_tree_scalability.ipynb`: UCI Covertype, 581.012 mẫu,
  54 features, 7 classes; so sánh cây baseline và cây regularized bằng 5-fold CV.

## Cấu hình Kaggle

Chọn **Accelerator: None (CPU)** để chạy hiệu quả nhất và có thể tắt Internet.
`sklearn.tree.DecisionTreeClassifier` không dùng GPU. Nếu cần ghi nhận môi trường Kaggle có
GPU, có thể bật accelerator; notebook sẽ tự lưu tên GPU, VRAM và driver nhưng vẫn ghi rõ
`model_compute_device: cpu`.

### Letter Recognition

1. Tạo một Kaggle Dataset chứa hai file
   `data/processed/letter_recognition/train.csv` và `test.csv` của project.
2. Upload notebook và chọn **Add Input** để gắn dataset vừa tạo.
3. Chạy **Run All**. Notebook tự tìm cặp CSV theo schema, không phụ thuộc tên thư mục
   do Kaggle sinh ra.

Nếu chỉ cung cấp `letter_recognition.csv` hoặc `letter-recognition.data`, notebook sẽ loại
exact duplicates và tạo stratified split với `test_size=0.20`, `random_state=42`.

### Handwritten Digits

Notebook ưu tiên `digits.csv` nếu file được gắn qua **Add Input**. Nếu không có, notebook
dùng `sklearn.datasets.load_digits()` tích hợp sẵn, nên vẫn chạy được khi tắt Internet.

### Covertype

Notebook ưu tiên `covertype.csv` hoặc `covtype.csv` được gắn bằng **Add Input**. Nếu không
tìm thấy file, notebook gọi `sklearn.datasets.fetch_covtype()`, vì vậy lần chạy đầu phải bật
**Internet On**. Chế độ mặc định dùng đủ 581.012 mẫu và 5-fold cross-validation. Biến môi
trường `COVERTYPE_QUICK_MODE=1` chỉ dành cho kiểm tra nhanh trên 50.000 mẫu, không dùng kết
quả quick mode trong báo cáo.

## Output

Mỗi notebook ghi các artifact vào `/kaggle/working`:

- `figures/`: confusion matrix, feature importance, hình cây và hình lỗi phân loại;
- `models/`: model `.joblib`;
- `results/`: metrics và metadata JSON theo result contract của project.
- `dt_letter_baseline__outputs.zip`, `dt_digits_baseline__outputs.zip` hoặc
  `dt_covertype_scalability__outputs.zip`: gói tải về
  chứa toàn bộ figures, model và result tương ứng.

Cell cuối notebook hiển thị liên kết tải ZIP. Nếu liên kết không hiện, mở bảng **Output**
của Kaggle Notebook và tải file ZIP tại đó.

Result JSON ghi `data_loading_seconds`, `training_seconds`, `prediction_seconds`,
`pipeline_seconds` và khối `hardware`. Chỉ dùng số đo từ lần chạy Kaggle cuối cùng trong report.

Câu mô tả phù hợp cho report:

> Thí nghiệm được thực thi trong môi trường Kaggle Notebook. Accelerator và thông số phần
> cứng được ghi tự động trong result JSON. Mặc dù môi trường có thể được cấp NVIDIA GPU,
> scikit-learn DecisionTreeClassifier thực hiện huấn luyện và suy luận trên CPU.

Đây là baseline Gini chưa pruning hoặc tuning. Không dùng test set để chọn
hyperparameter trong các notebook cải tiến tiếp theo.
