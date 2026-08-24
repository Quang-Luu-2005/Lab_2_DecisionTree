# Kaggle notebooks

Hai notebook baseline có thể upload trực tiếp lên Kaggle:

- `01_letter_decision_tree_baseline.ipynb`: Letter Recognition, 16 engineered features,
  26 classes.
- `02_digits_decision_tree_baseline.ipynb`: scikit-learn Handwritten Digits, ảnh 8x8,
  64 raw-pixel features, 10 classes.

## Cấu hình Kaggle

Chọn **Accelerator: None (CPU)** và có thể tắt Internet. `sklearn.tree.DecisionTreeClassifier`
không dùng GPU, trong khi cả hai dataset đủ nhỏ để CPU chạy nhanh.

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

## Output

Mỗi notebook ghi các artifact vào `/kaggle/working`:

- `figures/`: confusion matrix, feature importance, hình cây và hình lỗi phân loại;
- `models/`: model `.joblib`;
- `results/`: metrics và metadata JSON theo result contract của project.

Đây là baseline Gini chưa pruning hoặc tuning. Không dùng test set để chọn
hyperparameter trong các notebook cải tiến tiếp theo.
