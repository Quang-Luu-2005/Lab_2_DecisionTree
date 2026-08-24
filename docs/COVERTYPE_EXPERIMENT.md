# Covertype scalability experiment

## Mục đích

Covertype bổ sung một thí nghiệm quy mô lớn cho bài Decision Tree: 581.012 mẫu, 54 đặc
trưng số và 7 lớp. Bộ dữ liệu này không thay thế Letter Recognition là bộ chính; nó dùng để
kiểm tra khả năng mở rộng, overfitting và đánh đổi giữa hiệu năng với độ phức tạp của cây.

## Protocol

- Nguồn: UCI Covertype qua `sklearn.datasets.fetch_covtype`.
- Hold-out: `test_size=0.20`, `random_state=42`, phân tầng theo nhãn.
- Hai mô hình: baseline Gini không cắt tỉa và regularized Gini với `max_depth=20`,
  `min_samples_leaf=5`.
- Chọn mô hình bằng macro-F1 trung bình của 5-fold cross-validation trên tập train; tập test
  chỉ dùng cho đánh giá cuối.
- Báo cáo accuracy, error rate, macro precision/recall/F1, weighted-F1, train-test gap,
  thời gian fit/predict, độ sâu và số lá.
- Với cross-validation, báo cáo trung bình ± độ lệch chuẩn của accuracy, macro-F1 và thời gian.

## Tiêu chí kế thừa từ report năm trước

Các tiêu chí có thể áp dụng trực tiếp gồm mô tả môi trường chạy, train/test performance,
5-fold cross-validation, trung bình ± độ lệch chuẩn, runtime, độ ổn định và biểu đồ đánh đổi
hiệu năng–độ phức tạp. Các đại lượng Cost, Loss, Invalidity/Objective của phương pháp CET
không được dùng vì không có ý nghĩa tương ứng trong bài phân loại Decision Tree chuẩn.

## Kết quả kiểm chứng cục bộ

Lần Run All ngày 2026-08-25 trên toàn bộ dữ liệu đã hoàn tất thành công. Baseline đạt
hold-out accuracy 0,9389 và macro-F1 0,9034; cây sâu 41 với 23.956 lá. Trong 5-fold CV,
baseline đạt accuracy 0,9330 ± 0,0011 và macro-F1 0,8923 ± 0,0011. Regularized đạt
accuracy 0,8966 ± 0,0019 và macro-F1 0,8388 ± 0,0028, đổi lại giảm độ sâu xuống 20 và
số lá hold-out xuống 9.679. Thời gian phụ thuộc tải máy nên được lưu trong result JSON thay
vì ghi cố định trong tài liệu. Đây chỉ là số kiểm chứng local; báo cáo cuối nên lấy số từ
result JSON của lần chạy Kaggle chính thức.

Notebook tạo bốn figure (phân bố lớp, so sánh mô hình, confusion matrix và feature
importance), hai bảng CSV, result JSON, hai model và ZIP tải về. `DecisionTreeClassifier`
của scikit-learn chạy trên CPU ngay cả khi Kaggle cấp GPU.
