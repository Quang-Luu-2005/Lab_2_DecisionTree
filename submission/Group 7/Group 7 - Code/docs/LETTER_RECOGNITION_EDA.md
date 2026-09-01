# Letter Recognition — Exploratory Data Analysis

## Phạm vi

EDA mô tả sử dụng toàn bộ `letter_recognition_clean.csv`. Phân tích quan hệ giữa
target và features sử dụng riêng `train.csv` để không nhìn trước test set. EDA không
thực hiện feature selection hoặc thay đổi split chung.

## Hình dùng cho report và slide

### Phân bố lớp trước và sau deduplication

![Class distribution](../figures/eda_letter__class_distribution.png)

Raw dataset khá cân bằng với tỷ lệ lớp lớn nhất/nhỏ nhất là
**1.108**. Sau khi loại exact duplicates, tỷ lệ
này là **1.492**; các lớp bị loại nhiều mẫu
trùng nhất là I (231), X (109), N (95), Z (94), L (88). Vì vậy mọi metric multiclass phải báo thêm macro-F1,
không chỉ accuracy.

### Phân bố 16 features

![Feature distributions](../figures/eda_letter__feature_distributions.png)

Các feature có độ lệch chuẩn cao nhất là `y_box`, `x2bar`, `x2ybr`, `y_ege`, `xybar`. Nhiều phân bố rời rạc và lệch,
phù hợp với mô tả đây là các thống kê ảnh đã được lượng tử hóa vào miền 0–15. Các điểm
bị IQR đánh dấu vẫn nằm trong miền hợp lệ nên không bị xóa.

### Tương quan giữa features

![Correlation heatmap](../figures/eda_letter__correlation_heatmap.png)

| Feature 1 | Feature 2 | Pearson r |
|---|---|---:|
| `x_box` | `width` | 0.838 |
| `y_box` | `high` | 0.805 |
| `x_box` | `y_box` | 0.754 |
| `width` | `onpix` | 0.747 |
| `y_box` | `width` | 0.660 |

Mean absolute pairwise correlation là
**0.180**. Tương quan không được dùng để
tự động loại feature ở giai đoạn này; Decision Tree có thể chọn split phi tuyến và
feature importance sẽ được đánh giá ở thẻ baseline.

### Quan hệ class–feature trên train set

![Class-feature profiles](../figures/eda_letter__class_feature_profiles.png)

Các feature có mức biến thiên giữa class lớn nhất trên train set là `x2ybr`, `x_ege`, `y_bar`, `xegvy`, `xy2br`.
Heatmap cho thấy mỗi chữ cái có profile kết hợp nhiều feature, hỗ trợ dùng mô hình cây
thay vì quy tắc một biến đơn giản. Đây chỉ là nhận xét mô tả, không dùng test set và
không quyết định trước feature selection.

## Output dạng bảng

- `results/eda_letter__class_distribution.csv`
- `results/eda_letter__feature_statistics.csv`
- `results/eda_letter__correlation_matrix.csv`
- `results/eda_letter__class_feature_means_train.csv`
- `results/eda_letter__class_feature_profiles_train.csv`
- `results/eda_letter__insights.json`
- `results/eda_letter.json`

Tái tạo toàn bộ output:

```bash
python scripts/run_letter_eda.py
```
