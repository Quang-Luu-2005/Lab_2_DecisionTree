# Letter Recognition Dataset

## Nguồn và mục tiêu

Dataset chính của project là **Letter Recognition** từ UCI Machine Learning Repository.
Mỗi quan sát biểu diễn một ảnh ký tự in hoa đã được chuyển thành 16 đặc trưng thống kê
và cạnh. Bài toán là phân loại ký tự mục tiêu `letter` vào 26 lớp từ A đến Z.

- Source: https://archive.ics.uci.edu/dataset/59/letter%2Brecognition
- Raw file: `data/raw/letter_recognition/letter_recognition.csv`
- Số quan sát raw: **20,000**
- Số đặc trưng: **16**
- Số lớp: **26**
- Số mẫu mỗi lớp: **734–813**

## Các đặc trưng

| Feature | Ý nghĩa | Miền giá trị |
|---|---|---|
| `x_box` | Horizontal position of the character bounding box | Integer, 0–15 |
| `y_box` | Vertical position of the character bounding box | Integer, 0–15 |
| `width` | Width of the character bounding box | Integer, 0–15 |
| `high` | Height of the character bounding box | Integer, 0–15 |
| `onpix` | Number of foreground pixels | Integer, 0–15 |
| `x_bar` | Mean horizontal position of foreground pixels | Integer, 0–15 |
| `y_bar` | Mean vertical position of foreground pixels | Integer, 0–15 |
| `x2bar` | Horizontal variance | Integer, 0–15 |
| `y2bar` | Vertical variance | Integer, 0–15 |
| `xybar` | Horizontal-vertical correlation | Integer, 0–15 |
| `x2ybr` | Mean of x-squared times y | Integer, 0–15 |
| `xy2br` | Mean of x times y-squared | Integer, 0–15 |
| `x_ege` | Mean left-to-right edge count | Integer, 0–15 |
| `xegvy` | Correlation of horizontal edge count with y | Integer, 0–15 |
| `y_ege` | Mean bottom-to-top edge count | Integer, 0–15 |
| `yegvx` | Correlation of vertical edge count with x | Integer, 0–15 |

## Kiểm tra chất lượng

| Kiểm tra | Kết quả |
|---|---:|
| Missing cells | 0 |
| Exact duplicate rows | 1,332 |
| Rows in duplicate-feature groups | 2,177 |
| Feature groups with conflicting labels | 0 |
| Values outside 0–15 | 0 |
| Non-integer feature values | 0 |
| IQR-flagged cells | 10,540 |

Các điểm bị IQR đánh dấu không bị xóa vì đặc trưng đã được UCI scale vào miền hữu hạn
0–15; chúng là giá trị hợp lệ theo data contract. Không cần imputation. Các dòng trùng
hoàn toàn được loại trước khi split để cùng một vector đặc trưng không xuất hiện ở cả
train và test.

## Preprocessing và split dùng chung

- Sau deduplication: **18,668** mẫu
  (loại 1,332 mẫu,
  6.66%).
- Không scaling cho Decision Tree.
- `test_size=0.2`
- `random_state=42`
- `stratify=true`
- Train: **14,934** mẫu; test: **3,734** mẫu.
- Cả train và test đều có **26** lớp.
- Số vector đặc trưng xuất hiện ở cả hai split: **0**.

Mọi Decision Tree và benchmark phải dùng trực tiếp `train.csv` và `test.csv` bên dưới
để bảo đảm so sánh công bằng. Scaling cho SVM/KNN phải được fit chỉ trên train set bằng
pipeline riêng; không sửa hai file split chung.

## Output bàn giao

- `data/processed/letter_recognition/letter_recognition_clean.csv`
- `data/processed/letter_recognition/train.csv`
- `data/processed/letter_recognition/test.csv`
- `data/processed/letter_recognition/class_distribution.csv`
- `data/processed/letter_recognition/feature_summary.csv`
- `data/processed/letter_recognition/preparation_report.json`

Tái tạo toàn bộ output bằng:

```bash
python scripts/prepare_letter_recognition.py
```
