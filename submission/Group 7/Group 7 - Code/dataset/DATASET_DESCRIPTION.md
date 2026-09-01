# Dataset dùng trong bài

| Dataset | File | Số mẫu | Feature | Lớp | Target |
|---|---|---:|---:|---:|---|
| Letter Recognition | `letter_recognition/letter_recognition.csv` | 20.000 raw; 18.668 sau bỏ duplicate | 16 | 26 | `letter` |
| Handwritten Digits | `handwritten_digits/digits.csv` | 1.797 | 64 | 10 | `target` |
| Covertype | `covertype/covertype.csv` | 581.012 | 54 | 7 | `cover_type` |

## Letter Recognition

- Nguồn: UCI Machine Learning Repository, Letter Recognition.
- Mỗi dòng mô tả một ký tự in hoa A-Z bằng 16 đặc trưng hình học.
- Có 1.332 dòng trùng hoàn toàn; sau khi loại trùng còn 18.668 mẫu.
- `train.csv`: 14.934 mẫu; `test.csv`: 3.734 mẫu.
- Split dùng `random_state=42`, stratify theo nhãn và không có feature vector
  trùng giữa train/test.

## Handwritten Digits

- Nguồn: `sklearn.datasets.load_digits`.
- 1.797 ảnh xám 8 x 8 được trải thành 64 feature `pixel_00` đến `pixel_63`.
- Nhãn `target` nhận giá trị 0-9.

## Covertype

- Nguồn: UCI Machine Learning Repository, Covertype.
- 581.012 mẫu với 54 feature: 10 biến địa hình/khoảng cách, 4 biến Wilderness
  Area và 40 biến Soil Type dạng one-hot.
- Nhãn `cover_type` nhận giá trị 1-7; dữ liệu dùng trong bài không có missing value.

## File cần dùng

- Notebook 01, 04 và 05 dùng `letter_recognition/train.csv` và `test.csv`.
- Notebook 02 có thể dùng `load_digits()`; `digits.csv` được nộp kèm để đối chiếu
  và chạy kiểm tra end-to-end.
- Notebook 03, 04 và 05 dùng `covertype/covertype.csv`.

Nguồn chính thức:

- Letter Recognition: https://archive.ics.uci.edu/dataset/59/letter+recognition
- Covertype: https://archive.ics.uci.edu/dataset/31/covertype
- Digits: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
