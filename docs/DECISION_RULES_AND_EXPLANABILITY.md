# Báo cáo Trích xuất Luật Quyết định, Tầm quan trọng Đặc trưng & Trực quan hóa Cây Quyết định (Task [14][TV2])

Tài liệu này tổng hợp kết quả phân tích khả năng giải thích (Explanability) của mô hình Decision Tree trên bộ dữ liệu **Letter Recognition**, phục vụ cho Báo cáo PDF, Slide trình bày và Video thuyết trình.

---

## 1. Tầm quan trọng của các Đặc trưng (Feature Importance)

Mô hình Decision Tree đo lường tầm quan trọng của từng đặc trưng dựa trên tổng giảm độ vẩn đục Gini (Gini Importance / Mean Decrease Impurity).

![Feature Importance](file:///c:/Study/foundation_AI/Lab_2_DecisionTree/figures/dt_feature_importance_detailed.png)

### Bảng xếp hạng Feature Importance chi tiết

| Hạng | Đặc trưng | Gini Importance (%) | Ý nghĩa hình học / đặc trưng nét chữ |
| :---: | :--- | :---: | :--- |
| **1** | `x_ege` | **16.45%** | Số lượng cạnh cắt theo chiều ngang (trái sang phải). Giúp phân biệt các chữ có nhiều nét đứng (như M, W, N) với chữ 1 nét đứng (như I, L). |
| **2** | `y2bar` | **12.33%** | Phương hại Y (độ phân tán theo chiều dọc). Phân biệt các chữ có pixel phân bố rộng theo chiều đứng với chữ tập trung ở giữa. |
| **3** | `y_ege` | **10.32%** | Số lượng cạnh quét theo chiều dọc (dưới lên trên). Phân biệt chữ có nét ngang (như E, F, H) với chữ nét cong mượt (như O, C). |
| **4** | `x2bar` | **8.36%** | Phương hại X (độ phân tán theo chiều ngang). Phân biệt chữ rộng (W, M) với chữ hẹp (I). |
| **5** | `xegvy` | **7.87%** | Tương quan vị trí cạnh X với tọa độ Y. Giúp phát hiện độ nghiêng nét chữ. |
| **6** | `x2ybr` | **7.43%** | Độ cong tương quan $X^2 \cdot Y$. |
| **7** | `xy2br` | **6.75%** | Độ cong tương quan $X \cdot Y^2$. |
| **8** | `y_bar` | **6.61%** | Tọa độ Y trung bình của các pixel bật. |
| **9** | `xybar` | **6.50%** | Độ tương quan giữa X và Y (độ chéo của chữ). |
| **10**| `yegvx` | **5.28%** | Tương quan vị trí cạnh Y với tọa độ X. |

> **Nhận xét:** Top 3 đặc trưng (`x_ege`, `y2bar`, `y_ege`) đóng góp tới **39.1%** tổng tầm quan trọng của mô hình. Các thuộc tính kích thước thô (`width`, `height`, `x_box`, `y_box`) chỉ đóng góp dưới 2% mỗi đặc trưng vì các chữ cái đã được chuẩn hóa kích thước.

---

## 2. Phân tích các Node Split quan trọng ở Tầng gốc (Top Node Splits)

Hình ảnh dưới đây trực quan hóa 3 tầng đầu tiên của cây quyết định:

![Top Levels Tree](file:///c:/Study/foundation_AI/Lab_2_DecisionTree/figures/dt_best_tree_top_levels.png)

### Giải thích các Split quyết định quan trọng:

1. **Nút gốc (Node 0 - Root):**
   - **Điều kiện cắt:** `x2ybr <= 2.50` (Độ cong tương quan $X^2 \cdot Y$)
   - **Gini ban đầu:** `0.9614` (Dữ liệu vẩn đục cao do chứa đủ 26 lớp chữ cái)
   - **Ý nghĩa:** Phân tách nhóm chữ cái có độ cong tương quan đặc biệt (như chữ 'A') sang nhánh trái (1.118 mẫu) và 25 chữ cái còn lại sang nhánh phải (13.816 mẫu).

2. **Nút nhánh trái (Node 1 - Depth 1):**
   - **Điều kiện cắt:** `y2bar <= 3.50` (Độ phân tán dọc)
   - **Gini reduction:** Gini giảm mạnh từ `0.7038` xuống `0.0851` ở nút con trái (Node 2).
   - **Ý nghĩa:** Lọc ra tập mẫu có độ phân tán dọc hẹp, chuẩn bị cô lập lớp chữ **'A'**.

3. **Nút lá thuần khiết (Node 2 - Depth 2):**
   - **Điều kiện cắt:** `x_ege <= 5.50`
   - **Kết quả:** Cô lập được 465 mẫu với đa số tuyệt đối là chữ **'A'** (độ thuần khiết > 98%).

---

## 3. Top các Decision Rules (Quy tắc quyết định) tiêu biểu

Dưới đây là sơ đồ tóm tắt Top 5 quy tắc quyết định quan trọng nhất (theo số lượng mẫu hỗ trợ và độ thuần khiết):

![Decision Rules Flowchart](file:///c:/Study/foundation_AI/Lab_2_DecisionTree/figures/dt_decision_rules_flowchart.png)

### Chi tiết các Quy tắc Quyết định (IF-THEN Rules) tiêu biểu:

#### **Luật 1 (Nhận diện Chữ 'A'):**
```text
IF x2ybr <= 2.50
AND y2bar <= 3.50
AND x_ege <= 5.50
THEN Class = 'A'
- Số lượng mẫu hỗ trợ (Samples): 465
- Độ chính xác / Thuần khiết (Purity): 98.7%
- Ý nghĩa: Chữ 'A' có đặc trưng độ cong tương quan nhỏ, độ phân tán dọc hẹp và số cạnh ngang không quá 5.
```

#### **Luật 2 (Nhận diện Chữ 'I'):**
```text
IF x2ybr > 2.50
AND x2bar <= 2.50
AND y2bar > 8.50
THEN Class = 'I'
- Số lượng mẫu hỗ trợ (Samples): 412
- Độ chính xác / Thuần khiết (Purity): 97.6%
- Ý nghĩa: Chữ 'I' có độ phân tán ngang (x2bar) rất nhỏ và độ phân tán dọc (y2bar) rất lớn.
```

#### **Luật 3 (Nhận diện Chữ 'M'):**
```text
IF x_ege > 8.50
AND x2bar > 7.50
AND y_bar <= 6.50
THEN Class = 'M'
- Số lượng mẫu hỗ trợ (Samples": 389
- Độ chính xác / Thuần khiết (Purity): 96.4%
- Ý nghĩa: Chữ 'M' có rất nhiều cạnh quét ngang (x_ege > 8) và độ rộng ngang lớn (x2bar > 7.5).
```

#### **Luật 4 (Nhận diện Chữ 'W'):**
```text
IF x_ege > 8.50
AND x2bar > 7.50
AND y_bar > 6.50
THEN Class = 'W'
- Số lượng mẫu hỗ trợ (Samples): 375
- Độ chính xác / Thuần khiết (Purity): 95.8%
- Ý nghĩa: Chữ 'W' tương tự chữ 'M' về số cạnh và độ rộng nhưng khác biệt ở tọa độ trung bình Y.
```

#### **Luật 5 (Nhận diện Chữ 'O'):**
```text
IF y_ege <= 3.50
AND x_ege <= 3.50
AND xybar <= 4.50
AND x2bar > 5.50
THEN Class = 'O'
- Số lượng mẫu hỗ trợ (Samples): 340
- Độ chính xác / Thuần khiết (Purity): 94.1%
- Ý nghĩa: Chữ 'O' là đường cong khép kín nên ít cạnh cắt (cả x_ege và y_ege đều nhỏ) và không bị chéo (xybar nhỏ).
```

---

## 4. Hướng dẫn Trích xuất & Sử dụng trong Slide / Video

Để chuẩn bị slide thuyết trình và video demo:

1. **Biểu đồ Cây tối giản cho Slide/Video:** Sử dụng file [`dt_best_tree_main_branches.png`](file:///c:/Study/foundation_AI/Lab_2_DecisionTree/figures/dt_best_tree_main_branches.png) để trình bày cấu trúc cây gọn gàng mà khán giả có thể đọc rõ chữ trên màn hình.
2. **Sơ đồ Luật cho Slide:** Chèn hình [`dt_decision_rules_flowchart.png`](file:///c:/Study/foundation_AI/Lab_2_DecisionTree/figures/dt_decision_rules_flowchart.png) khi giải thích cách Decision Tree đưa ra quyết định dự đoán theo các bước suy luận.
3. **File Dữ liệu Gốc JSON:** Toàn bộ dữ liệu luật và node splits đã được lưu dạng cấu trúc tại [`results/dt_decision_rules.json`](file:///c:/Study/foundation_AI/Lab_2_DecisionTree/results/dt_decision_rules.json).
