# Đối chiếu đóng góp nhóm từ GitHub và Trello

Ngày đối chiếu: 01/09/2026.

## Phương pháp

- Phạm vi là 27 card hiện nằm trong danh sách `Hoàn thành` của board Trello `Lab2_DecisionTree`.
- Mỗi card hoàn thành được tính một đơn vị để tỷ lệ phản ánh đầu việc bàn giao, không phản ánh số dòng code.
- GitHub được dùng để xác minh tác giả commit và nội dung artifact; xác nhận của nhóm được dùng để quy phần việc còn lại cho người thực hiện chính.
- Hai card bonus B01 và B02 được tính vì đã hoàn thành và tạo artifact thực tế.

## Công việc được xác minh của Thắng

Tài khoản GitHub: `Ganth1811` (`quangthangngo181@gmail.com`).

| Card Trello | Công việc | Bằng chứng GitHub |
|---|---|---|
| 09 | So sánh Gini và Entropy | [Commit af0d0e1](https://github.com/Quang-Luu-2005/Lab_2_DecisionTree/commit/af0d0e1eeaf471cf9c26b8df6433979bd8c7e02e) |
| 10 | Depth sweep, phân tích underfitting/overfitting | [Commit af0d0e1](https://github.com/Quang-Luu-2005/Lab_2_DecisionTree/commit/af0d0e1eeaf471cf9c26b8df6433979bd8c7e02e) |
| 11 | Cải tiến DT bằng pre-pruning và post-pruning | [Commit a872072](https://github.com/Quang-Luu-2005/Lab_2_DecisionTree/commit/a872072265650ce8d71071cc611a33be6af88c3a) |
| 14 | Decision rules, feature importance và hình cây | [Commit 976002f](https://github.com/Quang-Luu-2005/Lab_2_DecisionTree/commit/976002f17e339b8cefdf1a4bcc03c2be83f74ea6) |

Repository có 27 commit trên toàn bộ hai nhánh `main` và `Thang`. Thắng có 3/27 commit và 4/27 card hoàn thành có thể ánh xạ trực tiếp, nhưng cả ba commit hiện chỉ nằm trên branch `Thang`, chưa được tích hợp vào `main`.

## Trạng thái tích hợp vào main

- `main` có 24 commit; toàn bộ được ghi nhận dưới hai identity của cùng người thực hiện chính: `Lưu Quang` và `Quang-Luu-2005`.
- `Thang` có 3 commit riêng, còn `main` có 10 commit riêng kể từ merge base `96a829b`.
- Không có merge commit trên `main` và không có Pull Request trong repository.
- Ba commit `af0d0e1`, `a872072`, `976002f` không phải tổ tiên của `main`; các artifact tương ứng vẫn chỉ tồn tại trên branch `Thang`.

Số card có thể ánh xạ trực tiếp tương ứng 85,2% / 14,8%, còn lịch sử tích hợp `main` tương ứng 100,0% / 0,0%. Để phản ánh cả khối lượng hoàn thiện bản nộp và phần việc Thắng đã thực hiện trên branch riêng, nhóm thống nhất ghi nhận tỷ lệ đóng góp cuối là 90,0% / 10,0%.

## Bảng tỷ lệ đóng góp

| Thành viên | Mô tả công việc | Bằng chứng | Tỷ lệ thống nhất |
|---|---|---|---:|
| Người thực hiện chính (`Quang-Luu-2005` / Lưu Quang) | Khởi tạo và điều phối project; dữ liệu/EDA; baseline; benchmark; cross-dataset; cải tiến mô hình; tổng hợp kết quả; report/slide; custom DT; kiểm thử end-to-end; tích hợp `main` và đóng gói | 24 commit trên `main`; 23 card được ghi nhận | 90,0% |
| Thắng (`Ganth1811`) | Gini vs Entropy (09); depth sweep (10); pre/post-pruning (11); decision rules, feature importance và hình cây (14) trên branch `Thang` | 3 commit branch; chưa merge vào `main` | 10,0% |
| Vũ Lê Trọng Văn | Chưa có task hoàn thành độc lập được xác minh | Không có | 0,0% |
| Nguyễn Duy Khang | Chưa có task hoàn thành độc lập được xác minh | Không có | 0,0% |
| **Tổng** |  |  | **100,0%** |

## Lưu ý sử dụng

Bảng trên là bản có thể kiểm chứng từ GitHub/Trello và xác nhận của người thực hiện chính. Trước khi nộp, nhóm nên xác nhận lại xem có đóng góp offline, quay video, thuyết trình hoặc chỉnh sửa chưa được commit hay không; nếu có thì cần quy đổi và cập nhật tỷ lệ.
