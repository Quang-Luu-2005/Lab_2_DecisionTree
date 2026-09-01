# Kịch bản thuyết trình - Nhóm 7

Thời lượng mục tiêu: 12-15 phút, dùng `docs/slides/decision_tree_lab2_presentation.pdf`.

## Phân vai

- Lưu Huy Minh Quang (23127016): trình bày chính, điều khiển slide và demo; phụ trách mở đầu, dữ liệu, baseline, cải tiến, custom DT, benchmark và kết luận.
- Ngô Quang Thắng (23127473): nếu tham gia, trình bày Gini so với Entropy, depth sweep, pruning, decision rules và feature importance.
- Vũ Lê Trọng Văn (20127095), Nguyễn Duy Khang (23127202): chưa có phần việc độc lập được xác minh; chỉ phân phần nói sau khi nhóm xác nhận thực tế tham gia video.

## Mạch trình bày

1. Slide 1-3 (1 phút): nêu bài toán, nguy cơ overfitting của cây và vai trò ba dataset.
2. Slide 4-8 (2 phút): giải thích Gini/Entropy, quy tắc split, protocol 80/20 và baseline Covertype.
3. Slide 9-12 (2 phút): đọc decision path, feature importance, confusion matrix và hạn chế của cây sâu.
4. Slide 13-18 (3 phút): trình bày pre-pruning, CCP, Hierarchical Shrinkage và lựa chọn tham số chỉ trên train/CV.
5. Slide 19-21 (2 phút): so sánh E0-E4 và nhấn mạnh CCP + HS giảm số lá 31,7%, giảm train-test gap 17,6%, đồng thời tăng macro-F1 0,19 điểm phần trăm so với baseline.
6. Slide 22-23 (2 phút): so sánh Decision Tree với RF, SVM, KNN trên ba dataset; không có mô hình thắng mọi dataset.
7. Slide 24 (1 phút): chốt câu trả lời cho ba câu hỏi nghiên cứu và giới hạn của kết quả.
8. Appendix A1-A6: chỉ mở khi giảng viên hỏi về bảng đầy đủ, tham số, rules, feature importance, hardware hoặc đóng góp nhóm.

## Demo ngắn

```powershell
python scripts\compare_custom_tree.py
python scripts\run_end_to_end_check.py
```

Nêu ba điểm: custom tree là triển khai NumPy từ đầu; phép so sánh dùng cùng split/seed/độ sâu; pipeline end-to-end đã kiểm tra cả Letter Recognition, Digits và Covertype.

## Checklist trước khi quay

- Mở slide ở chế độ toàn màn hình và thử toàn bộ hình/bảng.
- Chạy hai lệnh demo trước, tránh chờ train trong lúc ghi hình.
- Đọc số liệu đúng đơn vị: macro-F1, điểm phần trăm, số lá và train-test gap.
- Không tuyên bố causal effect hoặc GPU speedup ngoài phạm vi thí nghiệm.
- Kiểm tra microphone, độ phân giải, thời lượng và quyền truy cập link video.
