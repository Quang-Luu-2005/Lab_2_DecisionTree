# Kịch bản thuyết trình - Nhóm 7

Thời lượng mục tiêu: 12-15 phút, dùng `docs/slides/decision_tree_lab2_presentation.pdf`.

## Phân vai

- Lưu Huy Minh Quang (23127016): trình bày chính, điều khiển slide và demo; phụ trách mở đầu, dữ liệu, baseline, cải tiến, custom DT, benchmark và kết luận.
- Ngô Quang Thắng (23127473): nếu tham gia, trình bày Gini so với Entropy, depth sweep, pruning, decision rules và feature importance.
- Vũ Lê Trọng Văn (20127095): chưa có phần việc độc lập được xác minh.
- Nguyễn Duy Khang (23127202): quay, biên tập và hoàn thiện video thuyết trình.

## Mạch trình bày

1. Slide 1-3 (1 phút): nêu bài toán, nguy cơ overfitting và phạm vi core/extended study.
2. Slide 4-6 (1,5 phút): đặt CART trong công trình liên quan, giải thích Gini, mức giảm impurity và vòng lặp sinh cây.
3. Slide 7-12 (2 phút): giới thiệu ba dataset, mất cân bằng lớp, protocol train-only validation và định nghĩa accuracy, confusion matrix, precision, recall, Macro-F1.
4. Slide 13-17 (2 phút): đọc cây baseline, decision paths, ba metric chính, confusion matrix và feature importance.
5. Slide 18-25 (3 phút): giải thích ký hiệu E0-E4; cơ chế, tham số và kết quả riêng của pre-pruning, CCP và Hierarchical Shrinkage.
6. Slide 26-28 (1,5 phút): so sánh E0-E4, giải thích vì sao chọn E4 và kiểm tra robustness trên ba dataset.
7. Slide 29-31 (1,5 phút): giới thiệu cơ chế của RF/SVM/KNN, so sánh model thắng theo dataset và báo cáo độ ổn định bằng CV.
8. Slide 32-33 (1 phút): trình bày custom Decision Tree, đối chiếu sklearn và xác nhận pipeline end-to-end.
9. Slide 34-35 (1 phút): nêu threats to validity và kết luận theo ba câu hỏi nghiên cứu.
10. Appendix A1-A3: chỉ mở khi giảng viên hỏi về bảng benchmark đầy đủ, decision rules hoặc đóng góp nhóm.

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
