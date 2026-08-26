# Decision Tree Lab 2 report

Thư mục này là báo cáo thật của project; `docs/references/report_sample/` chỉ được giữ làm
mẫu tham khảo. Các figure và result trong `artifacts/` được giải nén từ năm lần chạy Kaggle:

- `gpu_three_dataset_four_model_comparison__outputs.zip`
- `dt_hierarchical_shrinkage__outputs.zip`
- `dt_covertype_scalability__outputs.zip`
- `dt_digits_baseline__outputs.zip`
- `dt_letter_baseline__outputs.zip`

Report chỉ lưu figures và results cần thiết cho việc đối chiếu; các model `.joblib` trong ZIP
không được giữ lại vì không cần cho việc biên dịch. Metadata ghi nhận notebook tổng hợp dùng
Tesla T4 cho ba mô hình cuML; các notebook Decision Tree và Hierarchical Shrinkage chạy trên
CPU dù Kaggle có hiển thị GPU.

Biên dịch từ thư mục gốc project:

```powershell
Push-Location docs/report
latexmk -pdf -interaction=nonstopmode -halt-on-error `
  -outdir=../../output/pdf decision_tree_lab2_report.tex
Pop-Location
```

File PDF cuối nằm tại `output/pdf/decision_tree_lab2_report.pdf`.
