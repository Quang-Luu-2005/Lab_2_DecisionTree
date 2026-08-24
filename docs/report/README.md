# Decision Tree Lab 2 report

Thư mục này là báo cáo thật của project; `docs/references/report_sample/` chỉ được giữ làm
mẫu tham khảo. Các figure và result trong `artifacts/` được giải nén từ ba lần chạy Kaggle:

- `dt_letter_baseline__outputs.zip`
- `dt_digits_baseline__outputs.zip`
- `dt_covertype_scalability__outputs.zip`

Chỉ `figures/` và `results/` được lưu cùng report. Model `.joblib` không được chép vào vì
không cần cho việc biên dịch và làm tăng kích thước repository.

Biên dịch từ thư mục gốc project:

```powershell
Push-Location docs/report
latexmk -pdf -interaction=nonstopmode -halt-on-error `
  -outdir=../../output/pdf decision_tree_lab2_report.tex
Pop-Location
```

File PDF cuối nằm tại `output/pdf/decision_tree_lab2_report.pdf`.
