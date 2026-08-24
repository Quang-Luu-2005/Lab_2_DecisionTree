"""Audit, clean, document, and split the UCI Letter Recognition dataset.

Run from the repository root:

    python scripts/prepare_letter_recognition.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from decision_tree_lab2.config import PROCESSED_DATA_DIR
from decision_tree_lab2.letter_data import (
    LETTER_CLASSES,
    LETTER_FEATURES,
    LETTER_TARGET,
    audit_letter_recognition,
    clean_letter_recognition,
    split_letter_recognition,
)

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "letter_recognition" / "letter_recognition.csv"
OUTPUT_DIR = PROCESSED_DATA_DIR / "letter_recognition"
CLEAN_CSV = OUTPUT_DIR / "letter_recognition_clean.csv"
TRAIN_CSV = OUTPUT_DIR / "train.csv"
TEST_CSV = OUTPUT_DIR / "test.csv"
CLASS_DISTRIBUTION_CSV = OUTPUT_DIR / "class_distribution.csv"
FEATURE_SUMMARY_CSV = OUTPUT_DIR / "feature_summary.csv"
REPORT_JSON = OUTPUT_DIR / "preparation_report.json"
DATASET_DESCRIPTION = PROJECT_ROOT / "docs" / "LETTER_RECOGNITION_DATASET.md"

FEATURE_DESCRIPTIONS = {
    "x_box": "Horizontal position of the character bounding box",
    "y_box": "Vertical position of the character bounding box",
    "width": "Width of the character bounding box",
    "high": "Height of the character bounding box",
    "onpix": "Number of foreground pixels",
    "x_bar": "Mean horizontal position of foreground pixels",
    "y_bar": "Mean vertical position of foreground pixels",
    "x2bar": "Horizontal variance",
    "y2bar": "Vertical variance",
    "xybar": "Horizontal-vertical correlation",
    "x2ybr": "Mean of x-squared times y",
    "xy2br": "Mean of x times y-squared",
    "x_ege": "Mean left-to-right edge count",
    "xegvy": "Correlation of horizontal edge count with y",
    "y_ege": "Mean bottom-to-top edge count",
    "yegvx": "Correlation of vertical edge count with x",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_class_distribution(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Build one class table covering raw, cleaned, train, and test sets."""

    table = pd.DataFrame({LETTER_TARGET: LETTER_CLASSES})
    for name, frame in (
        ("raw", raw),
        ("clean", cleaned),
        ("train", train),
        ("test", test),
    ):
        counts = frame[LETTER_TARGET].value_counts().reindex(LETTER_CLASSES, fill_value=0)
        table[f"{name}_count"] = table[LETTER_TARGET].map(counts).astype("int64")
        table[f"{name}_percentage"] = table[f"{name}_count"] / len(frame) * 100
    return table


def render_dataset_description(report: dict[str, Any]) -> str:
    """Render report-ready Markdown from the deterministic preparation report."""

    raw = report["raw_audit"]
    cleaning = report["cleaning"]
    split = report["split"]
    class_counts = [item["count"] for item in raw["class_distribution"].values()]

    feature_rows = "\n".join(
        f"| `{feature}` | {FEATURE_DESCRIPTIONS[feature]} | Integer, 0–15 |"
        for feature in LETTER_FEATURES
    )
    return f"""# Letter Recognition Dataset

## Nguồn và mục tiêu

Dataset chính của project là **Letter Recognition** từ UCI Machine Learning Repository.
Mỗi quan sát biểu diễn một ảnh ký tự in hoa đã được chuyển thành 16 đặc trưng thống kê
và cạnh. Bài toán là phân loại ký tự mục tiêu `{LETTER_TARGET}` vào 26 lớp từ A đến Z.

- Source: https://archive.ics.uci.edu/dataset/59/letter%2Brecognition
- Raw file: `data/raw/letter_recognition/letter_recognition.csv`
- Số quan sát raw: **{raw['rows']:,}**
- Số đặc trưng: **{raw['feature_count']}**
- Số lớp: **{raw['class_count']}**
- Số mẫu mỗi lớp: **{min(class_counts)}–{max(class_counts)}**

## Các đặc trưng

| Feature | Ý nghĩa | Miền giá trị |
|---|---|---|
{feature_rows}

## Kiểm tra chất lượng

| Kiểm tra | Kết quả |
|---|---:|
| Missing cells | {raw['missing_cells']:,} |
| Exact duplicate rows | {raw['exact_duplicate_rows']:,} |
| Rows in duplicate-feature groups | {raw['rows_in_duplicate_feature_groups']:,} |
| Feature groups with conflicting labels | {raw['conflicting_feature_groups']:,} |
| Values outside 0–15 | {raw['out_of_range_cells']:,} |
| Non-integer feature values | {raw['non_integer_feature_cells']:,} |
| IQR-flagged cells | {raw['iqr_outlier_cells']:,} |

Các điểm bị IQR đánh dấu không bị xóa vì đặc trưng đã được UCI scale vào miền hữu hạn
0–15; chúng là giá trị hợp lệ theo data contract. Không cần imputation. Các dòng trùng
hoàn toàn được loại trước khi split để cùng một vector đặc trưng không xuất hiện ở cả
train và test.

## Preprocessing và split dùng chung

- Sau deduplication: **{cleaning['rows_after']:,}** mẫu
  (loại {cleaning['exact_duplicates_removed']:,} mẫu,
  {cleaning['removal_percentage']:.2f}%).
- Không scaling cho Decision Tree.
- `test_size={split['test_size']}`
- `random_state={split['random_state']}`
- `stratify={str(split['stratify']).lower()}`
- Train: **{split['train_rows']:,}** mẫu; test: **{split['test_rows']:,}** mẫu.
- Cả train và test đều có **{split['train_class_count']}** lớp.
- Số vector đặc trưng xuất hiện ở cả hai split: **{split['feature_vectors_shared_across_splits']}**.

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
"""


def main() -> None:
    """Execute cards 02 and 03 and materialize their outputs."""

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Missing {RAW_CSV}. Run python scripts/download_datasets.py first."
        )

    raw = pd.read_csv(RAW_CSV)
    raw_audit = audit_letter_recognition(raw)
    if raw_audit["rows"] != 20_000:
        raise ValueError(f"Expected 20,000 rows, found {raw_audit['rows']}")
    if raw_audit["feature_count"] != 16:
        raise ValueError(f"Expected 16 features, found {raw_audit['feature_count']}")
    if raw_audit["class_labels"] != list(LETTER_CLASSES):
        raise ValueError("Expected target classes A-Z")

    cleaned, cleaning_report = clean_letter_recognition(raw)
    clean_audit = audit_letter_recognition(cleaned)
    train, test, split_report = split_letter_recognition(cleaned)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_CSV, index=False)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    class_distribution = build_class_distribution(raw, cleaned, train, test)
    class_distribution.to_csv(CLASS_DISTRIBUTION_CSV, index=False)
    feature_summary = pd.DataFrame.from_dict(raw_audit["feature_summary"], orient="index")
    feature_summary.index.name = "feature"
    feature_summary.reset_index().to_csv(FEATURE_SUMMARY_CSV, index=False)

    report = {
        "schema_version": "1.0",
        "dataset": "letter_recognition",
        "source_file": str(RAW_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_sha256": sha256(RAW_CSV),
        "raw_audit": raw_audit,
        "cleaning": cleaning_report,
        "clean_audit": clean_audit,
        "split": split_report,
        "outputs": {
            "clean_csv": str(CLEAN_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "train_csv": str(TRAIN_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "test_csv": str(TEST_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "class_distribution_csv": str(CLASS_DISTRIBUTION_CSV.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "feature_summary_csv": str(FEATURE_SUMMARY_CSV.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
        },
    }
    with REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")

    DATASET_DESCRIPTION.write_text(
        render_dataset_description(report),
        encoding="utf-8",
    )

    print(f"Raw rows: {raw_audit['rows']}")
    print(f"Exact duplicates removed: {cleaning_report['exact_duplicates_removed']}")
    print(f"Clean rows: {cleaning_report['rows_after']}")
    print(f"Train/test rows: {split_report['train_rows']}/{split_report['test_rows']}")
    print("Cross-split feature overlap: " f"{split_report['feature_vectors_shared_across_splits']}")
    print(f"Wrote: {OUTPUT_DIR}")
    print(f"Wrote: {DATASET_DESCRIPTION}")


if __name__ == "__main__":
    main()
