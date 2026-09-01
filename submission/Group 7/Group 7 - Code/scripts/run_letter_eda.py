"""Generate report-ready EDA artifacts for UCI Letter Recognition.

Run from the repository root after preparing the shared split:

    python scripts/run_letter_eda.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from decision_tree_lab2.config import FIGURES_DIR, RESULTS_DIR
from decision_tree_lab2.eda import (
    build_eda_insights,
    compute_class_distribution,
    compute_class_feature_profiles,
    compute_correlation_matrix,
    compute_feature_statistics,
)
from decision_tree_lab2.letter_data import LETTER_FEATURES, LETTER_TARGET
from decision_tree_lab2.results import build_result, save_result

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "letter_recognition" / "letter_recognition.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "letter_recognition"
CLEAN_CSV = PROCESSED_DIR / "letter_recognition_clean.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"

EXPERIMENT_ID = "eda_letter"
CLASS_TABLE = RESULTS_DIR / f"{EXPERIMENT_ID}__class_distribution.csv"
FEATURE_TABLE = RESULTS_DIR / f"{EXPERIMENT_ID}__feature_statistics.csv"
CORRELATION_TABLE = RESULTS_DIR / f"{EXPERIMENT_ID}__correlation_matrix.csv"
CLASS_MEANS_TABLE = RESULTS_DIR / f"{EXPERIMENT_ID}__class_feature_means_train.csv"
PROFILE_TABLE = RESULTS_DIR / f"{EXPERIMENT_ID}__class_feature_profiles_train.csv"
INSIGHTS_JSON = RESULTS_DIR / f"{EXPERIMENT_ID}__insights.json"
EDA_DOCUMENT = PROJECT_ROOT / "docs" / "LETTER_RECOGNITION_EDA.md"

FIGURE_STEMS = {
    "class_distribution": FIGURES_DIR / f"{EXPERIMENT_ID}__class_distribution",
    "feature_distributions": FIGURES_DIR / f"{EXPERIMENT_ID}__feature_distributions",
    "correlation_heatmap": FIGURES_DIR / f"{EXPERIMENT_ID}__correlation_heatmap",
    "class_feature_profiles": FIGURES_DIR / f"{EXPERIMENT_ID}__class_feature_profiles",
}

BLUE = "#2F6B9A"
ORANGE = "#D97732"
GRID = "#D8DEE6"
TEXT = "#243447"


def configure_style() -> None:
    """Set one consistent, print-friendly style for all EDA figures."""

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(figure: plt.Figure, stem: Path) -> list[str]:
    """Save one figure as both slide-ready PNG and report-ready PDF."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [
        str(png_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(pdf_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    ]


def plot_class_distribution(distribution: pd.DataFrame) -> plt.Figure:
    """Compare class counts before and after deduplication."""

    figure, axis = plt.subplots(figsize=(13, 6.5))
    positions = np.arange(len(distribution))
    width = 0.38
    axis.bar(
        positions - width / 2,
        distribution["raw_count"],
        width,
        label="Raw (20,000)",
        color=BLUE,
    )
    axis.bar(
        positions + width / 2,
        distribution["clean_count"],
        width,
        label="After exact deduplication (18,668)",
        color=ORANGE,
    )
    axis.set_title("Letter Recognition class distribution")
    axis.set_xlabel("Target letter")
    axis.set_ylabel("Number of samples")
    axis.set_xticks(positions, distribution[LETTER_TARGET])
    axis.set_ylim(0, distribution["raw_count"].max() * 1.13)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.grid(axis="x", visible=False)
    axis.legend(frameon=False, ncols=2, loc="upper center")
    figure.tight_layout()
    return figure


def plot_feature_distributions(cleaned: pd.DataFrame) -> plt.Figure:
    """Show all 16 integer-valued feature distributions on shared x limits."""

    figure, axes = plt.subplots(4, 4, figsize=(14, 11), sharex=True)
    bins = np.arange(-0.5, 16.5, 1)
    for index, (axis, feature) in enumerate(zip(axes.flat, LETTER_FEATURES, strict=True)):
        axis.hist(
            cleaned[feature],
            bins=bins,
            color=BLUE,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.35,
        )
        axis.axvline(
            cleaned[feature].median(),
            color=ORANGE,
            linewidth=1.5,
            linestyle="--",
        )
        axis.set_title(feature, fontsize=11)
        axis.set_xlim(-0.5, 15.5)
        axis.set_xticks([0, 5, 10, 15])
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.grid(axis="x", visible=False)
        if index % 4 == 0:
            axis.set_ylabel("Samples")
        if index >= 12:
            axis.set_xlabel("Scaled integer value")
    figure.suptitle(
        "Distributions of 16 engineered features",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.995,
        0.005,
        "Dashed line = median",
        ha="right",
        va="bottom",
        fontsize=9,
        color=TEXT,
    )
    figure.tight_layout()
    return figure


def plot_correlation_heatmap(correlation: pd.DataFrame) -> plt.Figure:
    """Plot the lower triangle of the feature correlation matrix."""

    figure, axis = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    annotations = correlation.map(lambda value: f"{value:.2f}" if abs(value) >= 0.40 else "")
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.35,
        linecolor="white",
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Pearson correlation", "shrink": 0.78},
        ax=axis,
    )
    axis.set_title("Feature correlation matrix")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(False)
    axis.tick_params(axis="x", rotation=45, labelsize=9)
    axis.tick_params(axis="y", rotation=0, labelsize=9)
    figure.tight_layout()
    return figure


def plot_class_feature_profiles(standardized_profiles: pd.DataFrame) -> plt.Figure:
    """Plot standardized class means computed from the training split only."""

    figure, axis = plt.subplots(figsize=(14, 10))
    limit = max(2.0, float(np.nanmax(np.abs(standardized_profiles.to_numpy()))))
    sns.heatmap(
        standardized_profiles,
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Standardized class mean (z-score)", "shrink": 0.78},
        ax=axis,
    )
    axis.set_title("Class–feature profiles (training split only)")
    axis.set_xlabel("Engineered feature")
    axis.set_ylabel("Target letter")
    axis.tick_params(axis="x", rotation=45, labelsize=9)
    axis.tick_params(axis="y", rotation=0, labelsize=9)
    figure.tight_layout()
    return figure


def render_eda_document(insights: dict[str, Any]) -> str:
    """Render concise observations that can be reused in report and slides."""

    correlations = "\n".join(
        f"| `{item['feature_1']}` | `{item['feature_2']}` | {item['correlation']:.3f} |"
        for item in insights["strongest_correlation_pairs"][:5]
    )
    separating = ", ".join(
        f"`{item['feature']}`" for item in insights["most_class_separating_features_train_only"]
    )
    variable = ", ".join(
        f"`{item['feature']}`" for item in insights["highest_variability_features"]
    )
    duplicate_classes = ", ".join(
        f"{item[LETTER_TARGET]} ({item['duplicates_removed']})"
        for item in insights["most_duplicates_removed"]
    )

    return f"""# Letter Recognition — Exploratory Data Analysis

## Phạm vi

EDA mô tả sử dụng toàn bộ `letter_recognition_clean.csv`. Phân tích quan hệ giữa
target và features sử dụng riêng `train.csv` để không nhìn trước test set. EDA không
thực hiện feature selection hoặc thay đổi split chung.

## Hình dùng cho report và slide

### Phân bố lớp trước và sau deduplication

![Class distribution](../figures/eda_letter__class_distribution.png)

Raw dataset khá cân bằng với tỷ lệ lớp lớn nhất/nhỏ nhất là
**{insights['raw_class_imbalance_ratio']:.3f}**. Sau khi loại exact duplicates, tỷ lệ
này là **{insights['clean_class_imbalance_ratio']:.3f}**; các lớp bị loại nhiều mẫu
trùng nhất là {duplicate_classes}. Vì vậy mọi metric multiclass phải báo thêm macro-F1,
không chỉ accuracy.

### Phân bố 16 features

![Feature distributions](../figures/eda_letter__feature_distributions.png)

Các feature có độ lệch chuẩn cao nhất là {variable}. Nhiều phân bố rời rạc và lệch,
phù hợp với mô tả đây là các thống kê ảnh đã được lượng tử hóa vào miền 0–15. Các điểm
bị IQR đánh dấu vẫn nằm trong miền hợp lệ nên không bị xóa.

### Tương quan giữa features

![Correlation heatmap](../figures/eda_letter__correlation_heatmap.png)

| Feature 1 | Feature 2 | Pearson r |
|---|---|---:|
{correlations}

Mean absolute pairwise correlation là
**{insights['mean_absolute_feature_correlation']:.3f}**. Tương quan không được dùng để
tự động loại feature ở giai đoạn này; Decision Tree có thể chọn split phi tuyến và
feature importance sẽ được đánh giá ở thẻ baseline.

### Quan hệ class–feature trên train set

![Class-feature profiles](../figures/eda_letter__class_feature_profiles.png)

Các feature có mức biến thiên giữa class lớn nhất trên train set là {separating}.
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
"""


def main() -> None:
    """Run the complete EDA pipeline for Trello card 04."""

    for path in (RAW_CSV, CLEAN_CSV, TRAIN_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run python scripts/prepare_letter_recognition.py first."
            )

    raw = pd.read_csv(RAW_CSV)
    cleaned = pd.read_csv(CLEAN_CSV)
    train = pd.read_csv(TRAIN_CSV)

    class_distribution = compute_class_distribution(raw, cleaned)
    feature_statistics = compute_feature_statistics(cleaned)
    correlation = compute_correlation_matrix(cleaned)
    class_means, standardized_profiles = compute_class_feature_profiles(train)
    insights = build_eda_insights(
        class_distribution,
        feature_statistics,
        correlation,
        standardized_profiles,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    class_distribution.to_csv(CLASS_TABLE, index=False)
    feature_statistics.to_csv(FEATURE_TABLE, index=False)
    correlation.to_csv(CORRELATION_TABLE)
    class_means.to_csv(CLASS_MEANS_TABLE)
    standardized_profiles.to_csv(PROFILE_TABLE)
    with INSIGHTS_JSON.open("w", encoding="utf-8") as file:
        json.dump(insights, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")

    configure_style()
    figure_paths: list[str] = []
    figure_paths.extend(
        save_figure(
            plot_class_distribution(class_distribution),
            FIGURE_STEMS["class_distribution"],
        )
    )
    figure_paths.extend(
        save_figure(
            plot_feature_distributions(cleaned),
            FIGURE_STEMS["feature_distributions"],
        )
    )
    figure_paths.extend(
        save_figure(
            plot_correlation_heatmap(correlation),
            FIGURE_STEMS["correlation_heatmap"],
        )
    )
    figure_paths.extend(
        save_figure(
            plot_class_feature_profiles(standardized_profiles),
            FIGURE_STEMS["class_feature_profiles"],
        )
    )

    metrics = {
        "raw_rows": len(raw),
        "clean_rows": len(cleaned),
        "raw_class_imbalance_ratio": insights["raw_class_imbalance_ratio"],
        "clean_class_imbalance_ratio": insights["clean_class_imbalance_ratio"],
        "mean_absolute_feature_correlation": insights["mean_absolute_feature_correlation"],
        "strongest_absolute_correlation": insights["strongest_correlation_pairs"][0][
            "absolute_correlation"
        ],
    }
    result = build_result(
        experiment_id=EXPERIMENT_ID,
        dataset="letter_recognition",
        model="descriptive_eda",
        metrics=metrics,
        figure_paths=[path for path in figure_paths if path.endswith(".png")],
        notes=(
            "Full cleaned data used for descriptive EDA; target-conditioned class-feature "
            "profiles computed from the canonical training split only."
        ),
    )
    save_result(result, RESULTS_DIR / f"{EXPERIMENT_ID}.json")
    EDA_DOCUMENT.write_text(render_eda_document(insights), encoding="utf-8")

    print(f"Generated {len(figure_paths)} figure files")
    print(f"Strongest correlation: {insights['strongest_correlation_pairs'][0]}")
    print(
        "Top class-separating features: "
        + ", ".join(
            item["feature"] for item in insights["most_class_separating_features_train_only"]
        )
    )
    print(f"Wrote: {EDA_DOCUMENT}")


if __name__ == "__main__":
    main()
