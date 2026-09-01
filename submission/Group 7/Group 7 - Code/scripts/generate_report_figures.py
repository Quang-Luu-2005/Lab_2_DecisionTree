"""Generate standalone, publication-ready figures for the LaTeX report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "report" / "artifacts" / "results"
FIGURES = ROOT / "docs" / "report" / "artifacts" / "figures"

MODEL_ORDER = ["Decision Tree", "Random Forest", "SVM (RBF)", "KNN"]
MODEL_LABELS = ["Decision Tree", "Random Forest", "SVM (RBF)", "KNN"]
DATASET_LABELS = {
    "letter_recognition": "Letter Recognition",
    "handwritten_digits": "Handwritten Digits",
    "covertype": "Covertype",
}
HS_ORDER = [
    "E0 Baseline CART",
    "E1 Pre-pruned CART",
    "E2 CCP-pruned CART",
    "E3 HS-DT",
    "E4 CCP+HS",
]
HS_LABELS = ["E0 Baseline", "E1 Pre-pruned", "E2 CCP", "E3 HS", "E4 CCP+HS"]
COLORS = {
    "accuracy": "#3B6EA8",
    "f1": "#E68A3F",
    "letter_recognition": "#4C78A8",
    "handwritten_digits": "#F58518",
    "covertype": "#54A24B",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, decimals: int = 3) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=f"%.{decimals}f", padding=2, fontsize=8)


def model_performance_figures(comparison: pd.DataFrame) -> None:
    for dataset, dataset_label in DATASET_LABELS.items():
        frame = comparison[comparison["dataset"] == dataset].set_index("model")
        frame = frame.loc[MODEL_ORDER]
        x = np.arange(len(frame))
        width = 0.36

        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        ax.bar(
            x - width / 2,
            frame["test_accuracy"],
            width,
            label="Accuracy",
            color=COLORS["accuracy"],
        )
        ax.bar(
            x + width / 2,
            frame["f1_macro"],
            width,
            label="Macro-F1",
            color=COLORS["f1"],
        )
        ax.set_title(f"Model performance - {dataset_label}")
        ax.set_ylabel("Score")
        ax.set_xticks(x, MODEL_LABELS)
        lower = max(0.0, min(frame["f1_macro"].min(), frame["test_accuracy"].min()) - 0.08)
        ax.set_ylim(lower, 1.04)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        annotate_bars(ax)
        fig.tight_layout()
        save(fig, f"report_model_performance__{dataset}.png")


def model_gap_figure(comparison: pd.DataFrame) -> None:
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    for index, (dataset, label) in enumerate(DATASET_LABELS.items()):
        frame = comparison[comparison["dataset"] == dataset].set_index("model").loc[MODEL_ORDER]
        ax.bar(
            x + (index - 1) * width,
            frame["generalization_gap"],
            width,
            label=label,
            color=COLORS[dataset],
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Generalization gap by model and dataset")
    ax.set_ylabel("Train accuracy - test accuracy")
    ax.set_xticks(x, MODEL_LABELS)
    ax.legend(ncols=3, loc="upper right")
    fig.tight_layout()
    save(fig, "report_model_generalization_gap.png")


def model_runtime_figure(comparison: pd.DataFrame) -> None:
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    for index, (dataset, label) in enumerate(DATASET_LABELS.items()):
        frame = comparison[comparison["dataset"] == dataset].set_index("model").loc[MODEL_ORDER]
        ax.bar(
            x + (index - 1) * width,
            frame["total_model_seconds"],
            width,
            label=label,
            color=COLORS[dataset],
        )
    ax.set_yscale("log")
    ax.set_title("Fit and full-test prediction time")
    ax.set_ylabel("Seconds (log scale)")
    ax.set_xticks(x, MODEL_LABELS)
    ax.legend(ncols=3, loc="upper left")
    fig.tight_layout()
    save(fig, "report_model_runtime.png")


def hs_performance_figures(hs: pd.DataFrame) -> None:
    for dataset, dataset_label in DATASET_LABELS.items():
        frame = hs[hs["dataset"] == dataset].set_index("model").loc[HS_ORDER]
        x = np.arange(len(frame))
        width = 0.36
        fig, ax = plt.subplots(figsize=(8.4, 4.7))
        ax.bar(
            x - width / 2,
            frame["test_accuracy"],
            width,
            label="Accuracy",
            color=COLORS["accuracy"],
        )
        ax.bar(
            x + width / 2,
            frame["f1_macro"],
            width,
            label="Macro-F1",
            color=COLORS["f1"],
        )
        ax.set_title(f"Decision Tree regularization - {dataset_label}")
        ax.set_ylabel("Score")
        ax.set_xticks(x, HS_LABELS, rotation=12, ha="right")
        lower = max(0.0, min(frame["f1_macro"].min(), frame["test_accuracy"].min()) - 0.04)
        ax.set_ylim(lower, 1.03)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        annotate_bars(ax)
        fig.tight_layout()
        save(fig, f"report_hs_performance__{dataset}.png")


def hs_complexity_figure(hs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = {"letter_recognition": "o", "handwritten_digits": "X", "covertype": "s"}
    annotation_offsets = {
        "letter_recognition": [(-20, -13), (5, 5), (5, -13), (5, 8), (-20, 8)],
        "handwritten_digits": [(5, 8), (5, 5), (5, -13), (5, -13), (5, 8)],
        "covertype": [(5, -13), (5, 5), (-20, -13), (5, 8), (-20, 8)],
    }
    for dataset, dataset_label in DATASET_LABELS.items():
        frame = hs[hs["dataset"] == dataset].set_index("model").loc[HS_ORDER]
        ax.scatter(
            frame["leaf_count"],
            frame["f1_macro"],
            s=62,
            marker=markers[dataset],
            color=COLORS[dataset],
            label=dataset_label,
            edgecolor="white",
            linewidth=0.6,
        )
        for model, leaf_count, f1, offset in zip(
            HS_LABELS,
            frame["leaf_count"],
            frame["f1_macro"],
            annotation_offsets[dataset],
        ):
            ax.annotate(
                model.split()[0],
                (leaf_count, f1),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xscale("log")
    ax.set_title("Macro-F1 versus Decision Tree complexity")
    ax.set_xlabel("Number of leaves (log scale)")
    ax.set_ylabel("Macro-F1")
    ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, "report_hs_performance_vs_complexity.png")


def hs_gap_figure(hs: pd.DataFrame) -> None:
    x = np.arange(len(HS_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for index, (dataset, dataset_label) in enumerate(DATASET_LABELS.items()):
        frame = hs[hs["dataset"] == dataset].set_index("model").loc[HS_ORDER]
        ax.bar(
            x + (index - 1) * width,
            frame["generalization_gap"],
            width,
            label=dataset_label,
            color=COLORS[dataset],
        )
    ax.set_title("Generalization gap after Decision Tree regularization")
    ax.set_ylabel("Train accuracy - test accuracy")
    ax.set_xticks(x, HS_LABELS, rotation=12, ha="right")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    save(fig, "report_hs_generalization_gap.png")


def main() -> None:
    configure_style()
    comparison = pd.read_csv(
        RESULTS / "gpu_three_dataset_four_model_comparison__summary.csv"
    )
    hs = pd.read_csv(RESULTS / "dt_hierarchical_shrinkage__summary.csv")
    model_performance_figures(comparison)
    model_gap_figure(comparison)
    model_runtime_figure(comparison)
    hs_performance_figures(hs)
    hs_complexity_figure(hs)
    hs_gap_figure(hs)


if __name__ == "__main__":
    main()
