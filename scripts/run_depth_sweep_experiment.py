"""Execute Card [10][TV2] Depth sweep — underfitting/overfitting curves.

Sweeps max_depth from 1 to 35 and unconstrained (None) for DecisionTreeClassifier.
Records train/test accuracy, macro-F1, error rate, generalization gap, depth, and leaf count.
Generates complexity vs performance plots with shaded Underfitting, Optimal, and Overfitting zones.
Saves schema v1.0 result JSON, CSV summary, and publication-quality figures.

Run from repository root:
    .venv\\Scripts\\python scripts/run_depth_sweep_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from decision_tree_lab2.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    STRATIFY_SPLIT,
    TEST_SIZE,
    ensure_project_dirs,
)
from decision_tree_lab2.letter_data import LETTER_FEATURES, LETTER_TARGET
from decision_tree_lab2.results import build_result, save_result


def load_letter_recognition() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load pre-split clean Letter Recognition datasets."""

    train_path = PROCESSED_DATA_DIR / "letter_recognition" / "train.csv"
    test_path = PROCESSED_DATA_DIR / "letter_recognition" / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Clean split datasets not found. Run scripts/prepare_letter_recognition.py first.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.loc[:, list(LETTER_FEATURES)]
    y_train = train[LETTER_TARGET]
    X_test = test.loc[:, list(LETTER_FEATURES)]
    y_test = test[LETTER_TARGET]

    return X_train, X_test, y_train, y_test


def run_depth_sweep(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    depths: list[int | None],
) -> pd.DataFrame:
    """Sweep max_depth hyperparameter across criteria and return structured results."""

    records = []

    for criterion in ["entropy", "gini"]:
        for d in depths:
            d_param = d if d is not None else 999
            d_label = str(d) if d is not None else "None (Unconstrained)"

            clf = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=d,
                random_state=RANDOM_STATE,
            )

            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            tr_pred = clf.predict(X_train)
            te_pred = clf.predict(X_test)
            pred_time = time.perf_counter() - t1

            tr_acc = float(accuracy_score(y_train, tr_pred))
            te_acc = float(accuracy_score(y_test, te_pred))
            tr_f1 = float(f1_score(y_train, tr_pred, average="macro", zero_division=0))
            te_f1 = float(f1_score(y_test, te_pred, average="macro", zero_division=0))

            actual_depth = int(clf.get_depth())
            leaves = int(clf.get_n_leaves())
            nodes = int(clf.tree_.node_count)
            gen_gap = float(tr_acc - te_acc)
            gen_gap_f1 = float(tr_f1 - te_f1)
            error_rate = float(1.0 - te_acc)

            # Categorize region
            if d_param <= 8:
                region = "Underfitting"
            elif 9 <= d_param <= 18:
                region = "Optimal"
            else:
                region = "Overfitting"

            records.append(
                {
                    "criterion": criterion,
                    "max_depth_param": d_param,
                    "max_depth_label": d_label,
                    "actual_depth": actual_depth,
                    "leaf_count": leaves,
                    "node_count": nodes,
                    "train_accuracy": tr_acc,
                    "test_accuracy": te_acc,
                    "error_rate": error_rate,
                    "train_macro_f1": tr_f1,
                    "test_macro_f1": te_f1,
                    "generalization_gap_acc": gen_gap,
                    "generalization_gap_f1": gen_gap_f1,
                    "training_seconds": fit_time,
                    "prediction_seconds": pred_time,
                    "region": region,
                }
            )

    return pd.DataFrame(records)


def set_plotting_style() -> None:
    """Apply clean aesthetic theme."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Inter", "DejaVu Sans", "Arial"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.autolayout": True,
            "figure.dpi": 300,
        }
    )


def plot_performance_curves(df_sweep: pd.DataFrame, output_path: Path) -> None:
    """Plot Train vs Test Accuracy & Macro-F1 with shaded Underfitting, Optimal, and Overfitting zones."""
    set_plotting_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    df_plot = df_sweep[df_sweep["max_depth_param"] != 999].copy()

    # Panel A: Accuracy Curves for Entropy & Gini
    ax1 = axes[0, 0]

    # Shading regions
    ax1.axvspan(1, 8, color="#fee0d2", alpha=0.5, label="Underfitting Zone (Depth 1–8)")
    ax1.axvspan(8, 18, color="#e5f5e0", alpha=0.5, label="Optimal Zone (Depth 9–18)")
    ax1.axvspan(18, 35, color="#deebf7", alpha=0.5, label="Overfitting Zone (Depth >= 19)")

    # Plot Entropy curves
    ent = df_plot[df_plot["criterion"] == "entropy"]
    ax1.plot(ent["max_depth_param"], ent["train_accuracy"], "o-", color="#d95f02", label="Entropy - Train Acc", linewidth=2, markersize=5)
    ax1.plot(ent["max_depth_param"], ent["test_accuracy"], "s--", color="#d95f02", label="Entropy - Test Acc", linewidth=2, markersize=5)

    # Plot Gini curves
    gini = df_plot[df_plot["criterion"] == "gini"]
    ax1.plot(gini["max_depth_param"], gini["train_accuracy"], "o-", color="#2b5c8f", label="Gini - Train Acc", linewidth=1.8, markersize=4, alpha=0.7)
    ax1.plot(gini["max_depth_param"], gini["test_accuracy"], "s--", color="#2b5c8f", label="Gini - Test Acc", linewidth=1.8, markersize=4, alpha=0.7)

    ax1.set_title("A. Train vs. Test Accuracy across Max Depth")
    ax1.set_xlabel("Max Depth Hyperparameter")
    ax1.set_ylabel("Accuracy Score")
    ax1.set_ylim(0.1, 1.02)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    # Panel B: Macro-F1 Curves
    ax2 = axes[0, 1]
    ax2.axvspan(1, 8, color="#fee0d2", alpha=0.5)
    ax2.axvspan(8, 18, color="#e5f5e0", alpha=0.5)
    ax2.axvspan(18, 35, color="#deebf7", alpha=0.5)

    ax2.plot(ent["max_depth_param"], ent["train_macro_f1"], "o-", color="#d95f02", label="Entropy - Train F1", linewidth=2, markersize=5)
    ax2.plot(ent["max_depth_param"], ent["test_macro_f1"], "s--", color="#d95f02", label="Entropy - Test F1", linewidth=2, markersize=5)
    ax2.plot(gini["max_depth_param"], gini["train_macro_f1"], "o-", color="#2b5c8f", label="Gini - Train F1", linewidth=1.8, markersize=4, alpha=0.7)
    ax2.plot(gini["max_depth_param"], gini["test_macro_f1"], "s--", color="#2b5c8f", label="Gini - Test F1", linewidth=1.8, markersize=4, alpha=0.7)

    ax2.set_title("B. Macro-F1 Score Trajectory")
    ax2.set_xlabel("Max Depth Hyperparameter")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_ylim(0.1, 1.02)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True)

    # Panel C: Generalization Gap (Train Acc - Test Acc)
    ax3 = axes[1, 0]
    ax3.axvspan(1, 8, color="#fee0d2", alpha=0.5)
    ax3.axvspan(8, 18, color="#e5f5e0", alpha=0.5)
    ax3.axvspan(18, 35, color="#deebf7", alpha=0.5)

    ax3.plot(ent["max_depth_param"], ent["generalization_gap_acc"], "o-", color="#d95f02", label="Entropy Generalization Gap", linewidth=2.2, markersize=5)
    ax3.plot(gini["max_depth_param"], gini["generalization_gap_acc"], "s-", color="#2b5c8f", label="Gini Generalization Gap", linewidth=2.2, markersize=5)

    ax3.set_title("C. Generalization Gap (Train Acc - Test Acc)")
    ax3.set_xlabel("Max Depth Hyperparameter")
    ax3.set_ylabel("Generalization Gap")
    ax3.legend(loc="upper left", fontsize=9, frameon=True)

    # Panel D: Leaf Count Growth
    ax4 = axes[1, 1]
    ax4.plot(ent["max_depth_param"], ent["leaf_count"], "o-", color="#d95f02", label="Entropy Leaf Count", linewidth=2.2, markersize=5)
    ax4.plot(gini["max_depth_param"], gini["leaf_count"], "s-", color="#2b5c8f", label="Gini Leaf Count", linewidth=2.2, markersize=5)

    ax4.set_title("D. Leaf Count Growth vs. Max Depth")
    ax4.set_xlabel("Max Depth Hyperparameter")
    ax4.set_ylabel("Number of Terminal Leaves")
    ax4.legend(loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_generalization_gap_detail(df_sweep: pd.DataFrame, output_path: Path) -> None:
    """Plot detailed Generalization Gap vs Leaf Count & Depth."""
    set_plotting_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df_sweep[df_sweep["max_depth_param"] != 999]

    sns.scatterplot(
        data=df_plot,
        x="leaf_count",
        y="test_accuracy",
        hue="criterion",
        size="generalization_gap_acc",
        sizes=(30, 200),
        palette={"entropy": "#d95f02", "gini": "#2b5c8f"},
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Test Accuracy vs. Model Complexity (Leaf Count) & Generalization Gap")
    ax.set_xlabel("Leaf Count (Model Complexity)")
    ax.set_ylabel("Test Accuracy")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_complexity_tradeoff(df_sweep: pd.DataFrame, output_path: Path) -> None:
    """Plot Pareto curve of Test Accuracy vs Model Depth & Leaf Count."""
    set_plotting_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df_sweep[df_sweep["max_depth_param"] != 999]

    for criterion, color in [("entropy", "#d95f02"), ("gini", "#2b5c8f")]:
        sub = df_plot[df_plot["criterion"] == criterion]
        ax.plot(sub["actual_depth"], sub["test_accuracy"], "o-", label=f"{criterion.capitalize()} Criterion", color=color, linewidth=2, markersize=6)

        # Annotate peak point
        max_row = sub.loc[sub["test_accuracy"].idxmax()]
        ax.annotate(
            f"Peak {criterion}: {max_row['test_accuracy']:.4f}\n(Depth={int(max_row['actual_depth'])})",
            (max_row["actual_depth"], max_row["test_accuracy"]),
            xytext=(max_row["actual_depth"] - 4, max_row["test_accuracy"] - 0.05),
            arrowprops=dict(facecolor=color, shrink=0.08, width=1.5, headwidth=6),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5),
        )

    ax.set_title("Pareto Curve: Test Accuracy vs. Actual Tree Depth")
    ax.set_xlabel("Actual Tree Depth")
    ax.set_ylabel("Test Accuracy")
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    ensure_project_dirs()
    print("Executing Card [10][TV2] Depth sweep — underfitting/overfitting curves...")

    X_train, X_test, y_train, y_test = load_letter_recognition()
    print(f"Dataset loaded: Train={X_train.shape}, Test={X_test.shape}")

    depth_values: list[int | None] = list(range(1, 36)) + [None]

    df_sweep = run_depth_sweep(X_train, X_test, y_train, y_test, depth_values)

    summary_csv = RESULTS_DIR / "dt_letter_depth_sweep__summary.csv"
    df_sweep.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")

    # Generate Figures
    fig1_path = FIGURES_DIR / "dt_depth_sweep__performance_curves.png"
    fig2_path = FIGURES_DIR / "dt_depth_sweep__generalization_gap.png"
    fig3_path = FIGURES_DIR / "dt_depth_sweep__complexity_tradeoff.png"

    plot_performance_curves(df_sweep, fig1_path)
    plot_generalization_gap_detail(df_sweep, fig2_path)
    plot_complexity_tradeoff(df_sweep, fig3_path)

    # Key statistics for result JSON
    ent_sweep = df_sweep[df_sweep["criterion"] == "entropy"]
    gini_sweep = df_sweep[df_sweep["criterion"] == "gini"]

    ent_best = ent_sweep.loc[ent_sweep["test_accuracy"].idxmax()]
    gini_best = gini_sweep.loc[gini_sweep["test_accuracy"].idxmax()]

    ent_full = ent_sweep[ent_sweep["max_depth_param"] == 999].iloc[0]
    gini_full = gini_sweep[gini_sweep["max_depth_param"] == 999].iloc[0]

    metrics = {
        "entropy_optimal_depth": float(ent_best["actual_depth"]),
        "entropy_optimal_test_accuracy": float(ent_best["test_accuracy"]),
        "entropy_optimal_test_f1": float(ent_best["test_macro_f1"]),
        "entropy_optimal_leaves": float(ent_best["leaf_count"]),
        "gini_optimal_depth": float(gini_best["actual_depth"]),
        "gini_optimal_test_accuracy": float(gini_best["test_accuracy"]),
        "gini_optimal_test_f1": float(gini_best["test_macro_f1"]),
        "gini_optimal_leaves": float(gini_best["leaf_count"]),
        "unpruned_entropy_train_accuracy": float(ent_full["train_accuracy"]),
        "unpruned_entropy_test_accuracy": float(ent_full["test_accuracy"]),
        "unpruned_entropy_generalization_gap": float(ent_full["generalization_gap_acc"]),
        "unpruned_gini_train_accuracy": float(gini_full["train_accuracy"]),
        "unpruned_gini_test_accuracy": float(gini_full["test_accuracy"]),
        "unpruned_gini_generalization_gap": float(gini_full["generalization_gap_acc"]),
        "underfitting_threshold_max_depth": 8.0,
        "optimal_max_depth_lower_bound": 9.0,
        "optimal_max_depth_upper_bound": 18.0,
        "overfitting_threshold_max_depth": 19.0,
    }

    relative_figs = [
        str(fig1_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(fig2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(fig3_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    ]

    json_result = build_result(
        experiment_id="dt_letter_depth_sweep",
        dataset="letter_recognition",
        model="DecisionTreeClassifier(max_depth_sweep)",
        metrics=metrics,
        figure_paths=relative_figs,
        notes=(
            "Card [10][TV2] Depth sweep experiment from max_depth=1 to 35 and unconstrained. "
            "Maps Underfitting (depth 1-8), Optimal (depth 9-18), and Overfitting (depth >=19) regions."
        ),
    )

    json_path = save_result(json_result)
    print(f"Saved result JSON to: {json_path}")

    # Display key summary table
    print("\n" + "=" * 90)
    print(" DEPTH SWEEP KEY MILESTONES & REGION BOUNDARIES ")
    print("=" * 90)
    display_rows = df_sweep[
        (df_sweep["max_depth_param"].isin([2, 5, 8, 12, 15, 18, 25, 35, 999]))
        & (df_sweep["criterion"] == "entropy")
    ]
    cols = [
        "max_depth_label",
        "region",
        "train_accuracy",
        "test_accuracy",
        "error_rate",
        "test_macro_f1",
        "generalization_gap_acc",
        "leaf_count",
    ]
    print(display_rows[cols].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
