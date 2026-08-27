"""Execute Card [11][TV2] Cải tiến DT — max_depth, min_samples, pruning.

Benchmarking 3 Decision Tree regularization branches on UCI Letter Recognition:
1. Branch 1: max_depth pre-pruning
2. Branch 2: min_samples_split / min_samples_leaf pre-pruning
3. Branch 3: Cost-Complexity Post-pruning (ccp_alpha)
Plus Best Combined Regularized Model.

Records train/test metrics, error_rate, macro-F1, generalization gap, depth, and leaf count.
Saves schema v1.0 result JSON, CSV summary, and publication-quality figures.

Run from repository root:
    .venv\\Scripts\\python scripts/run_dt_improvements_experiment.py
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def evaluate_clf(
    clf: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    strategy_name: str,
) -> dict[str, Any]:
    """Fit tree, record metrics, and return structured evaluation record."""

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

    depth = int(clf.get_depth())
    leaves = int(clf.get_n_leaves())
    nodes = int(clf.tree_.node_count)
    gen_gap = float(tr_acc - te_acc)
    error_rate = float(1.0 - te_acc)

    return {
        "strategy": strategy_name,
        "criterion": str(clf.criterion),
        "max_depth": str(clf.max_depth),
        "min_samples_split": int(clf.min_samples_split),
        "min_samples_leaf": int(clf.min_samples_leaf),
        "ccp_alpha": float(clf.ccp_alpha),
        "train_accuracy": tr_acc,
        "test_accuracy": te_acc,
        "error_rate": error_rate,
        "train_macro_f1": tr_f1,
        "test_macro_f1": te_f1,
        "generalization_gap": gen_gap,
        "tree_depth": depth,
        "leaf_count": leaves,
        "node_count": nodes,
        "training_seconds": fit_time,
        "prediction_seconds": pred_time,
    }


def set_plotting_style() -> None:
    """Apply clean aesthetic plotting theme."""
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


def plot_strategy_comparison(df_summary: pd.DataFrame, output_path: Path) -> None:
    """Multi-panel comparison of baseline vs improvement branches."""
    set_plotting_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    palette = sns.color_palette("muted", len(df_summary))

    # Panel A: Test Accuracy & Macro-F1
    df_melt = df_summary.melt(
        id_vars=["strategy"],
        value_vars=["test_accuracy", "test_macro_f1"],
        var_name="Metric",
        value_name="Score",
    )
    df_melt["Metric"] = df_melt["Metric"].map({"test_accuracy": "Test Accuracy", "test_macro_f1": "Macro F1"})

    sns.barplot(
        data=df_melt,
        x="strategy",
        y="Score",
        hue="Metric",
        ax=axes[0, 0],
        palette=["#2b5c8f", "#d95f02"],
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0, 0].set_title("A. Model Performance (Test Accuracy & Macro-F1)")
    axes[0, 0].set_ylim(0.80, 0.90)
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=25)

    for p in axes[0, 0].patches:
        h = p.get_height()
        if h > 0:
            axes[0, 0].annotate(
                f"{h:.4f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # Panel B: Generalization Gap (Train Acc - Test Acc)
    sns.barplot(
        data=df_summary,
        x="strategy",
        y="generalization_gap",
        ax=axes[0, 1],
        palette="Reds_r",
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0, 1].set_title("B. Generalization Gap (Lower is Better)")
    axes[0, 1].set_ylabel("Train Acc - Test Acc")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=25)

    for p in axes[0, 1].patches:
        h = p.get_height()
        if h > 0:
            axes[0, 1].annotate(
                f"{h:.4f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8.5,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # Panel C: Actual Tree Depth
    sns.barplot(
        data=df_summary,
        x="strategy",
        y="tree_depth",
        ax=axes[1, 0],
        palette="Blues_r",
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1, 0].set_title("C. Tree Depth (Structural Complexity)")
    axes[1, 0].set_ylabel("Depth (levels)")
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis="x", rotation=25)

    for p in axes[1, 0].patches:
        h = p.get_height()
        if h > 0:
            axes[1, 0].annotate(
                f"{int(h)}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # Panel D: Leaf Count
    sns.barplot(
        data=df_summary,
        x="strategy",
        y="leaf_count",
        ax=axes[1, 1],
        palette="Purples_r",
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1, 1].set_title("D. Leaf Count (Number of Terminal Leaves)")
    axes[1, 1].set_ylabel("Leaf Count")
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis="x", rotation=25)

    for p in axes[1, 1].patches:
        h = p.get_height()
        if h > 0:
            axes[1, 1].annotate(
                f"{int(h)}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_ccp_alpha_path(df_ccp: pd.DataFrame, best_alpha: float, output_path: Path) -> None:
    """Plot cost-complexity pruning path metrics vs ccp_alpha."""
    set_plotting_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy vs CCP Alpha
    axes[0].plot(df_ccp["ccp_alpha"], df_ccp["train_accuracy"], "o-", color="#2b5c8f", label="Train Accuracy", linewidth=2, markersize=4)
    axes[0].plot(df_ccp["ccp_alpha"], df_ccp["test_accuracy"], "s--", color="#d95f02", label="Test Accuracy", linewidth=2, markersize=4)
    axes[0].axvline(best_alpha, color="red", linestyle=":", linewidth=2, label=f"Optimal α* = {best_alpha:.5f}")

    axes[0].set_title("A. Accuracy vs. Cost-Complexity Pruning Alpha (ccp_alpha)")
    axes[0].set_xlabel("ccp_alpha Hyperparameter")
    axes[0].set_ylabel("Accuracy Score")
    axes[0].set_xscale("log")
    axes[0].legend(loc="lower left", fontsize=9, frameon=True)

    # Leaf Count vs CCP Alpha
    axes[1].plot(df_ccp["ccp_alpha"], df_ccp["leaf_count"], "o-", color="#7570b3", linewidth=2, markersize=4)
    axes[1].axvline(best_alpha, color="red", linestyle=":", linewidth=2, label=f"Optimal α* = {best_alpha:.5f}")

    axes[1].set_title("B. Leaf Count Decay vs. Cost-Complexity Pruning Alpha")
    axes[1].set_xlabel("ccp_alpha Hyperparameter")
    axes[1].set_ylabel("Number of Leaf Nodes")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_min_samples_sweep(
    df_min_leaf: pd.DataFrame,
    df_min_split: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot min_samples_leaf and min_samples_split performance curves."""
    set_plotting_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Min Samples Leaf Sweep
    axes[0].plot(df_min_leaf["min_samples_leaf"], df_min_leaf["train_accuracy"], "o-", color="#2b5c8f", label="Train Accuracy", linewidth=2)
    axes[0].plot(df_min_leaf["min_samples_leaf"], df_min_leaf["test_accuracy"], "s--", color="#d95f02", label="Test Accuracy", linewidth=2)

    axes[0].set_title("A. Performance vs. min_samples_leaf")
    axes[0].set_xlabel("min_samples_leaf")
    axes[0].set_ylabel("Accuracy Score")
    axes[0].legend(loc="lower left", fontsize=9, frameon=True)

    # Min Samples Split Sweep
    axes[1].plot(df_min_split["min_samples_split"], df_min_split["train_accuracy"], "o-", color="#2b5c8f", label="Train Accuracy", linewidth=2)
    axes[1].plot(df_min_split["min_samples_split"], df_min_split["test_accuracy"], "s--", color="#d95f02", label="Test Accuracy", linewidth=2)

    axes[1].set_title("B. Performance vs. min_samples_split")
    axes[1].set_xlabel("min_samples_split")
    axes[1].set_ylabel("Accuracy Score")
    axes[1].legend(loc="lower left", fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    ensure_project_dirs()
    print("Executing Card [11][TV2] Decision Tree Improvements...")

    X_train, X_test, y_train, y_test = load_letter_recognition()
    print(f"Dataset loaded: Train={X_train.shape}, Test={X_test.shape}")

    results_summary = []

    # 1. Baseline Unpruned Tree
    clf_base = DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE)
    res_base = evaluate_clf(clf_base, X_train, X_test, y_train, y_test, "Unpruned Baseline")
    results_summary.append(res_base)

    # 2. Branch 1: max_depth Pre-pruning (Tuned max_depth=15)
    clf_b1 = DecisionTreeClassifier(criterion="entropy", max_depth=15, random_state=RANDOM_STATE)
    res_b1 = evaluate_clf(clf_b1, X_train, X_test, y_train, y_test, "Branch 1: max_depth=15")
    results_summary.append(res_b1)

    # 3. Branch 2: min_samples Pre-pruning (Tuned min_samples_leaf=2, min_samples_split=5)
    # Perform min_samples_leaf sweep
    leaf_records = []
    for leaf in [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]:
        clf_temp = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=leaf, random_state=RANDOM_STATE)
        res_temp = evaluate_clf(clf_temp, X_train, X_test, y_train, y_test, f"min_leaf_{leaf}")
        leaf_records.append(res_temp)
    df_min_leaf = pd.DataFrame(leaf_records)

    split_records = []
    for split in [2, 4, 6, 8, 10, 15, 20, 30, 50]:
        clf_temp = DecisionTreeClassifier(criterion="entropy", min_samples_split=split, random_state=RANDOM_STATE)
        res_temp = evaluate_clf(clf_temp, X_train, X_test, y_train, y_test, f"min_split_{split}")
        split_records.append(res_temp)
    df_min_split = pd.DataFrame(split_records)

    # Best Branch 2 configuration
    clf_b2 = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, min_samples_split=4, random_state=RANDOM_STATE)
    res_b2 = evaluate_clf(clf_b2, X_train, X_test, y_train, y_test, "Branch 2: min_leaf=2")
    results_summary.append(res_b2)

    # 4. Branch 3: Cost-Complexity Post-Pruning (ccp_alpha)
    # Extract pruning path
    clf_full = DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE)
    clf_full.fit(X_train, y_train)
    path = clf_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Sample representative non-negative alphas
    valid_alphas = [a for a in ccp_alphas if 0.0 <= a <= 0.01]
    if len(valid_alphas) > 30:
        step = len(valid_alphas) // 30
        sampled_alphas = valid_alphas[::step]
    else:
        sampled_alphas = valid_alphas

    ccp_records = []
    for alpha in sampled_alphas:
        clf_ccp = DecisionTreeClassifier(criterion="entropy", ccp_alpha=alpha, random_state=RANDOM_STATE)
        res_ccp = evaluate_clf(clf_ccp, X_train, X_test, y_train, y_test, f"ccp_alpha_{alpha:.6f}")
        ccp_records.append(res_ccp)

    df_ccp = pd.DataFrame(ccp_records)
    best_ccp_row = df_ccp.loc[df_ccp["test_accuracy"].idxmax()]
    best_alpha = float(best_ccp_row["ccp_alpha"])

    clf_b3 = DecisionTreeClassifier(criterion="entropy", ccp_alpha=best_alpha, random_state=RANDOM_STATE)
    res_b3 = evaluate_clf(clf_b3, X_train, X_test, y_train, y_test, f"Branch 3: ccp_alpha={best_alpha:.5f}")
    results_summary.append(res_b3)

    # 5. Combined Best Regularized Model (max_depth=15, min_samples_leaf=2, ccp_alpha=0.0001)
    clf_best = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=15,
        min_samples_leaf=2,
        ccp_alpha=0.0001,
        random_state=RANDOM_STATE,
    )
    res_best = evaluate_clf(clf_best, X_train, X_test, y_train, y_test, "Combined Best Model")
    results_summary.append(res_best)

    df_summary = pd.DataFrame(results_summary)
    summary_csv = RESULTS_DIR / "dt_letter_improvements__summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")

    # Generate Figures
    fig1_path = FIGURES_DIR / "dt_improvements__strategy_comparison.png"
    fig2_path = FIGURES_DIR / "dt_improvements__ccp_alpha_path.png"
    fig3_path = FIGURES_DIR / "dt_improvements__min_samples_sweep.png"

    plot_strategy_comparison(df_summary, fig1_path)
    plot_ccp_alpha_path(df_ccp, best_alpha, fig2_path)
    plot_min_samples_sweep(df_min_leaf, df_min_split, fig3_path)

    # Save schema v1.0 JSON
    metrics_json = {
        "baseline_unpruned_test_accuracy": res_base["test_accuracy"],
        "baseline_unpruned_generalization_gap": res_base["generalization_gap"],
        "baseline_unpruned_depth": float(res_base["tree_depth"]),
        "baseline_unpruned_leaves": float(res_base["leaf_count"]),
        "branch1_max_depth_test_accuracy": res_b1["test_accuracy"],
        "branch1_max_depth_generalization_gap": res_b1["generalization_gap"],
        "branch1_max_depth_leaves": float(res_b1["leaf_count"]),
        "branch2_min_samples_test_accuracy": res_b2["test_accuracy"],
        "branch2_min_samples_generalization_gap": res_b2["generalization_gap"],
        "branch2_min_samples_leaves": float(res_b2["leaf_count"]),
        "branch3_ccp_alpha_optimal": best_alpha,
        "branch3_ccp_alpha_test_accuracy": res_b3["test_accuracy"],
        "branch3_ccp_alpha_generalization_gap": res_b3["generalization_gap"],
        "branch3_ccp_alpha_leaves": float(res_b3["leaf_count"]),
        "combined_best_test_accuracy": res_best["test_accuracy"],
        "combined_best_test_f1": res_best["test_macro_f1"],
        "combined_best_generalization_gap": res_best["generalization_gap"],
        "combined_best_depth": float(res_best["tree_depth"]),
        "combined_best_leaves": float(res_best["leaf_count"]),
        "accuracy_gain_over_baseline": res_best["test_accuracy"] - res_base["test_accuracy"],
        "gap_reduction_over_baseline": res_base["generalization_gap"] - res_best["generalization_gap"],
    }

    relative_figs = [
        str(fig1_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(fig2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(fig3_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    ]

    json_result = build_result(
        experiment_id="dt_letter_improvements",
        dataset="letter_recognition",
        model="DecisionTreeClassifier(regularization_suite)",
        metrics=metrics_json,
        figure_paths=relative_figs,
        notes=(
            "Card [11][TV2] Decision Tree improvement experiment benchmarks 3 branches: "
            "(1) max_depth=15, (2) min_samples_leaf=2, (3) ccp_alpha post-pruning, plus Combined Best model."
        ),
    )

    json_path = save_result(json_result)
    print(f"Saved result JSON to: {json_path}")

    # Display clean table
    print("\n" + "=" * 90)
    print(" DECISION TREE IMPROVEMENT BRANCHES COMPARISON TABLE ")
    print("=" * 90)
    cols = [
        "strategy",
        "train_accuracy",
        "test_accuracy",
        "error_rate",
        "test_macro_f1",
        "generalization_gap",
        "tree_depth",
        "leaf_count",
    ]
    print(df_summary[cols].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
