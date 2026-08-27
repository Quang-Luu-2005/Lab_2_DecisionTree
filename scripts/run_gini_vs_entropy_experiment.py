"""Execute Card [09][TV2] Experiment — Gini vs Entropy.

Compares DecisionTreeClassifier with criterion='gini' vs criterion='entropy'
on the exact same split and configurations for Letter Recognition (and cross-datasets).
Records accuracy, error rate, macro-F1, tree depth, leaf count, timing, and feature importances.
Saves schema v1.0 result JSON and publication-quality figures.

Run from repository root:
    .venv\\Scripts\\python scripts/run_gini_vs_entropy_experiment.py
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from decision_tree_lab2.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
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


def evaluate_model(
    clf: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Train DecisionTreeClassifier and extract precise evaluation metrics."""

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    pred_time = time.perf_counter() - t1

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    error_rate = float(1.0 - test_acc)

    prec_macro = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))

    depth = int(clf.get_depth())
    leaves = int(clf.get_n_leaves())
    node_count = int(clf.tree_.node_count)

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "error_rate": error_rate,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_mac,
        "training_seconds": fit_time,
        "prediction_seconds": pred_time,
        "tree_depth": depth,
        "leaf_count": leaves,
        "node_count": node_count,
        "y_test_pred": y_test_pred,
        "feature_importances": clf.feature_importances_,
    }


def set_plotting_style() -> None:
    """Apply clean, modern plotting theme."""
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


def plot_metrics_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create a multi-panel visual comparison between Gini and Entropy."""
    set_plotting_style()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    palette = {"gini": "#2b5c8f", "entropy": "#d95f02"}

    # Panel 1: Accuracy and Macro-F1
    df_perf = summary_df.melt(
        id_vars=["config", "criterion"],
        value_vars=["test_accuracy", "f1_macro"],
        var_name="Metric",
        value_name="Score",
    )
    df_perf["Metric"] = df_perf["Metric"].map({"test_accuracy": "Test Accuracy", "f1_macro": "Macro F1"})
    sns.barplot(
        data=df_perf,
        x="config",
        y="Score",
        hue="criterion",
        ax=axes[0, 0],
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0, 0].set_title("A. Predictive Performance (Accuracy & Macro-F1)")
    axes[0, 0].set_ylim(0.70, 1.0)
    axes[0, 0].set_xlabel("Configuration")
    axes[0, 0].set_ylabel("Score")

    # Add numeric annotations
    for p in axes[0, 0].patches:
        height = p.get_height()
        if height > 0:
            axes[0, 0].annotate(
                f"{height:.4f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # Panel 2: Error Rate Comparison
    sns.barplot(
        data=summary_df,
        x="config",
        y="error_rate",
        hue="criterion",
        ax=axes[0, 1],
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0, 1].set_title("B. Classification Error Rate (Lower is Better)")
    axes[0, 1].set_xlabel("Configuration")
    axes[0, 1].set_ylabel("Error Rate")

    for p in axes[0, 1].patches:
        height = p.get_height()
        if height > 0:
            axes[0, 1].annotate(
                f"{height:.4f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # Panel 3: Tree Depth
    sns.barplot(
        data=summary_df,
        x="config",
        y="tree_depth",
        hue="criterion",
        ax=axes[1, 0],
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1, 0].set_title("C. Maximum Tree Depth")
    axes[1, 0].set_xlabel("Configuration")
    axes[1, 0].set_ylabel("Tree Depth (levels)")

    for p in axes[1, 0].patches:
        height = p.get_height()
        if height > 0:
            axes[1, 0].annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # Panel 4: Number of Leaf Nodes
    sns.barplot(
        data=summary_df,
        x="config",
        y="leaf_count",
        hue="criterion",
        ax=axes[1, 1],
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1, 1].set_title("D. Number of Terminal Leaves")
    axes[1, 1].set_xlabel("Configuration")
    axes[1, 1].set_ylabel("Leaf Count")

    for p in axes[1, 1].patches:
        height = p.get_height()
        if height > 0:
            axes[1, 1].annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
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


def plot_depth_trajectory(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    """Evaluate Gini vs Entropy across a range of max_depth values."""

    depth_range = list(range(2, 36, 2)) + [None]
    records = []

    for depth in depth_range:
        depth_label = depth if depth is not None else 999
        for criterion in ["gini", "entropy"]:
            clf = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=depth,
                random_state=RANDOM_STATE,
            )
            clf.fit(X_train, y_train)
            tr_acc = accuracy_score(y_train, clf.predict(X_train))
            te_acc = accuracy_score(y_test, clf.predict(X_test))
            f1 = f1_score(y_test, clf.predict(X_test), average="macro", zero_division=0)
            actual_depth = clf.get_depth()
            leaves = clf.get_n_leaves()

            records.append(
                {
                    "max_depth_param": depth_label,
                    "actual_depth": actual_depth,
                    "criterion": criterion,
                    "train_accuracy": tr_acc,
                    "test_accuracy": te_acc,
                    "error_rate": 1.0 - te_acc,
                    "f1_macro": f1,
                    "leaf_count": leaves,
                }
            )

    df_depth = pd.DataFrame(records)

    set_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Accuracy vs Max Depth
    df_plot = df_depth[df_depth["max_depth_param"] != 999]
    sns.lineplot(
        data=df_plot,
        x="max_depth_param",
        y="test_accuracy",
        hue="criterion",
        style="criterion",
        markers=True,
        dashes=False,
        ax=axes[0],
        palette={"gini": "#2b5c8f", "entropy": "#d95f02"},
        linewidth=2.2,
        markersize=7,
    )
    axes[0].set_title("A. Test Accuracy vs. Maximum Depth Cap")
    axes[0].set_xlabel("Max Depth Hyperparameter")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_ylim(0.2, 0.9)

    # Plot Leaf Count vs Depth
    sns.lineplot(
        data=df_plot,
        x="max_depth_param",
        y="leaf_count",
        hue="criterion",
        style="criterion",
        markers=True,
        dashes=False,
        ax=axes[1],
        palette={"gini": "#2b5c8f", "entropy": "#d95f02"},
        linewidth=2.2,
        markersize=7,
    )
    axes[1].set_title("B. Leaf Count Expansion vs. Maximum Depth Cap")
    axes[1].set_xlabel("Max Depth Hyperparameter")
    axes[1].set_ylabel("Number of Leaf Nodes")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    return df_depth


def plot_feature_importance_comparison(
    imp_gini: np.ndarray,
    imp_entropy: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Plot feature importance ranking comparison between Gini and Entropy."""
    set_plotting_style()

    df_imp = pd.DataFrame(
        {
            "Feature": feature_names,
            "Gini Impurity": imp_gini,
            "Entropy (Gain)": imp_entropy,
        }
    )
    df_imp["Diff (Entropy - Gini)"] = df_imp["Entropy (Gain)"] - df_imp["Gini Impurity"]
    df_imp = df_imp.sort_values(by="Gini Impurity", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    df_melt = df_imp.melt(
        id_vars=["Feature"],
        value_vars=["Gini Impurity", "Entropy (Gain)"],
        var_name="Criterion",
        value_name="Importance",
    )

    sns.barplot(
        data=df_melt,
        x="Feature",
        y="Importance",
        hue="Criterion",
        ax=ax,
        palette={"Gini Impurity": "#2b5c8f", "Entropy (Gain)": "#d95f02"},
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_title("Feature Importance Profile: Gini Impurity vs Entropy (Information Gain)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Normalized Feature Importance")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    ensure_project_dirs()
    print("Executing Card [09][TV2] Experiment — Gini vs Entropy...")

    # Load shared split dataset
    X_train, X_test, y_train, y_test = load_letter_recognition()
    print(f"Dataset loaded: Train={X_train.shape}, Test={X_test.shape}")

    # Configurations to benchmark
    configs = [
        {"name": "Full Depth (Unpruned)", "max_depth": None, "min_samples_leaf": 1},
        {"name": "Regularized (depth=15)", "max_depth": 15, "min_samples_leaf": 5},
        {"name": "Pruned (depth=10)", "max_depth": 10, "min_samples_leaf": 10},
    ]

    results_list = []
    eval_cache = {}

    for cfg in configs:
        for criterion in ["gini", "entropy"]:
            model_id = f"dt_letter_{criterion}_{cfg['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}"
            clf = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                random_state=RANDOM_STATE,
            )
            res = evaluate_model(clf, X_train, X_test, y_train, y_test)
            eval_cache[(criterion, cfg["name"])] = res

            row = {
                "experiment_id": model_id,
                "config": cfg["name"],
                "criterion": criterion,
                "max_depth_param": str(cfg["max_depth"]),
                "min_samples_leaf": cfg["min_samples_leaf"],
                "train_accuracy": res["train_accuracy"],
                "test_accuracy": res["test_accuracy"],
                "error_rate": res["error_rate"],
                "precision_macro": res["precision_macro"],
                "recall_macro": res["recall_macro"],
                "f1_macro": res["f1_macro"],
                "tree_depth": res["tree_depth"],
                "leaf_count": res["leaf_count"],
                "node_count": res["node_count"],
                "training_seconds": res["training_seconds"],
                "prediction_seconds": res["prediction_seconds"],
            }
            results_list.append(row)

    summary_df = pd.DataFrame(results_list)
    summary_csv_path = RESULTS_DIR / "dt_letter_gini_vs_entropy__summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary CSV saved to: {summary_csv_path}")

    # Generate Figures
    fig1_path = FIGURES_DIR / "dt_gini_vs_entropy__metrics.png"
    fig2_path = FIGURES_DIR / "dt_gini_vs_entropy__depth_trajectory.png"
    fig3_path = FIGURES_DIR / "dt_gini_vs_entropy__feature_importance.png"

    plot_metrics_comparison(summary_df, fig1_path)

    df_depth_traj = plot_depth_trajectory(X_train, X_test, y_train, y_test, fig2_path)

    full_gini_res = eval_cache[("gini", "Full Depth (Unpruned)")]
    full_entropy_res = eval_cache[("entropy", "Full Depth (Unpruned)")]
    plot_feature_importance_comparison(
        full_gini_res["feature_importances"],
        full_entropy_res["feature_importances"],
        list(LETTER_FEATURES),
        fig3_path,
    )

    # Save schema v1.0 JSON result for the main baseline comparison
    full_gini = eval_cache[("gini", "Full Depth (Unpruned)")]
    full_entropy = eval_cache[("entropy", "Full Depth (Unpruned)")]

    main_metrics = {
        "gini_train_accuracy": full_gini["train_accuracy"],
        "gini_test_accuracy": full_gini["test_accuracy"],
        "gini_error_rate": full_gini["error_rate"],
        "gini_f1_macro": full_gini["f1_macro"],
        "gini_tree_depth": full_gini["tree_depth"],
        "gini_leaf_count": full_gini["leaf_count"],
        "gini_training_seconds": full_gini["training_seconds"],
        "entropy_train_accuracy": full_entropy["train_accuracy"],
        "entropy_test_accuracy": full_entropy["test_accuracy"],
        "entropy_error_rate": full_entropy["error_rate"],
        "entropy_f1_macro": full_entropy["f1_macro"],
        "entropy_tree_depth": full_entropy["tree_depth"],
        "entropy_leaf_count": full_entropy["leaf_count"],
        "entropy_training_seconds": full_entropy["training_seconds"],
        "accuracy_difference_entropy_minus_gini": full_entropy["test_accuracy"] - full_gini["test_accuracy"],
        "f1_macro_difference_entropy_minus_gini": full_entropy["f1_macro"] - full_gini["f1_macro"],
        "depth_difference_entropy_minus_gini": float(full_entropy["tree_depth"] - full_gini["tree_depth"]),
        "leaf_count_difference_entropy_minus_gini": float(full_entropy["leaf_count"] - full_gini["leaf_count"]),
    }

    relative_fig_paths = [
        str(fig1_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(fig2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(fig3_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    ]

    json_result = build_result(
        experiment_id="dt_letter_gini_vs_entropy",
        dataset="letter_recognition",
        model="DecisionTreeClassifier(gini_vs_entropy)",
        metrics=main_metrics,
        figure_paths=relative_fig_paths,
        notes=(
            "Card [09][TV2] Gini vs Entropy experiment on Letter Recognition clean split. "
            "Evaluates full depth and regularized trees, comparing accuracy, error rate, macro-F1, depth, and leaves."
        ),
    )

    json_path = save_result(json_result)
    print(f"Saved result JSON to: {json_path}")

    # Display clean formatted summary table in console
    print("\n" + "=" * 80)
    print(" EXPERIMENT BENCHMARK SUMMARY TABLE: GINI vs ENTROPY ")
    print("=" * 80)
    display_cols = [
        "config",
        "criterion",
        "test_accuracy",
        "error_rate",
        "f1_macro",
        "tree_depth",
        "leaf_count",
        "training_seconds",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
