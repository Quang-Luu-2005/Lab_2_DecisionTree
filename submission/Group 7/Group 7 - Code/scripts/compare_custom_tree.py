"""Compare the from-scratch tree with sklearn on the project's Digits dataset.

Run from the repository root:

    python scripts/compare_custom_tree.py

The two estimators receive the exact same train/test split and tree parameters.
The script writes a result contract, a comparison CSV, and the custom tree text
under ``results/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from decision_tree_lab2.config import RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from decision_tree_lab2.custom_tree import CustomDecisionTreeClassifier
from decision_tree_lab2.results import build_result, save_result
from decision_tree_lab2.split import split_train_test


EXPERIMENT_ID = "custom_vs_sklearn_decision_tree"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum tree depth used by both estimators (default: 8).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples per leaf used by both estimators (default: 1).",
    )
    return parser.parse_args()


def evaluate(name: str, model: object, X_train, X_test, y_train, y_test) -> dict[str, object]:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    prediction_seconds = time.perf_counter() - started
    return {
        "model": name,
        "train_accuracy": float(accuracy_score(y_train, train_prediction)),
        "test_accuracy": float(accuracy_score(y_test, test_prediction)),
        "error_rate": float(1.0 - accuracy_score(y_test, test_prediction)),
        "f1_macro": float(f1_score(y_test, test_prediction, average="macro")),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "tree_depth": int(model.get_depth()),
        "leaf_count": int(model.get_n_leaves()),
        "predictions": test_prediction,
    }


def main() -> None:
    args = parse_args()
    if args.max_depth < 0 or args.min_samples_leaf < 1:
        raise SystemExit("--max-depth must be >= 0 and --min-samples-leaf must be >= 1")

    digits = load_digits()
    X_train, X_test, y_train, y_test = split_train_test(digits.data, digits.target)
    model_parameters = {
        "max_depth": args.max_depth,
        "min_samples_split": 2,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": RANDOM_STATE,
    }
    custom = CustomDecisionTreeClassifier(**model_parameters)
    sklearn_model = DecisionTreeClassifier(**model_parameters)
    custom_result = evaluate(
        "CustomDecisionTreeClassifier", custom, X_train, X_test, y_train, y_test
    )
    sklearn_result = evaluate(
        "sklearn DecisionTreeClassifier", sklearn_model, X_train, X_test, y_train, y_test
    )
    custom_predictions = custom_result.pop("predictions")
    sklearn_predictions = sklearn_result.pop("predictions")

    comparison = pd.DataFrame([custom_result, sklearn_result])
    comparison["prediction_agreement_vs_sklearn"] = np.nan
    comparison.loc[
        comparison["model"] == "CustomDecisionTreeClassifier",
        "prediction_agreement_vs_sklearn",
    ] = float(np.mean(custom_predictions == sklearn_predictions))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = RESULTS_DIR / f"{EXPERIMENT_ID}__comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    tree_path = RESULTS_DIR / f"{EXPERIMENT_ID}__custom_tree.txt"
    tree_path.write_text(
        custom.export_text([f"pixel_{i:02d}" for i in range(X_train.shape[1])]),
        encoding="utf-8",
    )

    custom_test_accuracy = float(custom_result["test_accuracy"])
    sklearn_test_accuracy = float(sklearn_result["test_accuracy"])
    agreement = float(np.mean(custom_predictions == sklearn_predictions))
    result = build_result(
        experiment_id=EXPERIMENT_ID,
        dataset="handwritten_digits",
        model="CustomDecisionTreeClassifier vs sklearn DecisionTreeClassifier",
        metrics={
            "custom_test_accuracy": custom_test_accuracy,
            "sklearn_test_accuracy": sklearn_test_accuracy,
            "absolute_accuracy_difference": abs(custom_test_accuracy - sklearn_test_accuracy),
            "test_prediction_agreement": agreement,
            "custom_error_rate": float(custom_result["error_rate"]),
            "sklearn_error_rate": float(sklearn_result["error_rate"]),
        },
        notes=(
            "Both models use the same stratified 80/20 split, random_state=42, Gini criterion, "
            f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}. The custom "
            "implementation exhaustively scans distinct numeric thresholds and does not use "
            "sklearn "
            "internals; small differences can arise from tie-breaking."
        ),
    )
    result["dataset_metadata"] = {
        "samples": int(digits.data.shape[0]),
        "features": int(digits.data.shape[1]),
        "classes": int(len(digits.target_names)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    result["model_parameters"] = model_parameters
    result["models"] = [custom_result, sklearn_result]
    result["comparison_artifact"] = str(comparison_path.relative_to(PROJECT_ROOT)).replace(
        "\\", "/"
    )
    result["custom_tree_artifact"] = str(tree_path.relative_to(PROJECT_ROOT)).replace(
        "\\", "/"
    )
    save_result(result, RESULTS_DIR / f"{EXPERIMENT_ID}.json")

    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Prediction agreement: {agreement:.6%}")
    print(f"Wrote: {comparison_path}")
    print(f"Wrote: {tree_path}")


if __name__ == "__main__":
    main()
