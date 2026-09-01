"""Validate the local data-to-metrics pipeline on all three project datasets.

The check performs the concrete path ``load -> validate/preprocess -> split ->
fit -> predict -> metrics`` for Letter Recognition, Handwritten Digits and
Covertype using the project's shared split protocol.  It intentionally runs a
small, CPU-only baseline rather than the optional GPU benchmark notebook.

Run from the repository root:

    python scripts/run_end_to_end_check.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from decision_tree_lab2.config import RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from decision_tree_lab2.split import split_train_test


def load_letter() -> dict[str, object]:
    train_path = PROJECT_ROOT / "data" / "processed" / "letter_recognition" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "letter_recognition" / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing processed Letter split. Run python scripts/download_datasets.py and "
            "python scripts/prepare_letter_recognition.py first."
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = "letter"
    features = [column for column in train.columns if column != target]
    if features != [column for column in test.columns if column != target]:
        raise ValueError("Letter train/test feature columns do not match")
    return {
        "name": "letter_recognition",
        "source": "data/processed/letter_recognition/{train,test}.csv",
        "X_train": train[features].to_numpy(dtype=float),
        "X_test": test[features].to_numpy(dtype=float),
        "y_train": train[target].to_numpy(),
        "y_test": test[target].to_numpy(),
        "features": len(features),
        "samples": len(train) + len(test),
        "preprocessing": (
            "exact-duplicate removal and canonical stratified split were materialized upstream"
        ),
    }


def load_csv_dataset(name: str, target: str, path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python scripts/download_datasets.py first.")
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"Missing target column {target!r} in {path}")
    features = [column for column in frame.columns if column != target]
    if frame[features].isna().any().any():
        raise ValueError(f"Missing feature values in {path}")
    X_train, X_test, y_train, y_test = split_train_test(
        frame[features].to_numpy(dtype=float),
        frame[target].to_numpy(),
    )
    return {
        "name": name,
        "source": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": len(features),
        "samples": len(frame),
        "preprocessing": (
            "schema, numeric and missing-value validation; no scaling for Decision Tree"
        ),
    }


def evaluate_dataset(bundle: dict[str, object]) -> dict[str, object]:
    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    started = time.perf_counter()
    model = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    prediction_seconds = time.perf_counter() - started
    test_accuracy = float(accuracy_score(y_test, test_prediction))
    return {
        "dataset": bundle["name"],
        "source": bundle["source"],
        "steps": {
            "load": True,
            "validate_and_preprocess": True,
            "split": True,
            "fit": True,
            "predict": True,
            "metrics": True,
        },
        "samples": int(bundle["samples"]),
        "features": int(bundle["features"]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "classes": int(len(np.unique(np.concatenate([y_train, y_test])))),
        "train_accuracy": float(accuracy_score(y_train, train_prediction)),
        "test_accuracy": test_accuracy,
        "error_rate": float(1.0 - test_accuracy),
        "f1_macro": float(f1_score(y_test, test_prediction, average="macro")),
        "tree_depth": int(model.get_depth()),
        "leaf_count": int(model.get_n_leaves()),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "preprocessing": bundle["preprocessing"],
    }


def main() -> None:
    pipeline_started = time.perf_counter()
    datasets = [
        load_letter(),
        load_csv_dataset(
            "handwritten_digits",
            "target",
            PROJECT_ROOT / "data" / "raw" / "handwritten_digits" / "digits.csv",
        ),
        load_csv_dataset(
            "covertype",
            "cover_type",
            PROJECT_ROOT / "data" / "raw" / "covertype" / "covertype.csv",
        ),
    ]
    rows = [evaluate_dataset(bundle) for bundle in datasets]
    pipeline_seconds = time.perf_counter() - pipeline_started
    result = {
        "schema_version": "1.0",
        "experiment_id": "end_to_end_pipeline_validation",
        "dataset": "multiple",
        "model": "DecisionTreeClassifier baseline smoke validation",
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "test_used_for_tuning": False,
        },
        "metrics": {
            "datasets_validated": len(rows),
            "all_steps_passed": int(
                all(all(step for step in row["steps"].values()) for row in rows)
            ),
            "pipeline_seconds": float(pipeline_seconds),
        },
        "artifacts": {
            "figure_paths": [],
            "model_path": None,
        },
        "notes": (
            "Local CPU validation of load, schema/preprocessing validation, shared split, fit, "
            "predict and metrics for all three project datasets. This does not replace the "
            "optional "
            "GPU-only four-model benchmark notebook."
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "datasets": rows,
    }
    output_path = RESULTS_DIR / "end_to_end_pipeline_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2))
    print(
        pd.DataFrame(rows)[
            ["dataset", "samples", "features", "test_accuracy", "error_rate", "f1_macro"]
        ].to_string(index=False)
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
