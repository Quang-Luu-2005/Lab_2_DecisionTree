"""Generate standalone Kaggle notebooks for model benchmarks and comparison."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "notebooks" / "benchmark_models"
COMPARISON_DIR = PROJECT_ROOT / "notebooks" / "model_comparison"


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "none", "isGpuEnabled": False, "internet": False},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_IMPORTS = r"""
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import load_digits
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

try:
    from IPython.display import FileLink, display
except ImportError:
    FileLink = None

    def display(value):
        print(value)

sns.set_theme(style="whitegrid")
"""


COMMON_SETUP = r"""
RANDOM_STATE = 42
TEST_SIZE = 0.20
KAGGLE_WORKING = Path("/kaggle/working")
RUN_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd()
FIGURES_DIR = RUN_ROOT / "figures"
RESULTS_DIR = RUN_ROOT / "results"
MODELS_DIR = RUN_ROOT / "models"
for directory in (FIGURES_DIR, RESULTS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LETTER_FEATURES = [
    "x_box", "y_box", "width", "high", "onpix", "x_bar", "y_bar", "x2bar",
    "y2bar", "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx",
]


def detect_hardware():
    hardware = {
        "environment": "kaggle" if KAGGLE_WORKING.exists() else "local",
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "gpu_available": False,
        "gpus": [],
        "model_compute_device": "cpu",
    }
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        for line in output.splitlines():
            name, memory_mb, driver = [part.strip() for part in line.split(",", maxsplit=2)]
            hardware["gpus"].append(
                {"name": name, "memory_mb": int(memory_mb), "driver_version": driver}
            )
        hardware["gpu_available"] = bool(hardware["gpus"])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return hardware


hardware = detect_hardware()
print({"python": platform.python_version(), "sklearn": sklearn.__version__})
print("Hardware:", hardware)
"""


COMMON_DATA = r"""
def find_letter_split():
    roots = [Path.cwd(), Path.cwd().parent, Path("/kaggle/input")]
    checked = set()
    for root in roots:
        if not root.exists():
            continue
        for train_path in root.rglob("train.csv"):
            test_path = train_path.with_name("test.csv")
            key = str(train_path.resolve())
            if key in checked or not test_path.exists():
                continue
            checked.add(key)
            try:
                columns = set(pd.read_csv(train_path, nrows=2).columns)
                test_columns = set(pd.read_csv(test_path, nrows=2).columns)
            except (OSError, pd.errors.ParserError):
                continue
            required = set(LETTER_FEATURES + ["letter"])
            if required.issubset(columns) and required.issubset(test_columns):
                return train_path, test_path
    raise FileNotFoundError(
        "Không tìm thấy canonical Letter train.csv/test.csv. "
        "Trên Kaggle, hãy Add Input dataset chứa hai file processed này."
    )


def load_benchmark_datasets():
    started = time.perf_counter()
    letter_train_path, letter_test_path = find_letter_split()
    letter_train = pd.read_csv(letter_train_path)
    letter_test = pd.read_csv(letter_test_path)

    digits = load_digits(as_frame=True)
    digits_frame = digits.frame.rename(columns={"target": "label"})
    digits_train, digits_test = train_test_split(
        digits_frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=digits_frame["label"],
    )
    datasets = {
        "letter_recognition": {
            "X_train": letter_train[LETTER_FEATURES],
            "y_train": letter_train["letter"],
            "X_test": letter_test[LETTER_FEATURES],
            "y_test": letter_test["letter"],
            "source": str(letter_train_path.parent),
        },
        "handwritten_digits": {
            "X_train": digits_train.drop(columns="label"),
            "y_train": digits_train["label"].astype("int64"),
            "X_test": digits_test.drop(columns="label"),
            "y_test": digits_test["label"].astype("int64"),
            "source": "sklearn.datasets.load_digits",
        },
    }
    return datasets, time.perf_counter() - started


datasets, data_loading_seconds = load_benchmark_datasets()
for dataset_name, parts in datasets.items():
    print(
        dataset_name,
        "train/test:",
        parts["X_train"].shape,
        parts["X_test"].shape,
        "source:",
        parts["source"],
    )
"""


def benchmark_notebook(spec: dict[str, object]) -> dict[str, object]:
    model_import = spec["model_import"]
    estimator_code = indent(spec["estimator_code"], "    ")
    title = spec["title"]
    experiment_id = spec["experiment_id"]
    model_name = spec["model_name"]
    scaling_note = spec["scaling_note"]

    imports = COMMON_IMPORTS + "\n" + model_import
    setup = (
        COMMON_SETUP
        + f'\nEXPERIMENT_ID = "{experiment_id}"\n'
        + f'MODEL_NAME = "{model_name}"\n'
        + f"MODEL_CONFIG = {spec['model_config']!r}\n"
    )
    train = f"""
def make_estimator():
{estimator_code}


def evaluate_dataset(dataset_name, parts):
    estimator = make_estimator()
    fit_started = time.perf_counter()
    estimator.fit(parts["X_train"], parts["y_train"])
    training_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    prediction = estimator.predict(parts["X_test"])
    prediction_seconds = time.perf_counter() - predict_started
    train_prediction = estimator.predict(parts["X_train"])

    train_accuracy = accuracy_score(parts["y_train"], train_prediction)
    test_accuracy = accuracy_score(parts["y_test"], prediction)
    metrics = {{
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "error_rate": 1.0 - test_accuracy,
        "precision_macro": precision_score(
            parts["y_test"], prediction, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            parts["y_test"], prediction, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(parts["y_test"], prediction, average="macro"),
        "generalization_gap": train_accuracy - test_accuracy,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "train_samples": len(parts["X_train"]),
        "test_samples": len(parts["X_test"]),
        "features": parts["X_train"].shape[1],
    }}

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        parts["y_test"], prediction, cmap="Blues", colorbar=False, values_format="d", ax=ax
    )
    ax.set_title(f"{{dataset_name}} - {{MODEL_NAME}}")
    fig.tight_layout()
    figure_path = FIGURES_DIR / f"{{EXPERIMENT_ID}}__{{dataset_name}}__confusion_matrix.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()

    model_path = MODELS_DIR / f"{{EXPERIMENT_ID}}__{{dataset_name}}.joblib"
    joblib.dump(estimator, model_path)
    return estimator, metrics, figure_path, model_path


pipeline_started = time.perf_counter()
evaluations = {{}}
artifact_paths = []
for dataset_name, parts in datasets.items():
    _, metrics, figure_path, model_path = evaluate_dataset(dataset_name, parts)
    evaluations[dataset_name] = metrics
    artifact_paths.extend([figure_path, model_path])

summary = pd.DataFrame(evaluations).T
display(summary.round(4))
"""
    save = r"""
summary_path = RESULTS_DIR / f"{EXPERIMENT_ID}__summary.csv"
summary.reset_index(names="dataset").to_csv(summary_path, index=False)
result_path = RESULTS_DIR / f"{EXPERIMENT_ID}.json"
pipeline_seconds = time.perf_counter() - pipeline_started
result = {
    "schema_version": "1.0",
    "experiment_id": EXPERIMENT_ID,
    "model": MODEL_NAME,
    "model_config": MODEL_CONFIG,
    "split": {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": True,
        "letter_split": "canonical preprocessed train/test",
    },
    "data_loading_seconds": data_loading_seconds,
    "pipeline_seconds": pipeline_seconds,
    "hardware": hardware,
    "datasets": evaluations,
    "notes": "All sklearn estimators compute on CPU; preprocessing is fit on train only.",
    "created_at_utc": datetime.now(UTC).isoformat(),
}
result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
artifact_paths.extend([summary_path, result_path])

archive_path = RUN_ROOT / f"{EXPERIMENT_ID}__outputs.zip"
with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
    for artifact_path in artifact_paths:
        archive.write(artifact_path, artifact_path.relative_to(RUN_ROOT))

print(f"Created ZIP ({archive_path.stat().st_size / 1024**2:.1f} MB): {archive_path}")
if FileLink is not None:
    display(FileLink(str(archive_path)))
"""
    return notebook(
        [
            markdown(
                f"""
                # {title}

                Benchmark độc lập cho Letter Recognition và Handwritten Digits. Hai bộ dữ
                liệu dùng đúng protocol `test_size=0.20`, `random_state=42`, stratification;
                Letter dùng canonical split đã chuẩn bị. Accelerator khuyến nghị: **None (CPU)**.
                """
            ),
            code(imports),
            code(setup),
            markdown(
                f"""
                ## Preprocessing

                {scaling_note} Mọi transformer chỉ được fit trên tập train thông qua pipeline,
                tránh data leakage.
                """
            ),
            code(COMMON_DATA),
            code(train),
            markdown(
                "## Kết quả\n\nSo sánh accuracy, macro-F1, train-test gap và thời gian với các model khác ở notebook tổng hợp."
            ),
            code(save),
        ]
    )


THREE_DATASET_IMPORTS = r"""
import gc
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import fetch_covtype, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from IPython.display import FileLink, display
except ImportError:
    FileLink = None

    def display(value):
        print(value)

sns.set_theme(style="whitegrid")
"""


THREE_DATASET_SETUP = r"""
PIPELINE_STARTED = time.perf_counter()
RANDOM_STATE = 42
TEST_SIZE = 0.20
EXPERIMENT_ID = "three_dataset_three_model_benchmark"
TARGET = "cover_type"
PREDICTION_BATCH_SIZE = int(os.getenv("PREDICTION_BATCH_SIZE", "2000"))
TRAIN_DIAGNOSTIC_SIZE = int(os.getenv("TRAIN_DIAGNOSTIC_SIZE", "10000"))

KAGGLE_WORKING = Path("/kaggle/working")
RUN_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd()
FIGURES_DIR = RUN_ROOT / "figures"
RESULTS_DIR = RUN_ROOT / "results"
for directory in (FIGURES_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def detect_hardware():
    hardware = {
        "environment": "kaggle" if KAGGLE_WORKING.exists() else "local",
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "gpu_available": False,
        "gpus": [],
        "model_compute_device": "cpu",
    }
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        for line in output.splitlines():
            name, memory_mb, driver = [part.strip() for part in line.split(",", maxsplit=2)]
            hardware["gpus"].append(
                {"name": name, "memory_mb": int(float(memory_mb)), "driver_version": driver}
            )
        hardware["gpu_available"] = bool(hardware["gpus"])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return hardware


hardware = detect_hardware()
print({"python": platform.python_version(), "sklearn": sklearn.__version__})
print("Hardware:", hardware)
print("Compute device used by all three sklearn estimators: CPU")
"""


THREE_DATASET_DATA = r"""
data_started = time.perf_counter()
LETTER_FEATURES = [
    "x_box", "y_box", "width", "high", "onpix", "x_bar", "y_bar", "x2bar",
    "y2bar", "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx",
]


def find_letter_split():
    roots = [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent]
    for root in roots:
        if not root.exists():
            continue
        for train_path in sorted(root.rglob("train.csv")):
            test_path = train_path.with_name("test.csv")
            if not test_path.exists():
                continue
            try:
                train_columns = set(pd.read_csv(train_path, nrows=2).columns)
                test_columns = set(pd.read_csv(test_path, nrows=2).columns)
            except (OSError, pd.errors.ParserError):
                continue
            required = set(LETTER_FEATURES + ["letter"])
            if required.issubset(train_columns) and required.issubset(test_columns):
                return train_path, test_path
    raise FileNotFoundError(
        "Missing canonical Letter train.csv/test.csv. Add the processed dataset to Kaggle."
    )


def find_covertype_csv():
    roots = [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent]
    for root in roots:
        if not root.exists():
            continue
        for name in ("covertype.csv", "covtype.csv"):
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
    return None


def normalize_target(frame):
    target_candidates = {"cover_type", "covertype", "cover type", "target"}
    for column in frame.columns:
        if str(column).strip().lower() in target_candidates:
            return frame.rename(columns={column: TARGET})
    raise ValueError("Covertype CSV must contain a cover_type or target column")


letter_train_path, letter_test_path = find_letter_split()
letter_train = pd.read_csv(letter_train_path)
letter_test = pd.read_csv(letter_test_path)

digits = load_digits(as_frame=True)
digits_frame = digits.frame.rename(columns={"target": "label"})
digits_train, digits_test = train_test_split(
    digits_frame,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=digits_frame["label"],
)

covertype_path = find_covertype_csv()
if covertype_path is None:
    bundle = fetch_covtype(
        as_frame=True,
        data_home=RUN_ROOT / ".cache" / "scikit_learn_data",
    )
    covertype_frame = bundle.data.copy()
    covertype_frame[TARGET] = bundle.target.astype("int64").to_numpy()
    covertype_source = "sklearn.datasets.fetch_covtype() / UCI Covertype"
else:
    covertype_frame = normalize_target(pd.read_csv(covertype_path))
    covertype_source = str(covertype_path)

covertype_frame[TARGET] = pd.to_numeric(
    covertype_frame[TARGET], errors="raise"
).astype("int64")
assert covertype_frame.shape[1] == 55
assert covertype_frame[TARGET].nunique() == 7
assert not covertype_frame.isna().any().any()

covertype_features = [column for column in covertype_frame.columns if column != TARGET]
covertype_X = covertype_frame[covertype_features]
covertype_y = covertype_frame[TARGET]
covertype_X_train, covertype_X_test, covertype_y_train, covertype_y_test = train_test_split(
    covertype_X,
    covertype_y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=covertype_y,
)

datasets = {
    "letter_recognition": {
        "X_train": letter_train[LETTER_FEATURES],
        "y_train": letter_train["letter"],
        "X_test": letter_test[LETTER_FEATURES],
        "y_test": letter_test["letter"],
        "source": str(letter_train_path.parent),
        "scope": "full_canonical_split",
    },
    "handwritten_digits": {
        "X_train": digits_train.drop(columns="label"),
        "y_train": digits_train["label"].astype("int64"),
        "X_test": digits_test.drop(columns="label"),
        "y_test": digits_test["label"].astype("int64"),
        "source": "sklearn.datasets.load_digits",
        "scope": "full_dataset_split",
    },
    "covertype": {
        "X_train": covertype_X_train,
        "y_train": covertype_y_train,
        "X_test": covertype_X_test,
        "y_test": covertype_y_test,
        "source": covertype_source,
        "scope": "full_dataset_split",
    },
}

for dataset_name, parts in datasets.items():
    diagnostic_size = min(TRAIN_DIAGNOSTIC_SIZE, len(parts["X_train"]))
    if diagnostic_size == len(parts["X_train"]):
        diagnostic_X = parts["X_train"]
        diagnostic_y = parts["y_train"]
    else:
        diagnostic_X, _, diagnostic_y, _ = train_test_split(
            parts["X_train"],
            parts["y_train"],
            train_size=diagnostic_size,
            random_state=RANDOM_STATE,
            stratify=parts["y_train"],
        )
    parts["X_train_diagnostic"] = diagnostic_X
    parts["y_train_diagnostic"] = diagnostic_y

data_loading_seconds = time.perf_counter() - data_started
print("Data loading:", round(data_loading_seconds, 2), "seconds")
for dataset_name, parts in datasets.items():
    print(
        dataset_name,
        "train/test:",
        parts["X_train"].shape,
        parts["X_test"].shape,
        "diagnostic train:",
        parts["X_train_diagnostic"].shape,
    )
"""


THREE_DATASET_BENCHMARK = r"""
MODEL_CONFIGS = {
    "Random Forest": {
        "n_estimators": 100,
        "criterion": "gini",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "learning_strategy": "eager",
    },
    "SVM (RBF)": {
        "pipeline": "StandardScaler -> SVC",
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "cache_size_mb": 4096,
        "learning_strategy": "eager",
    },
    "KNN": {
        "pipeline": "StandardScaler -> KNeighborsClassifier",
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "brute",
        "n_jobs": -1,
        "learning_strategy": "lazy",
    },
}

MODEL_SLUGS = {
    "Random Forest": "random_forest",
    "SVM (RBF)": "svm_rbf",
    "KNN": "knn",
}


def make_estimator(model_name):
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "SVM (RBF)":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=4096)),
            ]
        )
    if model_name == "KNN":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="uniform",
                        algorithm="brute",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise KeyError(model_name)


def predict_in_batches(estimator, frame, dataset_name, model_name, stage):
    predictions = []
    total_batches = (len(frame) + PREDICTION_BATCH_SIZE - 1) // PREDICTION_BATCH_SIZE
    started = time.perf_counter()
    for batch_number, start in enumerate(
        range(0, len(frame), PREDICTION_BATCH_SIZE),
        start=1,
    ):
        stop = min(start + PREDICTION_BATCH_SIZE, len(frame))
        predictions.append(estimator.predict(frame.iloc[start:stop]))
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
            elapsed = time.perf_counter() - started
            print(
                f"{dataset_name}/{model_name} {stage}: "
                f"batch {batch_number}/{total_batches}; elapsed={elapsed:.1f}s"
            )
    return np.concatenate(predictions), time.perf_counter() - started


evaluations = {dataset_name: {} for dataset_name in datasets}
failures = []


def write_checkpoint():
    result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "models": list(MODEL_CONFIGS),
        "datasets": evaluations,
        "failures": failures,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
        },
        "data_loading_seconds": data_loading_seconds,
        "hardware": hardware,
        "notes": (
            "All three models run on full datasets and CPU. KNN is a lazy learner, so its "
            "fit time is not directly comparable to eager learners. Covertype train accuracy "
            "uses a fixed stratified diagnostic sample; test metrics use the entire test set."
        ),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    result_path = RESULTS_DIR / f"{EXPERIMENT_ID}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for dataset_name, model_results in evaluations.items():
        for model_name, metrics in model_results.items():
            rows.append({"dataset": dataset_name, "model": model_name, **metrics})
    if rows:
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / f"{EXPERIMENT_ID}__summary.csv",
            index=False,
        )
    return result_path


def evaluate_model(dataset_name, model_name):
    parts = datasets[dataset_name]
    estimator = make_estimator(model_name)
    fit_started = time.perf_counter()
    estimator.fit(parts["X_train"], parts["y_train"])
    training_seconds = time.perf_counter() - fit_started
    print(f"{dataset_name}/{model_name} fit complete: {training_seconds:.2f}s")

    test_prediction, prediction_seconds = predict_in_batches(
        estimator,
        parts["X_test"],
        dataset_name,
        model_name,
        "test prediction",
    )
    train_prediction, diagnostic_prediction_seconds = predict_in_batches(
        estimator,
        parts["X_train_diagnostic"],
        dataset_name,
        model_name,
        "train diagnostic prediction",
    )
    train_accuracy = accuracy_score(parts["y_train_diagnostic"], train_prediction)
    test_accuracy = accuracy_score(parts["y_test"], test_prediction)
    metrics = {
        "status": "completed",
        "learning_strategy": MODEL_CONFIGS[model_name]["learning_strategy"],
        "train_accuracy": float(train_accuracy),
        "train_accuracy_sample_size": len(parts["X_train_diagnostic"]),
        "test_accuracy": float(test_accuracy),
        "error_rate": float(1.0 - test_accuracy),
        "precision_macro": float(
            precision_score(parts["y_test"], test_prediction, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(parts["y_test"], test_prediction, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(parts["y_test"], test_prediction, average="macro")),
        "generalization_gap": float(train_accuracy - test_accuracy),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "diagnostic_prediction_seconds": float(diagnostic_prediction_seconds),
        "total_model_seconds": float(training_seconds + prediction_seconds),
        "train_samples": len(parts["X_train"]),
        "test_samples": len(parts["X_test"]),
        "features": parts["X_train"].shape[1],
        "dataset_scope": parts["scope"],
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        parts["y_test"],
        test_prediction,
        labels=np.sort(parts["y_test"].unique()),
        cmap="Blues",
        colorbar=False,
        values_format="d",
        ax=ax,
    )
    ax.set_title(f"{dataset_name.replace('_', ' ').title()} - {model_name}")
    fig.tight_layout()
    figure_path = FIGURES_DIR / (
        f"{EXPERIMENT_ID}__{dataset_name}__{MODEL_SLUGS[model_name]}__confusion_matrix.png"
    )
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    del estimator, test_prediction, train_prediction
    gc.collect()
    return metrics


def run_and_checkpoint(dataset_name, model_name):
    print(f"\n===== {dataset_name} / {model_name} =====")
    started = time.perf_counter()
    try:
        evaluations[dataset_name][model_name] = evaluate_model(dataset_name, model_name)
    except (MemoryError, OSError, RuntimeError, ValueError) as error:
        failures.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "error_type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(f"{dataset_name}/{model_name} failed: {type(error).__name__}: {error}")
    result_path = write_checkpoint()
    print("Checkpoint:", result_path)
"""


THREE_DATASET_SUMMARY = r"""
rows = []
for dataset_name, model_results in evaluations.items():
    for model_name, metrics in model_results.items():
        rows.append({"dataset": dataset_name, "model": model_name, **metrics})
if not rows:
    raise RuntimeError("No model completed; inspect the checkpoint JSON.")

summary = pd.DataFrame(rows).sort_values(["dataset", "f1_macro"], ascending=[True, False])
display(summary.round(4))

artifact_paths = list(FIGURES_DIR.glob(f"{EXPERIMENT_ID}__*.png"))
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
performance_long = summary.melt(
    id_vars=["dataset", "model"],
    value_vars=["test_accuracy", "f1_macro"],
    var_name="metric",
    value_name="score",
)
for ax, (dataset_name, frame) in zip(
    axes,
    performance_long.groupby("dataset"),
    strict=True,
):
    sns.barplot(data=frame, x="model", y="score", hue="metric", ax=ax)
    ax.set(
        title=dataset_name.replace("_", " ").title(),
        xlabel="",
        ylabel="Score",
        ylim=(0, 1),
    )
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Three-model performance on three full datasets")
fig.tight_layout()
performance_path = FIGURES_DIR / f"{EXPERIMENT_ID}__performance.png"
fig.savefig(performance_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
artifact_paths.append(performance_path)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(
    data=summary,
    x="dataset",
    y="generalization_gap",
    hue="model",
    errorbar=None,
    ax=axes[0],
)
axes[0].set(title="Generalization gap", xlabel="", ylabel="Train accuracy - test accuracy")
axes[0].tick_params(axis="x", rotation=15)
sns.barplot(
    data=summary,
    x="dataset",
    y="total_model_seconds",
    hue="model",
    errorbar=None,
    ax=axes[1],
)
axes[1].set(
    title="Fit + full-test prediction (log scale)",
    xlabel="",
    ylabel="Seconds",
    yscale="log",
)
axes[1].tick_params(axis="x", rotation=15)
fig.tight_layout()
runtime_path = FIGURES_DIR / f"{EXPERIMENT_ID}__gap_and_runtime.png"
fig.savefig(runtime_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
artifact_paths.append(runtime_path)

best_by_dataset = (
    summary.loc[summary.groupby("dataset")["f1_macro"].idxmax()]
    .set_index("dataset")[["model", "test_accuracy", "f1_macro"]]
    .to_dict(orient="index")
)
print("Best by macro-F1:", best_by_dataset)
for dataset_name, frame in summary.groupby("dataset"):
    fastest_fit = frame.loc[frame["training_seconds"].idxmin()]
    fastest_prediction = frame.loc[frame["prediction_seconds"].idxmin()]
    fastest_total = frame.loc[frame["total_model_seconds"].idxmin()]
    eager = frame.loc[frame["learning_strategy"] == "eager"]
    fastest_eager_fit = eager.loc[eager["training_seconds"].idxmin()]
    print(
        dataset_name,
        "fastest raw fit=",
        fastest_fit["model"],
        "; fastest eager fit=",
        fastest_eager_fit["model"],
        "; fastest prediction=",
        fastest_prediction["model"],
        "; fastest total=",
        fastest_total["model"],
    )

pipeline_seconds = time.perf_counter() - PIPELINE_STARTED
result_path = write_checkpoint()
result = json.loads(result_path.read_text(encoding="utf-8"))
result["pipeline_seconds"] = pipeline_seconds
result["best_by_dataset"] = best_by_dataset
result["created_at_utc"] = datetime.now(UTC).isoformat()
result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
summary_path = RESULTS_DIR / f"{EXPERIMENT_ID}__summary.csv"
artifact_paths.extend([summary_path, result_path])

archive_path = RUN_ROOT / f"{EXPERIMENT_ID}__outputs.zip"
with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
    for artifact_path in sorted(set(artifact_paths)):
        archive.write(artifact_path, artifact_path.relative_to(RUN_ROOT))
print(f"Pipeline: {pipeline_seconds:.2f}s")
print(f"Created ZIP ({archive_path.stat().st_size / 1024**2:.1f} MB): {archive_path}")
if FileLink is not None:
    display(FileLink(str(archive_path)))
"""


def three_dataset_three_model_notebook() -> dict[str, object]:
    model_names = [
        "Random Forest",
        "SVM (RBF)",
        "KNN",
    ]
    cells = [
        markdown(
            r"""
            # Three datasets × three models

            Notebook benchmark đúng ba model `Random Forest`, `SVM (RBF)` và `KNN` trên cả
            `Letter Recognition`, `Handwritten Digits` và `Covertype`. Mỗi dataset có split
            riêng nhưng ba model của dataset đó dùng cùng train/test indices (`test_size=0.20`,
            `random_state=42`, stratification). Covertype dùng toàn bộ 581.012 mẫu.

            Chọn **Accelerator: None (CPU)**. Các estimator scikit-learn này không dùng GPU.
            KNN/SVM trên Covertype có thể mất rất lâu; notebook lưu checkpoint sau từng model,
            dự đoán theo batch và đặt KNN/SVM của Covertype ở cuối quy trình.
            """
        ),
        code(THREE_DATASET_IMPORTS),
        code(THREE_DATASET_SETUP),
        code(THREE_DATASET_DATA),
        markdown(
            r"""
            ## Model protocol

            Random Forest không scaling. SVM và KNN dùng `StandardScaler` trong pipeline, fit
            trên train only. Test metrics dùng toàn bộ test set; với Covertype, train accuracy
            và generalization gap dùng diagnostic sample stratified 10.000 dòng để tránh thêm
            một lượt inference 464.809 dòng cho KNN/SVM. File JSON ghi rõ phạm vi này.
            """
        ),
        code(THREE_DATASET_BENCHMARK),
    ]
    for dataset_name in ("letter_recognition", "handwritten_digits", "covertype"):
        for model_name in model_names:
            cells.extend(
                [
                    markdown(f"## {dataset_name.replace('_', ' ').title()} — {model_name}"),
                    code(f'run_and_checkpoint("{dataset_name}", "{model_name}")'),
                ]
            )
    cells.extend(
        [
            markdown(
                r"""
                ## Tổng hợp và tải output

                Khi phân tích tốc độ, đọc cả fit, prediction và tổng thời gian. KNN là lazy
                learner nên thời gian fit rất ngắn không có nghĩa là pipeline end-to-end nhanh.
                """
            ),
            code(THREE_DATASET_SUMMARY),
        ]
    )
    return notebook(cells)


COVERTYPE_IMPORTS = r"""
import gc
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from IPython.display import FileLink, display
except ImportError:
    FileLink = None

    def display(value):
        print(value)

sns.set_theme(style="whitegrid")
"""


COVERTYPE_SETUP = r"""
PIPELINE_STARTED = time.perf_counter()
RANDOM_STATE = 42
TEST_SIZE = 0.20
EXPERIMENT_ID = "covertype_four_model_benchmark"
TARGET = "cover_type"
PREDICTION_BATCH_SIZE = int(os.getenv("COVERTYPE_PREDICTION_BATCH_SIZE", "2000"))
TRAIN_DIAGNOSTIC_SIZE = int(os.getenv("COVERTYPE_TRAIN_DIAGNOSTIC_SIZE", "10000"))

KAGGLE_WORKING = Path("/kaggle/working")
RUN_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd()
FIGURES_DIR = RUN_ROOT / "figures"
RESULTS_DIR = RUN_ROOT / "results"
for directory in (FIGURES_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def detect_hardware():
    hardware = {
        "environment": "kaggle" if KAGGLE_WORKING.exists() else "local",
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "gpu_available": False,
        "gpus": [],
        "model_compute_device": "cpu",
    }
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        for line in output.splitlines():
            name, memory_mb, driver = [part.strip() for part in line.split(",", maxsplit=2)]
            hardware["gpus"].append(
                {"name": name, "memory_mb": int(float(memory_mb)), "driver_version": driver}
            )
        hardware["gpu_available"] = bool(hardware["gpus"])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return hardware


hardware = detect_hardware()
print({"python": platform.python_version(), "sklearn": sklearn.__version__})
print("Hardware:", hardware)
print("Compute device used by all four sklearn estimators: CPU")
"""


COVERTYPE_DATA = r"""
data_started = time.perf_counter()


def find_covertype_csv():
    roots = [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent]
    for root in roots:
        if not root.exists():
            continue
        for name in ("covertype.csv", "covtype.csv"):
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
    return None


def normalize_target(frame):
    target_candidates = {"cover_type", "covertype", "cover type", "target"}
    for column in frame.columns:
        if str(column).strip().lower() in target_candidates:
            return frame.rename(columns={column: TARGET})
    raise ValueError("CSV must contain a Cover_Type/cover_type/target column")


data_path = find_covertype_csv()
if data_path is None:
    bundle = fetch_covtype(
        as_frame=True,
        data_home=RUN_ROOT / ".cache" / "scikit_learn_data",
    )
    covertype_df = bundle.data.copy()
    covertype_df[TARGET] = bundle.target.astype("int64").to_numpy()
    data_source = "sklearn.datasets.fetch_covtype() / UCI Covertype"
else:
    covertype_df = normalize_target(pd.read_csv(data_path))
    data_source = str(data_path)

covertype_df[TARGET] = pd.to_numeric(covertype_df[TARGET], errors="raise").astype("int64")
assert covertype_df.shape[1] == 55
assert covertype_df[TARGET].nunique() == 7
assert not covertype_df.isna().any().any()

FEATURES = [column for column in covertype_df.columns if column != TARGET]
X = covertype_df[FEATURES]
y = covertype_df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

diagnostic_size = min(TRAIN_DIAGNOSTIC_SIZE, len(X_train))
X_train_diagnostic, _, y_train_diagnostic, _ = train_test_split(
    X_train,
    y_train,
    train_size=diagnostic_size,
    random_state=RANDOM_STATE,
    stratify=y_train,
)

data_loading_seconds = time.perf_counter() - data_started
dataset_memory_mb = covertype_df.memory_usage(deep=True).sum() / 1024**2
print("Source:", data_source)
print("Full dataset:", covertype_df.shape)
print("Train/test:", X_train.shape, X_test.shape)
print("Train diagnostic sample:", X_train_diagnostic.shape)
print(f"DataFrame memory: {dataset_memory_mb:.1f} MB")
print(f"Data loading: {data_loading_seconds:.2f} seconds")
"""


COVERTYPE_BENCHMARK = r"""
MODEL_CONFIGS = {
    "Decision Tree": {
        "estimator": "DecisionTreeClassifier",
        "criterion": "gini",
        "random_state": RANDOM_STATE,
        "learning_strategy": "eager",
    },
    "Random Forest": {
        "estimator": "RandomForestClassifier",
        "n_estimators": 100,
        "criterion": "gini",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "learning_strategy": "eager",
    },
    "KNN": {
        "pipeline": "StandardScaler -> KNeighborsClassifier",
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "brute",
        "n_jobs": -1,
        "learning_strategy": "lazy",
    },
    "SVM (RBF)": {
        "pipeline": "StandardScaler -> SVC",
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "cache_size_mb": 4096,
        "learning_strategy": "eager",
    },
}

MODEL_SLUGS = {
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "KNN": "knn",
    "SVM (RBF)": "svm_rbf",
}


def make_estimator(model_name):
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "KNN":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="uniform",
                        algorithm="brute",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "SVM (RBF)":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=4096),
                ),
            ]
        )
    raise KeyError(model_name)


def predict_in_batches(estimator, frame, model_name, stage):
    predictions = []
    total_batches = (len(frame) + PREDICTION_BATCH_SIZE - 1) // PREDICTION_BATCH_SIZE
    started = time.perf_counter()
    for batch_number, start in enumerate(
        range(0, len(frame), PREDICTION_BATCH_SIZE),
        start=1,
    ):
        stop = min(start + PREDICTION_BATCH_SIZE, len(frame))
        predictions.append(estimator.predict(frame.iloc[start:stop]))
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
            elapsed = time.perf_counter() - started
            print(
                f"{model_name} {stage}: batch {batch_number}/{total_batches}; "
                f"elapsed={elapsed:.1f}s"
            )
    return np.concatenate(predictions), time.perf_counter() - started


evaluations = {}
failures = {}


def write_checkpoint():
    checkpoint = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "dataset": "covertype",
        "dataset_scope": "full",
        "data_source": data_source,
        "dataset_samples": len(covertype_df),
        "features": len(FEATURES),
        "classes": int(y.nunique()),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "train_accuracy_scope": {
            "kind": "fixed_stratified_diagnostic_sample",
            "samples": len(X_train_diagnostic),
            "reason": "Avoid an additional full-train inference pass for SVM and KNN.",
        },
        "prediction_protocol": {
            "scope": "entire_test_set",
            "batch_size": PREDICTION_BATCH_SIZE,
        },
        "model_configs": MODEL_CONFIGS,
        "models": evaluations,
        "failures": failures,
        "data_loading_seconds": data_loading_seconds,
        "dataset_memory_mb": dataset_memory_mb,
        "hardware": hardware,
        "notes": (
            "All models use the same full-data split and CPU. KNN fit time is not directly "
            "comparable to eager learners because KNN is a lazy learner."
        ),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    result_path = RESULTS_DIR / f"{EXPERIMENT_ID}.json"
    result_path.write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if evaluations:
        rows = []
        for name, metrics in evaluations.items():
            rows.append({"model": name, **metrics})
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / f"{EXPERIMENT_ID}__summary.csv",
            index=False,
        )
    return result_path


def evaluate_model(model_name):
    print(f"\n===== {model_name} =====")
    estimator = make_estimator(model_name)
    fit_started = time.perf_counter()
    estimator.fit(X_train, y_train)
    training_seconds = time.perf_counter() - fit_started
    print(f"{model_name} fit complete: {training_seconds:.2f}s")

    test_prediction, prediction_seconds = predict_in_batches(
        estimator,
        X_test,
        model_name,
        "test prediction",
    )
    train_prediction, diagnostic_prediction_seconds = predict_in_batches(
        estimator,
        X_train_diagnostic,
        model_name,
        "train diagnostic prediction",
    )

    train_accuracy = accuracy_score(y_train_diagnostic, train_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)
    metrics = {
        "status": "completed",
        "learning_strategy": MODEL_CONFIGS[model_name]["learning_strategy"],
        "train_accuracy": float(train_accuracy),
        "train_accuracy_sample_size": len(X_train_diagnostic),
        "test_accuracy": float(test_accuracy),
        "error_rate": float(1.0 - test_accuracy),
        "precision_macro": float(
            precision_score(y_test, test_prediction, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, test_prediction, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, test_prediction, average="macro")),
        "generalization_gap": float(train_accuracy - test_accuracy),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "diagnostic_prediction_seconds": float(diagnostic_prediction_seconds),
        "total_model_seconds": float(training_seconds + prediction_seconds),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": len(FEATURES),
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_prediction,
        labels=np.sort(y.unique()),
        cmap="Blues",
        colorbar=False,
        values_format="d",
        ax=ax,
    )
    ax.set_title(f"Covertype full test set - {model_name}")
    fig.tight_layout()
    figure_path = FIGURES_DIR / (
        f"{EXPERIMENT_ID}__{MODEL_SLUGS[model_name]}__confusion_matrix.png"
    )
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    del estimator, test_prediction, train_prediction
    gc.collect()
    return metrics


def run_and_checkpoint(model_name):
    model_started = time.perf_counter()
    try:
        evaluations[model_name] = evaluate_model(model_name)
    except (MemoryError, OSError, RuntimeError, ValueError) as error:
        failures[model_name] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
            "elapsed_seconds": time.perf_counter() - model_started,
        }
        print(f"{model_name} failed: {type(error).__name__}: {error}")
    result_path = write_checkpoint()
    print("Checkpoint:", result_path)
"""


COVERTYPE_SUMMARY = r"""
if not evaluations:
    raise RuntimeError("No model completed; inspect the failure records in the checkpoint JSON.")

summary = pd.DataFrame(
    [{"model": model_name, **metrics} for model_name, metrics in evaluations.items()]
).sort_values("f1_macro", ascending=False)
display(summary.round(4))

artifact_paths = list(FIGURES_DIR.glob(f"{EXPERIMENT_ID}__*.png"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
performance_long = summary.melt(
    id_vars="model",
    value_vars=["test_accuracy", "f1_macro"],
    var_name="metric",
    value_name="score",
)
sns.barplot(data=performance_long, x="model", y="score", hue="metric", ax=axes[0])
axes[0].set(title="Covertype full-set performance", xlabel="", ylabel="Score", ylim=(0, 1))
axes[0].tick_params(axis="x", rotation=20)

timing_long = summary.melt(
    id_vars="model",
    value_vars=["training_seconds", "prediction_seconds"],
    var_name="stage",
    value_name="seconds",
)
sns.barplot(data=timing_long, x="model", y="seconds", hue="stage", ax=axes[1])
axes[1].set(title="Covertype runtime (log scale)", xlabel="", ylabel="Seconds", yscale="log")
axes[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
overview_path = FIGURES_DIR / f"{EXPERIMENT_ID}__performance_and_runtime.png"
fig.savefig(overview_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
artifact_paths.append(overview_path)

fastest_fit = summary.loc[summary["training_seconds"].idxmin()]
fastest_predict = summary.loc[summary["prediction_seconds"].idxmin()]
fastest_total = summary.loc[summary["total_model_seconds"].idxmin()]
eager = summary.loc[summary["learning_strategy"] == "eager"]
fastest_eager_fit = eager.loc[eager["training_seconds"].idxmin()]

print("Fastest raw fit:", fastest_fit["model"])
print(
    "Fastest eager-model fit:",
    fastest_eager_fit["model"],
    "(KNN is excluded here because fitting is lazy)",
)
print("Fastest full-test prediction:", fastest_predict["model"])
print("Fastest fit + full-test prediction:", fastest_total["model"])

pipeline_seconds = time.perf_counter() - PIPELINE_STARTED
result_path = write_checkpoint()
result = json.loads(result_path.read_text(encoding="utf-8"))
result["pipeline_seconds"] = pipeline_seconds
result["conclusions"] = {
    "fastest_raw_fit_model": fastest_fit["model"],
    "fastest_eager_fit_model": fastest_eager_fit["model"],
    "fastest_full_test_prediction_model": fastest_predict["model"],
    "fastest_fit_plus_prediction_model": fastest_total["model"],
    "knn_timing_caveat": "KNN is a lazy learner; most computation occurs during prediction.",
}
result["created_at_utc"] = datetime.now(UTC).isoformat()
result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

summary_path = RESULTS_DIR / f"{EXPERIMENT_ID}__summary.csv"
artifact_paths.extend([summary_path, result_path])
archive_path = RUN_ROOT / f"{EXPERIMENT_ID}__outputs.zip"
with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
    for artifact_path in sorted(set(artifact_paths)):
        archive.write(artifact_path, artifact_path.relative_to(RUN_ROOT))

print(f"Pipeline: {pipeline_seconds:.2f}s")
print(f"Created ZIP ({archive_path.stat().st_size / 1024**2:.1f} MB): {archive_path}")
if FileLink is not None:
    display(FileLink(str(archive_path)))
"""


def covertype_benchmark_notebook() -> dict[str, object]:
    return notebook(
        [
            markdown(
                r"""
                # Covertype full benchmark - Decision Tree, Random Forest, KNN, SVM

                Benchmark dùng **toàn bộ 581.012 mẫu** và một stratified split chung cho bốn
                mô hình. Mục tiêu là đo accuracy, macro-F1, thời gian fit và thời gian dự đoán
                trên toàn bộ test set. Accelerator khuyến nghị: **None (CPU)** vì bốn estimator
                scikit-learn này đều chạy CPU; bật GPU Kaggle không làm chúng nhanh hơn.

                SVM RBF và KNN có thể chạy rất lâu trên Covertype. Notebook lưu checkpoint JSON
                sau từng mô hình, dự đoán theo batch và đặt SVM cuối cùng. Không lưu model binary
                để tránh output ZIP quá lớn.
                """
            ),
            code(COVERTYPE_IMPORTS),
            code(COVERTYPE_SETUP),
            code(COVERTYPE_DATA),
            markdown(
                r"""
                ## Protocol

                Tất cả mô hình dùng cùng train/test indices (`random_state=42`, test 20%).
                Accuracy/F1 được tính trên toàn bộ test set. Generalization gap dùng một mẫu
                train stratified cố định 10.000 dòng để tránh thêm một lượt dự đoán 464.809 dòng
                rất tốn thời gian cho KNN/SVM; phạm vi này được ghi rõ trong JSON.
                """
            ),
            code(COVERTYPE_BENCHMARK),
            markdown("## 1. Decision Tree"),
            code('run_and_checkpoint("Decision Tree")'),
            markdown("## 2. Random Forest"),
            code('run_and_checkpoint("Random Forest")'),
            markdown("## 3. KNN (lazy learner)"),
            code('run_and_checkpoint("KNN")'),
            markdown("## 4. SVM RBF"),
            code('run_and_checkpoint("SVM (RBF)")'),
            markdown(
                r"""
                ## Tổng hợp và tải output

                Khi diễn giải tốc độ, không dùng riêng thời gian `fit` để kết luận KNN nhanh:
                KNN là lazy learner nên phần lớn chi phí chuyển sang lúc prediction. Hãy so sánh
                cả `training_seconds`, `prediction_seconds` và `total_model_seconds`.
                """
            ),
            code(COVERTYPE_SUMMARY),
        ]
    )


COMPARISON_IMPORTS = r"""
import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from IPython.display import FileLink, display
except ImportError:
    FileLink = None

    def display(value):
        print(value)

sns.set_theme(style="whitegrid")
"""


COMPARISON_CODE = r"""
RANDOM_STATE = 42
TEST_SIZE = 0.20
KAGGLE_WORKING = Path("/kaggle/working")
RUN_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd()
FIGURES_DIR = RUN_ROOT / "figures"
RESULTS_DIR = RUN_ROOT / "results"
for directory in (FIGURES_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

EXPERIMENT_ID = "three_dataset_model_comparison"
EXPECTED_RESULTS = {
    "dt_letter_baseline.json",
    "dt_digits_baseline.json",
    "dt_covertype_scalability.json",
    "gpu_three_dataset_three_model_benchmark.json",
}


def search_roots():
    roots = [Path("/kaggle/input"), Path.cwd() / "results", Path.cwd()]
    return [root for root in roots if root.exists()]


def read_result_json(filename):
    for root in search_roots():
        direct_matches = sorted(root.rglob(filename))
        if direct_matches:
            path = direct_matches[0]
            print(f"Loaded {filename}: {path}")
            return json.loads(path.read_text(encoding="utf-8"))
    for root in search_roots():
        for archive_path in sorted(root.rglob("*.zip")):
            try:
                with ZipFile(archive_path) as archive:
                    member = next(
                        (name for name in archive.namelist() if name.endswith(filename)), None
                    )
                    if member:
                        print(f"Loaded {filename}: {archive_path}!{member}")
                        return json.loads(archive.read(member).decode("utf-8"))
            except (OSError, ValueError):
                continue
    raise FileNotFoundError(
        f"Missing {filename}. Add Input chứa bốn output ZIP/JSON trước khi Run All."
    )


loaded = {name: read_result_json(name) for name in EXPECTED_RESULTS}

three_model_required = {"Random Forest", "KNN", "SVM (RBF)"}
three_model_result = loaded["gpu_three_dataset_three_model_benchmark.json"]
three_model_missing = []
for dataset_name in ("letter_recognition", "handwritten_digits", "covertype"):
    available_models = set(three_model_result.get("datasets", {}).get(dataset_name, {}))
    missing_models = three_model_required - available_models
    if missing_models:
        three_model_missing.extend(
            f"{dataset_name}: {model_name}" for model_name in sorted(missing_models)
        )
if three_model_missing:
    raise ValueError(
        "Three-dataset benchmark is incomplete. Missing completed models: "
        + ", ".join(three_model_missing)
        + ". Re-run notebook 05 and use its final output ZIP."
    )
"""


COMPARISON_NORMALIZE = r"""
records = []
for filename, result in loaded.items():
    if filename.startswith("dt_"):
        metrics = result["metrics"]
        records.append(
            {
                "dataset": result["dataset"],
                "model": "Decision Tree",
                "train_accuracy": metrics["train_accuracy"],
                "test_accuracy": metrics["test_accuracy"],
                "f1_macro": metrics["f1_macro"],
                "error_rate": metrics["error_rate"],
                "generalization_gap": metrics["train_accuracy"] - metrics["test_accuracy"],
                "training_seconds": metrics["training_seconds"],
                "prediction_seconds": metrics["prediction_seconds"],
                "total_model_seconds": (
                    metrics["training_seconds"] + metrics["prediction_seconds"]
                ),
                "dataset_scope": "full",
                "train_accuracy_scope": "entire_train_set",
                "learning_strategy": "eager",
            }
        )
    elif filename == "gpu_three_dataset_three_model_benchmark.json":
        for dataset_name, model_results in result["datasets"].items():
            for model_name, metrics in model_results.items():
                records.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "train_accuracy": metrics["train_accuracy"],
                        "test_accuracy": metrics["test_accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "error_rate": metrics["error_rate"],
                        "generalization_gap": metrics["generalization_gap"],
                        "training_seconds": metrics["training_seconds"],
                        "prediction_seconds": metrics["prediction_seconds"],
                        "total_model_seconds": metrics["total_model_seconds"],
                        "dataset_scope": metrics["dataset_scope"],
                        "train_accuracy_scope": (
                            "diagnostic_sample"
                            if metrics["train_accuracy_sample_size"] < metrics["train_samples"]
                            else "entire_train_set"
                        ),
                        "learning_strategy": metrics["learning_strategy"],
                    }
                )
    else:
        for dataset_name, metrics in result["datasets"].items():
            records.append(
                {
                    "dataset": dataset_name,
                    "model": result["model"],
                    "train_accuracy": metrics["train_accuracy"],
                    "test_accuracy": metrics["test_accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "error_rate": metrics["error_rate"],
                    "generalization_gap": metrics["generalization_gap"],
                    "training_seconds": metrics["training_seconds"],
                    "prediction_seconds": metrics["prediction_seconds"],
                    "total_model_seconds": (
                        metrics["training_seconds"] + metrics["prediction_seconds"]
                    ),
                    "dataset_scope": "full",
                    "train_accuracy_scope": "entire_train_set",
                    "learning_strategy": "lazy" if result["model"] == "KNN" else "eager",
                }
            )

comparison = pd.DataFrame(records).sort_values(["dataset", "f1_macro"], ascending=[True, False])
display(comparison.round(4))

best_by_dataset = (
    comparison.loc[comparison.groupby("dataset")["f1_macro"].idxmax()]
    .set_index("dataset")[["model", "test_accuracy", "f1_macro"]]
    .to_dict(orient="index")
)
print("Best by macro-F1:", best_by_dataset)

insights = {}
for dataset_name, frame in comparison.groupby("dataset"):
    best = frame.loc[frame["f1_macro"].idxmax()]
    decision_tree = frame.loc[frame["model"] == "Decision Tree"].iloc[0]
    fastest_train = frame.loc[frame["training_seconds"].idxmin()]
    eager_frame = frame.loc[frame["learning_strategy"] == "eager"]
    fastest_eager_train = eager_frame.loc[eager_frame["training_seconds"].idxmin()]
    fastest_predict = frame.loc[frame["prediction_seconds"].idxmin()]
    fastest_total = frame.loc[frame["total_model_seconds"].idxmin()]
    smallest_gap = frame.loc[frame["generalization_gap"].idxmin()]
    insights[dataset_name] = {
        "best_macro_f1_model": best["model"],
        "best_macro_f1": float(best["f1_macro"]),
        "macro_f1_gain_over_decision_tree": float(
            best["f1_macro"] - decision_tree["f1_macro"]
        ),
        "fastest_training_model": fastest_train["model"],
        "fastest_eager_training_model": fastest_eager_train["model"],
        "fastest_prediction_model": fastest_predict["model"],
        "fastest_fit_plus_prediction_model": fastest_total["model"],
        "smallest_generalization_gap_model": smallest_gap["model"],
    }
display(pd.DataFrame(insights).T)
"""


COMPARISON_PLOTS = r"""
artifact_paths = []

metrics_long = comparison.melt(
    id_vars=["dataset", "model"],
    value_vars=["test_accuracy", "f1_macro"],
    var_name="metric",
    value_name="score",
)
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
for ax, (dataset_name, frame) in zip(axes, metrics_long.groupby("dataset"), strict=True):
    sns.barplot(data=frame, x="model", y="score", hue="metric", ax=ax)
    ax.set(title=dataset_name.replace("_", " ").title(), ylim=(0.0, 1.0), xlabel="")
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Model performance on the shared test splits")
fig.tight_layout()
performance_path = FIGURES_DIR / f"{EXPERIMENT_ID}__performance.png"
fig.savefig(performance_path, dpi=200, bbox_inches="tight")
plt.show()
artifact_paths.append(performance_path)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(data=comparison, x="model", y="generalization_gap", hue="dataset", ax=axes[0])
axes[0].set(title="Generalization gap", xlabel="", ylabel="Train accuracy - test accuracy")
axes[0].tick_params(axis="x", rotation=20)
sns.barplot(
    data=comparison,
    x="dataset",
    y="total_model_seconds",
    hue="model",
    errorbar=None,
    ax=axes[1],
)
axes[1].set(
    title="Fit + full-test prediction (log scale)",
    xlabel="",
    ylabel="Seconds",
    yscale="log",
)
axes[1].tick_params(axis="x", rotation=15)
fig.tight_layout()
tradeoff_path = FIGURES_DIR / f"{EXPERIMENT_ID}__gap_and_runtime.png"
fig.savefig(tradeoff_path, dpi=200, bbox_inches="tight")
plt.show()
artifact_paths.append(tradeoff_path)
"""


COMPARISON_SAVE = r"""
comparison_path = RESULTS_DIR / f"{EXPERIMENT_ID}.csv"
comparison.to_csv(comparison_path, index=False)
summary_path = RESULTS_DIR / f"{EXPERIMENT_ID}.json"
summary = {
    "schema_version": "1.0",
    "experiment_id": EXPERIMENT_ID,
    "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratify": True},
    "selection_metric": "macro-F1 on each dataset's shared test split (reporting only)",
    "timing_note": (
        "KNN is a lazy learner; use fit + full-test prediction for end-to-end comparison. "
        "Covertype generalization gap uses a fixed 10k stratified train diagnostic sample."
    ),
    "best_by_dataset": best_by_dataset,
    "insights": insights,
    "records": json.loads(comparison.to_json(orient="records")),
    "created_at_utc": datetime.now(UTC).isoformat(),
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
artifact_paths.extend([comparison_path, summary_path])

archive_path = RUN_ROOT / f"{EXPERIMENT_ID}__outputs.zip"
with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
    for artifact_path in artifact_paths:
        archive.write(artifact_path, artifact_path.relative_to(RUN_ROOT))

print(f"Created ZIP: {archive_path}")
if FileLink is not None:
    display(FileLink(str(archive_path)))
"""


def comparison_notebook() -> dict[str, object]:
    return notebook(
        [
            markdown(
                r"""
                # Model comparison - Letter, Digits và Covertype

                Tổng hợp Decision Tree, Random Forest, SVM và KNN trên cả ba dataset. Trong từng
                dataset, bốn mô hình dùng cùng train/test split. Notebook chỉ đọc result JSON;
                không huấn luyện lại model. Trên Kaggle, hãy **Add Input** chứa bốn output ZIP
                hoặc các JSON đã giải nén.
                """
            ),
            code(COMPARISON_IMPORTS),
            code(COMPARISON_CODE),
            code(COMPARISON_NORMALIZE),
            markdown(
                r"""
                ## Nguyên tắc đọc kết quả

                Không chọn model chỉ theo accuracy. So sánh đồng thời macro-F1, train-test gap,
                thời gian huấn luyện và suy luận. Với KNN cần ưu tiên tổng `fit + prediction` vì
                đây là lazy learner. Kết quả test dùng để báo cáo, không dùng để tuning.
                """
            ),
            code(COMPARISON_PLOTS),
            code(COMPARISON_SAVE),
        ]
    )


def main() -> None:
    from generate_gpu_benchmark_notebook import main as generate_gpu_benchmark_notebook

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    generate_gpu_benchmark_notebook()

    old_covertype_path = BENCHMARK_DIR / "07_covertype_four_model_benchmark.ipynb"
    if old_covertype_path.exists():
        old_covertype_path.unlink()

    comparison_path = COMPARISON_DIR / "06_three_dataset_model_comparison.ipynb"
    comparison_path.write_text(
        json.dumps(comparison_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {comparison_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
