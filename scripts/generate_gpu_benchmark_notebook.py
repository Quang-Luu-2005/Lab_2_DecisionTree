"""Generate the GPU-only Kaggle benchmark notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "benchmark_models"
    / ("05_three_dataset_three_model_benchmark.ipynb")
)


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
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isGpuEnabled": True,
                "internet": False,
            },
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


GPU_CODE = r"""
import gc
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

try:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier
    from cuml.neighbors import KNeighborsClassifier
    from cuml.preprocessing import StandardScaler
    from cuml.svm import SVC
except ImportError as error:
    raise RuntimeError(
        "RAPIDS/cuML is required. In Kaggle, select a GPU accelerator and use an image "
        "with RAPIDS installed; this notebook intentionally has no CPU fallback."
    ) from error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import fetch_covtype, load_digits
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from IPython.display import FileLink, display
except ImportError:
    FileLink = None

    def display(value):
        print(value)


RANDOM_STATE = 42
TEST_SIZE = 0.20
PREDICTION_BATCH_SIZE = int(os.getenv("GPU_PREDICTION_BATCH_SIZE", "4096"))
TRAIN_DIAGNOSTIC_SIZE = int(os.getenv("GPU_TRAIN_DIAGNOSTIC_SIZE", "10000"))
EXPERIMENT_ID = "gpu_three_dataset_three_model_benchmark"
PIPELINE_STARTED = time.perf_counter()
KAGGLE_WORKING = Path("/kaggle/working")
RUN_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd()
FIGURES_DIR = RUN_ROOT / "figures"
RESULTS_DIR = RUN_ROOT / "results"
for directory in (FIGURES_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

if cp.cuda.runtime.getDeviceCount() < 1:
    raise RuntimeError("No CUDA GPU detected. Select Kaggle GPU accelerator before Run All.")
GPU_DEVICE = cp.cuda.runtime.getDeviceProperties(0)["name"]
if isinstance(GPU_DEVICE, bytes):
    GPU_DEVICE = GPU_DEVICE.decode()
GPU_MEMORY_MB = int(cp.cuda.runtime.memGetInfo()[1] / 1024**2)
print({"python": platform.python_version(), "sklearn": sklearn.__version__})
print({"cuda_device": GPU_DEVICE, "free_gpu_memory_mb": GPU_MEMORY_MB})
print("All model fit/predict operations in this notebook use cuML on GPU.")

LETTER_FEATURES = [
    "x_box", "y_box", "width", "high", "onpix", "x_bar", "y_bar", "x2bar",
    "y2bar", "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx",
]
MODEL_NAMES = ["Random Forest", "SVM (RBF)", "KNN"]


def sync_gpu():
    cp.cuda.Stream.null.synchronize()


def to_host(values):
    if isinstance(values, cp.ndarray):
        return cp.asnumpy(values)
    return np.asarray(values)


def find_letter_split():
    roots = [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent]
    required = set(LETTER_FEATURES + ["letter"])
    for root in roots:
        if not root.exists():
            continue
        for train_path in sorted(root.rglob("train.csv")):
            test_path = train_path.with_name("test.csv")
            if not test_path.exists():
                continue
            try:
                train_columns = set(pd.read_csv(train_path, nrows=1).columns)
                test_columns = set(pd.read_csv(test_path, nrows=1).columns)
            except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
                continue
            if required.issubset(train_columns) and required.issubset(test_columns):
                return train_path, test_path
    raise FileNotFoundError("Add canonical Letter train.csv/test.csv as Kaggle Input.")


def find_covertype_csv():
    roots = [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent]
    for root in roots:
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*.csv")):
            try:
                columns = pd.read_csv(candidate, nrows=0).columns.tolist()
            except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
                continue
            if len(columns) == 55 and {"cover_type", "Cover_Type", "target", "label"}.intersection(columns):
                return candidate
    return None


def normalize_covertype(frame):
    for column in ("cover_type", "Cover_Type", "target", "label"):
        if column in frame.columns:
            return frame.rename(columns={column: "cover_type"})
    raise ValueError("Covertype CSV must contain cover_type, Cover_Type, target or label.")


def load_datasets():
    started = time.perf_counter()
    letter_train_path, letter_test_path = find_letter_split()
    letter_train = pd.read_csv(letter_train_path)
    letter_test = pd.read_csv(letter_test_path)
    letter = {
        "X_train": letter_train[LETTER_FEATURES].to_numpy(dtype=np.float32),
        "y_train": letter_train["letter"].to_numpy(),
        "X_test": letter_test[LETTER_FEATURES].to_numpy(dtype=np.float32),
        "y_test": letter_test["letter"].to_numpy(),
        "source": str(letter_train_path.parent),
        "scope": "canonical_preprocessed_split",
    }

    digits = load_digits()
    digits_train, digits_test, digits_y_train, digits_y_test = train_test_split(
        digits.data.astype(np.float32),
        digits.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=digits.target,
    )
    digit_dataset = {
        "X_train": digits_train,
        "y_train": digits_y_train,
        "X_test": digits_test,
        "y_test": digits_y_test,
        "source": "sklearn.datasets.load_digits",
        "scope": "full_dataset_split",
    }

    covertype_path = find_covertype_csv()
    if covertype_path is None:
        bundle = fetch_covtype(as_frame=True, data_home=RUN_ROOT / ".cache" / "scikit_learn_data")
        covertype_frame = bundle.data.copy()
        covertype_frame["cover_type"] = bundle.target.astype("int32").to_numpy()
        covertype_source = "sklearn.datasets.fetch_covtype() / UCI Covertype"
    else:
        covertype_frame = normalize_covertype(pd.read_csv(covertype_path))
        covertype_source = str(covertype_path)
    covertype_frame["cover_type"] = pd.to_numeric(
        covertype_frame["cover_type"], errors="raise"
    ).astype("int32")
    assert covertype_frame.shape[1] == 55
    assert covertype_frame["cover_type"].nunique() == 7
    assert not covertype_frame.isna().any().any()
    covertype_features = [column for column in covertype_frame.columns if column != "cover_type"]
    covertype_train, covertype_test = train_test_split(
        covertype_frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=covertype_frame["cover_type"],
    )
    covertype_dataset = {
        "X_train": covertype_train[covertype_features].to_numpy(dtype=np.float32),
        "y_train": covertype_train["cover_type"].to_numpy(dtype=np.int32),
        "X_test": covertype_test[covertype_features].to_numpy(dtype=np.float32),
        "y_test": covertype_test["cover_type"].to_numpy(dtype=np.int32),
        "source": covertype_source,
        "scope": "full_dataset_split",
    }
    return {
        "letter_recognition": letter,
        "handwritten_digits": digit_dataset,
        "covertype": covertype_dataset,
    }, time.perf_counter() - started


def prepare_dataset(parts):
    encoder = LabelEncoder().fit(parts["y_train"])
    y_train = encoder.transform(parts["y_train"]).astype(np.int32)
    y_test = encoder.transform(parts["y_test"]).astype(np.int32)
    X_train = cp.asarray(parts["X_train"], dtype=cp.float32)
    X_test = cp.asarray(parts["X_test"], dtype=cp.float32)
    y_train_gpu = cp.asarray(y_train)
    return X_train, y_train_gpu, X_test, y_train, y_test, encoder


def make_estimator(model_name):
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            split_criterion="gini",
            max_depth=None,
            max_features="sqrt",
            n_bins=128,
            random_state=RANDOM_STATE,
            n_streams=4,
            output_type="numpy",
        ), None
    if model_name == "SVM (RBF)":
        return SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            cache_size=4096,
            decision_function_shape="ovr",
            output_type="numpy",
        ), "scale"
    if model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            algorithm="brute",
            output_type="numpy",
        ), "scale"
    raise KeyError(model_name)


def transform_for_model(X_train, X_test, scaling):
    if scaling is None:
        return X_train, X_test, None
    scaler = StandardScaler(output_type="cupy")
    sync_gpu()
    scaler.fit(X_train)
    sync_gpu()
    return scaler.transform(X_train), scaler.transform(X_test), scaler


def predict_in_batches(model, X, batch_size=PREDICTION_BATCH_SIZE):
    predictions = []
    started = time.perf_counter()
    for start in range(0, X.shape[0], batch_size):
        stop = min(start + batch_size, X.shape[0])
        predictions.append(to_host(model.predict(X[start:stop])).reshape(-1))
    sync_gpu()
    return np.concatenate(predictions), time.perf_counter() - started


def calculate_metrics(y_train, train_prediction, y_test, test_prediction):
    train_accuracy = accuracy_score(y_train, train_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)
    return {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "error_rate": float(1.0 - test_accuracy),
        "precision_macro": float(precision_score(y_test, test_prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, test_prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, test_prediction, average="macro")),
        "generalization_gap": float(train_accuracy - test_accuracy),
    }


def evaluate_model(dataset_name, parts, model_name):
    X_train, y_train_gpu, X_test, y_train, y_test, encoder = prepare_dataset(parts)
    estimator, scaling = make_estimator(model_name)
    X_train_model, X_test_model, scaler = transform_for_model(X_train, X_test, scaling)
    sync_gpu()
    fit_started = time.perf_counter()
    estimator.fit(X_train_model, y_train_gpu)
    sync_gpu()
    training_seconds = time.perf_counter() - fit_started

    diagnostic_size = min(TRAIN_DIAGNOSTIC_SIZE, X_train_model.shape[0])
    diagnostic_X = X_train_model[:diagnostic_size]
    diagnostic_y = y_train[:diagnostic_size]
    train_prediction_encoded, diagnostic_prediction_seconds = predict_in_batches(
        estimator, diagnostic_X
    )
    test_prediction_encoded, prediction_seconds = predict_in_batches(estimator, X_test_model)
    train_prediction = encoder.inverse_transform(train_prediction_encoded.astype(int))
    test_prediction = encoder.inverse_transform(test_prediction_encoded.astype(int))
    metrics = calculate_metrics(
        parts["y_train"][:diagnostic_size], train_prediction, parts["y_test"], test_prediction
    )
    metrics.update(
        {
            "status": "completed",
            "dataset": dataset_name,
            "model": model_name,
            "compute_device": "GPU",
            "gpu_name": GPU_DEVICE,
            "learning_strategy": "lazy" if model_name == "KNN" else "eager",
            "train_accuracy_sample_size": int(diagnostic_size),
            "training_seconds": float(training_seconds),
            "prediction_seconds": float(prediction_seconds),
            "diagnostic_prediction_seconds": float(diagnostic_prediction_seconds),
            "total_model_seconds": float(training_seconds + prediction_seconds),
            "train_samples": int(len(parts["X_train"])),
            "test_samples": int(len(parts["X_test"])),
            "features": int(parts["X_train"].shape[1]),
            "dataset_scope": parts["scope"],
        }
    )
    del estimator, scaler, X_train_model, X_test_model, X_train, X_test, y_train_gpu
    gc.collect()
    return metrics, test_prediction, encoder.classes_


datasets, data_loading_seconds = load_datasets()
evaluations = []
figure_paths = []
for dataset_name, parts in datasets.items():
    print(f"\n===== {dataset_name} =====")
    for model_name in MODEL_NAMES:
        metrics, prediction, labels = evaluate_model(dataset_name, parts, model_name)
        evaluations.append(metrics)
        print(
            model_name,
            "test_accuracy=",
            round(metrics["test_accuracy"], 4),
            "macro_f1=",
            round(metrics["f1_macro"], 4),
            "training_seconds=",
            round(metrics["training_seconds"], 2),
            "prediction_seconds=",
            round(metrics["prediction_seconds"], 2),
        )
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay.from_predictions(
            parts["y_test"], prediction, labels=labels, cmap="Blues", colorbar=False, values_format="d", ax=ax
        )
        ax.set_title(f"{dataset_name.replace('_', ' ').title()} - {model_name}")
        fig.tight_layout()
        path = FIGURES_DIR / (
            f"{EXPERIMENT_ID}__{dataset_name}__{model_name.split('(')[0].strip().lower().replace(' ', '_')}__confusion_matrix.png"
        )
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(path)

summary = pd.DataFrame(evaluations)
display(
    summary[
        [
            "dataset", "model", "compute_device", "gpu_name", "test_accuracy", "f1_macro",
            "generalization_gap", "training_seconds", "prediction_seconds",
        ]
    ].round(4)
)

fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5), squeeze=False)
for ax, (dataset_name, frame) in zip(axes[0], summary.groupby("dataset"), strict=True):
    plot_frame = frame.melt(
        id_vars=["model"], value_vars=["test_accuracy", "f1_macro"], var_name="metric", value_name="score"
    )
    sns.barplot(data=plot_frame, x="model", y="score", hue="metric", ax=ax)
    ax.set(title=dataset_name.replace("_", " ").title(), xlabel="", ylabel="Score", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("GPU benchmark: Random Forest, SVM and KNN on three datasets")
fig.tight_layout()
performance_path = FIGURES_DIR / f"{EXPERIMENT_ID}__performance.png"
fig.savefig(performance_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
figure_paths.append(performance_path)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(data=summary, x="dataset", y="generalization_gap", hue="model", errorbar=None, ax=axes[0])
axes[0].set(title="Generalization gap", xlabel="", ylabel="Train accuracy - test accuracy")
axes[0].tick_params(axis="x", rotation=20)
sns.barplot(data=summary, x="dataset", y="total_model_seconds", hue="model", errorbar=None, ax=axes[1])
axes[1].set(title="GPU fit + full-test prediction", xlabel="", ylabel="Seconds", yscale="log")
axes[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
runtime_path = FIGURES_DIR / f"{EXPERIMENT_ID}__gap_and_runtime.png"
fig.savefig(runtime_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
figure_paths.append(runtime_path)

summary_path = RESULTS_DIR / f"{EXPERIMENT_ID}__summary.csv"
summary.to_csv(summary_path, index=False)
pipeline_seconds = time.perf_counter() - PIPELINE_STARTED
dataset_results = {dataset_name: {} for dataset_name in datasets}
for record in evaluations:
    dataset_results[record["dataset"]][record["model"]] = record
result = {
    "schema_version": "1.0",
    "experiment_id": EXPERIMENT_ID,
    "model": "cuML Random Forest, SVM (RBF), KNN",
    "dataset": "multiple",
    "datasets": dataset_results,
    "dataset_names": list(datasets),
    "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratify": True},
    "metrics": evaluations,
    "data_loading_seconds": float(data_loading_seconds),
    "pipeline_seconds": float(pipeline_seconds),
    "hardware": {
        "compute_device": "GPU",
        "gpu_name": GPU_DEVICE,
        "free_gpu_memory_at_start_mb": GPU_MEMORY_MB,
        "cuda_runtime": str(cp.cuda.runtime.runtimeGetVersion()),
    },
    "artifacts": {
        "figure_paths": [str(path.relative_to(RUN_ROOT)) for path in figure_paths],
        "table_paths": [str(summary_path.relative_to(RUN_ROOT))],
    },
    "notes": (
        "GPU-only benchmark using direct cuML estimators. Data transfer, fit and prediction are "
        "timed separately at the model boundary; Covertype train accuracy uses a fixed diagnostic "
        "sample because full KNN train prediction is unnecessarily expensive. No CPU fallback is allowed."
    ),
    "created_at_utc": datetime.now(UTC).isoformat(),
}
result_path = RESULTS_DIR / f"{EXPERIMENT_ID}.json"
result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
artifact_paths = [summary_path, result_path, *figure_paths]
archive_path = RUN_ROOT / f"{EXPERIMENT_ID}__outputs.zip"
with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
    for artifact_path in sorted(set(artifact_paths)):
        archive.write(artifact_path, artifact_path.relative_to(RUN_ROOT))
print(f"Pipeline: {pipeline_seconds:.2f}s")
print(f"Created ZIP ({archive_path.stat().st_size / 1024**2:.1f} MB): {archive_path}")
if FileLink is not None:
    display(FileLink(str(archive_path)))
"""


def make_notebook() -> dict[str, object]:
    return notebook(
        [
            markdown(
                """
                # GPU benchmark — three datasets, three models

                Notebook này dành cho **Kaggle GPU**, không chạy CPU fallback. Random Forest,
                SVM (RBF) và KNN đều dùng trực tiếp RAPIDS/cuML; các vòng lặp batch chỉ điều phối
                các kernel GPU. Nếu không thấy CUDA hoặc cuML, notebook dừng ngay để tránh báo sai
                runtime GPU.

                Dataset selection giữ nguyên: Letter dùng canonical `train.csv`/`test.csv`, Digits
                dùng `load_digits()`, Covertype dùng toàn bộ dữ liệu. Cùng `test_size=0.20`,
                `random_state=42`, `stratify=True`; test chỉ dùng để báo cáo.

                **Kaggle setup:** chọn GPU Accelerator (T4/P100 tuỳ quota), giữ Internet off nếu
                RAPIDS đã có trong image. Kết quả cuối được đóng thành
                `gpu_three_dataset_three_model_benchmark__outputs.zip`.
                """
            ),
            code(GPU_CODE),
            markdown(
                """
                ## Lưu ý về Decision Tree exact

                Notebook này tăng tốc phần benchmark model khác. `DecisionTreeClassifier` đơn và
                Hierarchical Shrinkage exact vẫn có notebook riêng vì RAPIDS hiện không cung cấp
                backend GPU tương đương cho chúng; không thay chúng bằng một model khác trong
                nghiên cứu chính.
                """
            ),
        ]
    )


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(make_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
