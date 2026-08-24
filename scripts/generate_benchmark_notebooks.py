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

EXPERIMENT_ID = "letter_digits_model_comparison"
EXPECTED_RESULTS = {
    "dt_letter_baseline.json",
    "dt_digits_baseline.json",
    "rf_letter_digits_benchmark.json",
    "svm_letter_digits_benchmark.json",
    "knn_letter_digits_benchmark.json",
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
        f"Missing {filename}. Add Input chứa năm output ZIP/JSON trước khi Run All."
    )


loaded = {name: read_result_json(name) for name in EXPECTED_RESULTS}
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
    fastest_predict = frame.loc[frame["prediction_seconds"].idxmin()]
    smallest_gap = frame.loc[frame["generalization_gap"].idxmin()]
    insights[dataset_name] = {
        "best_macro_f1_model": best["model"],
        "best_macro_f1": float(best["f1_macro"]),
        "macro_f1_gain_over_decision_tree": float(
            best["f1_macro"] - decision_tree["f1_macro"]
        ),
        "fastest_training_model": fastest_train["model"],
        "fastest_prediction_model": fastest_predict["model"],
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
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
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

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=comparison, x="model", y="generalization_gap", hue="dataset", ax=axes[0])
axes[0].set(title="Generalization gap", xlabel="", ylabel="Train accuracy - test accuracy")
axes[0].tick_params(axis="x", rotation=20)
timing_long = comparison.melt(
    id_vars=["dataset", "model"],
    value_vars=["training_seconds", "prediction_seconds"],
    var_name="stage",
    value_name="seconds",
)
timing_long["model_dataset"] = (
    timing_long["model"]
    + "\n"
    + timing_long["dataset"].map(
        {"letter_recognition": "Letter", "handwritten_digits": "Digits"}
    )
)
sns.barplot(
    data=timing_long,
    x="model_dataset",
    y="seconds",
    hue="stage",
    errorbar=None,
    ax=axes[1],
)
axes[1].set(title="Runtime (log scale)", xlabel="", ylabel="Seconds", yscale="log")
axes[1].tick_params(axis="x", rotation=25)
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
    "selection_metric": "macro-F1 on the shared test split (reporting only)",
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
                # Model comparison - Letter Recognition và Handwritten Digits

                Tổng hợp Decision Tree, Random Forest, SVM và KNN trên cùng train/test
                protocol. Notebook chỉ đọc result JSON; không huấn luyện lại model. Trên Kaggle,
                hãy **Add Input** chứa năm output ZIP hoặc các JSON đã giải nén.
                """
            ),
            code(COMPARISON_IMPORTS),
            code(COMPARISON_CODE),
            code(COMPARISON_NORMALIZE),
            markdown(
                r"""
                ## Nguyên tắc đọc kết quả

                Không chọn model chỉ theo accuracy. So sánh đồng thời macro-F1, train-test gap,
                thời gian huấn luyện và suy luận. Kết quả test dùng để báo cáo, không dùng để tuning.
                """
            ),
            code(COMPARISON_PLOTS),
            code(COMPARISON_SAVE),
        ]
    )


def main() -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "filename": "04_random_forest_letter_digits.ipynb",
            "title": "Random Forest benchmark - Letter và Digits",
            "experiment_id": "rf_letter_digits_benchmark",
            "model_name": "Random Forest",
            "model_import": "from sklearn.ensemble import RandomForestClassifier\n",
            "estimator_code": dedent(
                """
                    return RandomForestClassifier(
                        n_estimators=300,
                        criterion="gini",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )
                """
            ).rstrip(),
            "scaling_note": "Random Forest không cần scaling.",
            "model_config": {
                "n_estimators": 300,
                "criterion": "gini",
                "random_state": 42,
                "n_jobs": -1,
                "scaling": False,
            },
        },
        {
            "filename": "05_svm_letter_digits.ipynb",
            "title": "SVM benchmark - Letter và Digits",
            "experiment_id": "svm_letter_digits_benchmark",
            "model_name": "SVM (RBF)",
            "model_import": dedent(
                """
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import SVC
                """
            ),
            "estimator_code": dedent(
                """
                    return Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("model", SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=2048)),
                        ]
                    )
                """
            ).rstrip(),
            "scaling_note": "SVM dùng StandardScaler.",
            "model_config": {
                "pipeline": "StandardScaler -> SVC",
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "cache_size_mb": 2048,
            },
        },
        {
            "filename": "06_knn_letter_digits.ipynb",
            "title": "KNN benchmark - Letter và Digits",
            "experiment_id": "knn_letter_digits_benchmark",
            "model_name": "KNN",
            "model_import": dedent(
                """
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                """
            ),
            "estimator_code": dedent(
                """
                    return Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("model", KNeighborsClassifier(n_neighbors=5, weights="uniform", n_jobs=-1)),
                        ]
                    )
                """
            ).rstrip(),
            "scaling_note": "KNN dùng StandardScaler để khoảng cách không bị lệch thang đo.",
            "model_config": {
                "pipeline": "StandardScaler -> KNeighborsClassifier",
                "n_neighbors": 5,
                "weights": "uniform",
                "n_jobs": -1,
            },
        },
    ]
    for spec in specs:
        path = BENCHMARK_DIR / spec["filename"]
        path.write_text(
            json.dumps(benchmark_notebook(spec), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")

    comparison_path = COMPARISON_DIR / "07_letter_digits_model_comparison.ipynb"
    comparison_path.write_text(
        json.dumps(comparison_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {comparison_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
