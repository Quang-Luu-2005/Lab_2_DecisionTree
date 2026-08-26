"""Generate the Kaggle-ready Hierarchical Shrinkage benchmark notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "decision_tree"
    / ("04_hierarchical_shrinkage_three_dataset_benchmark.ipynb")
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
            "kaggle": {"accelerator": "none", "isGpuEnabled": False, "internet": True},
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


NOTEBOOK_CODE = r"""
import gc
import json
import os
import platform
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from shutil import which
from zipfile import ZIP_DEFLATED, ZipFile

try:
    from imodels import HSTreeClassifier
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imodels==3.0.0"])
    from imodels import HSTreeClassifier

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from IPython.display import FileLink, display
except ImportError:
    FileLink = None

    def display(value):
        print(value)


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
DATASET_SELECTION = "all"
MAX_TUNING_SAMPLES = 0
CV_FOLDS = int(os.getenv("HS_CV_FOLDS", str(CV_FOLDS)))
DATASET_SELECTION = os.getenv("HS_DATASET", DATASET_SELECTION).strip().lower()
MAX_TUNING_SAMPLES = int(os.getenv("HS_MAX_TUNING_SAMPLES", str(MAX_TUNING_SAMPLES)))
EXPERIMENT_ID = "dt_hierarchical_shrinkage"
PIPELINE_STARTED = time.perf_counter()
KAGGLE_WORKING = Path("/kaggle/working")
RUN_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd()
FIGURES_DIR = RUN_ROOT / "figures"
RESULTS_DIR = RUN_ROOT / "results"
for directory in (FIGURES_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

LETTER_FEATURES = [
    "x_box", "y_box", "width", "high", "onpix", "x_bar", "y_bar", "x2bar",
    "y2bar", "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx",
]
PREPRUNE_CANDIDATES = [
    {"max_depth": 8, "min_samples_leaf": 1},
    {"max_depth": 16, "min_samples_leaf": 1},
    {"max_depth": None, "min_samples_leaf": 5},
    {"max_depth": None, "min_samples_leaf": 20},
]
HS_LAMBDA_VALUES = [0.0, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]


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
    nvidia_smi = which("nvidia-smi")
    if nvidia_smi:
        query = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if query.returncode == 0:
            for line in query.stdout.strip().splitlines():
                try:
                    name, memory_mb, driver = [part.strip() for part in line.split(",", 2)]
                    hardware["gpus"].append(
                        {
                            "name": name,
                            "memory_mb": int(float(memory_mb)),
                            "driver_version": driver,
                        }
                    )
                except ValueError:
                    continue
            hardware["gpu_available"] = bool(hardware["gpus"])
    return hardware


HARDWARE = detect_hardware()
print({"python": platform.python_version(), "sklearn": sklearn.__version__})
print("Hardware:", HARDWARE)
print("Hierarchical Shrinkage and Decision Tree compute device: CPU")


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
    raise FileNotFoundError(
        "Missing canonical Letter train.csv/test.csv. Add the processed Letter dataset "
        "as a Kaggle Input or run scripts/prepare_letter_recognition.py locally."
    )


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
            target_like = {"cover_type", "Cover_Type", "target", "label"}
            if len(columns) == 55 and target_like.intersection(columns):
                return candidate
    return None


def normalize_covertype(frame):
    for column in ("cover_type", "Cover_Type", "target", "label"):
        if column in frame.columns:
            return frame.rename(columns={column: "cover_type"})
    raise ValueError("Covertype CSV must contain cover_type, Cover_Type, target or label.")


def load_datasets():
    started = time.perf_counter()
    requested = choose_dataset_names()
    datasets = {}
    if "letter_recognition" in requested:
        letter_train_path, letter_test_path = find_letter_split()
        letter_train = pd.read_csv(letter_train_path)
        letter_test = pd.read_csv(letter_test_path)
        datasets["letter_recognition"] = {
            "X_train": letter_train[LETTER_FEATURES],
            "y_train": letter_train["letter"],
            "X_test": letter_test[LETTER_FEATURES],
            "y_test": letter_test["letter"],
            "features": LETTER_FEATURES,
            "source": str(letter_train_path.parent),
            "scope": "canonical_preprocessed_split",
        }
    if "handwritten_digits" in requested:
        digits_bundle = load_digits(as_frame=True)
        digits_frame = digits_bundle.frame.rename(columns={"target": "label"})
        digits_train, digits_test = train_test_split(
            digits_frame,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=digits_frame["label"],
        )
        digit_features = [column for column in digits_train.columns if column != "label"]
        datasets["handwritten_digits"] = {
            "X_train": digits_train[digit_features],
            "y_train": digits_train["label"].astype("int64"),
            "X_test": digits_test[digit_features],
            "y_test": digits_test["label"].astype("int64"),
            "features": digit_features,
            "source": "sklearn.datasets.load_digits",
            "scope": "full_dataset_split",
        }
    if "covertype" in requested:
        covertype_path = find_covertype_csv()
        if covertype_path is None:
            bundle = fetch_covtype(
                as_frame=True,
                data_home=RUN_ROOT / ".cache" / "scikit_learn_data",
            )
            covertype_frame = bundle.data.copy()
            covertype_frame["cover_type"] = bundle.target.astype("int64").to_numpy()
            covertype_source = "sklearn.datasets.fetch_covtype() / UCI Covertype"
        else:
            covertype_frame = normalize_covertype(pd.read_csv(covertype_path))
            covertype_source = str(covertype_path)
        covertype_frame["cover_type"] = pd.to_numeric(
            covertype_frame["cover_type"], errors="raise"
        ).astype("int64")
        assert covertype_frame.shape[1] == 55
        assert covertype_frame["cover_type"].nunique() == 7
        assert not covertype_frame.isna().any().any()
        covertype_features = [
            column for column in covertype_frame.columns if column != "cover_type"
        ]
        covertype_X_train, covertype_X_test, covertype_y_train, covertype_y_test = train_test_split(
            covertype_frame[covertype_features],
            covertype_frame["cover_type"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=covertype_frame["cover_type"],
        )
        datasets["covertype"] = {
            "X_train": covertype_X_train,
            "y_train": covertype_y_train,
            "X_test": covertype_X_test,
            "y_test": covertype_y_test,
            "features": covertype_features,
            "source": covertype_source,
            "scope": "full_dataset_split",
        }
    return datasets, time.perf_counter() - started


def choose_dataset_names():
    all_names = ["letter_recognition", "handwritten_digits", "covertype"]
    if DATASET_SELECTION in {"all", ""}:
        return all_names
    requested = [name.strip() for name in DATASET_SELECTION.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(all_names))
    if unknown:
        raise ValueError(f"Unknown HS_DATASET value(s): {unknown}; use {all_names} or all.")
    return requested


selected_dataset_names = choose_dataset_names()
datasets, data_loading_seconds = load_datasets()
print("Selected datasets:", selected_dataset_names)
for dataset_name, parts in datasets.items():
    print(
        dataset_name,
        "train/test:",
        parts["X_train"].shape,
        parts["X_test"].shape,
        "classes:",
        parts["y_train"].nunique(),
    )


def make_tuning_data(parts):
    if MAX_TUNING_SAMPLES <= 0 or MAX_TUNING_SAMPLES >= len(parts["X_train"]):
        return parts["X_train"], parts["y_train"], "full_train"
    X_tune, _, y_tune, _ = train_test_split(
        parts["X_train"],
        parts["y_train"],
        train_size=MAX_TUNING_SAMPLES,
        random_state=RANDOM_STATE,
        stratify=parts["y_train"],
    )
    return X_tune, y_tune, f"stratified_train_subset_{MAX_TUNING_SAMPLES}"


def make_cv_splits(X, y):
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(X, y))


def score_predictions(y_true, prediction):
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro")),
    }


def cv_tree_candidates(X, y, candidates, cv_splits):
    rows = []
    for candidate_id, parameters in candidates:
        fold_scores = []
        fit_seconds = []
        score_seconds = []
        for train_indices, validation_indices in cv_splits:
            model = DecisionTreeClassifier(random_state=RANDOM_STATE, **parameters)
            started = time.perf_counter()
            model.fit(X.iloc[train_indices], y.iloc[train_indices])
            fit_seconds.append(time.perf_counter() - started)
            started = time.perf_counter()
            prediction = model.predict(X.iloc[validation_indices])
            score_seconds.append(time.perf_counter() - started)
            fold_scores.append(score_predictions(y.iloc[validation_indices], prediction))
        rows.append(
            {
                "candidate_id": candidate_id,
                "branch": "pre_pruning" if candidate_id.startswith("pre_") else "ccp",
                "parameters": parameters,
                "cv_accuracy_mean": float(np.mean([row["accuracy"] for row in fold_scores])),
                "cv_accuracy_std": float(np.std([row["accuracy"] for row in fold_scores])),
                "cv_f1_macro_mean": float(np.mean([row["f1_macro"] for row in fold_scores])),
                "cv_f1_macro_std": float(np.std([row["f1_macro"] for row in fold_scores])),
                "cv_fit_seconds_mean": float(np.mean(fit_seconds)),
                "cv_score_seconds_mean": float(np.mean(score_seconds)),
            }
        )
    return pd.DataFrame(rows)


def make_ccp_candidates(X, y):
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X, y)
    path = model.cost_complexity_pruning_path(X, y)
    positive = np.asarray(path.ccp_alphas)[np.asarray(path.ccp_alphas) > 0]
    if len(positive) == 0:
        values = np.array([0.0])
    else:
        values = np.unique(
            np.concatenate(([0.0], np.quantile(positive, [0.25, 0.5, 0.75, 0.95, 1.0])))
        )
    return [(f"ccp_{alpha:.12g}", {"ccp_alpha": float(alpha)}) for alpha in values]


def choose_best_cv_row(frame):
    return frame.sort_values(
        ["cv_f1_macro_mean", "cv_accuracy_mean", "cv_f1_macro_std"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()


def cv_hierarchical_shrinkage(X, y, base_parameters, cv_splits):
    score_rows = {value: [] for value in HS_LAMBDA_VALUES}
    fit_rows = {value: [] for value in HS_LAMBDA_VALUES}
    score_time_rows = {value: [] for value in HS_LAMBDA_VALUES}
    for train_indices, validation_indices in cv_splits:
        base_model = DecisionTreeClassifier(random_state=RANDOM_STATE, **base_parameters)
        started = time.perf_counter()
        base_model.fit(X.iloc[train_indices], y.iloc[train_indices])
        base_fit_seconds = time.perf_counter() - started
        for reg_param in HS_LAMBDA_VALUES:
            started = time.perf_counter()
            hs_model = HSTreeClassifier(
                estimator_=deepcopy(base_model),
                reg_param=reg_param,
                shrinkage_scheme_="node_based",
            )
            fit_rows[reg_param].append(base_fit_seconds + time.perf_counter() - started)
            started = time.perf_counter()
            prediction = hs_model.predict(X.iloc[validation_indices])
            score_time_rows[reg_param].append(time.perf_counter() - started)
            score_rows[reg_param].append(
                score_predictions(y.iloc[validation_indices], prediction)
            )
            del hs_model
        del base_model
    rows = []
    for reg_param in HS_LAMBDA_VALUES:
        rows.append(
            {
                "candidate_id": f"lambda_{reg_param:g}",
                "branch": "hierarchical_shrinkage",
                "parameters": {
                    "reg_param": reg_param,
                    "shrinkage_scheme": "node_based",
                    "base_parameters": base_parameters,
                },
                "cv_accuracy_mean": float(np.mean([row["accuracy"] for row in score_rows[reg_param]])),
                "cv_accuracy_std": float(np.std([row["accuracy"] for row in score_rows[reg_param]])),
                "cv_f1_macro_mean": float(np.mean([row["f1_macro"] for row in score_rows[reg_param]])),
                "cv_f1_macro_std": float(np.std([row["f1_macro"] for row in score_rows[reg_param]])),
                "cv_fit_seconds_mean": float(np.mean(fit_rows[reg_param])),
                "cv_score_seconds_mean": float(np.mean(score_time_rows[reg_param])),
            }
        )
    return pd.DataFrame(rows)


def evaluate_final_model(dataset_name, model_name, parts, base_parameters, reg_param=None):
    started = time.perf_counter()
    base_model = DecisionTreeClassifier(random_state=RANDOM_STATE, **base_parameters)
    base_model.fit(parts["X_train"], parts["y_train"])
    base_depth = int(base_model.get_depth())
    base_leaves = int(base_model.get_n_leaves())
    if reg_param is None:
        model = base_model
        hs_structure_unchanged = None
    else:
        model = HSTreeClassifier(
            estimator_=deepcopy(base_model),
            reg_param=float(reg_param),
            shrinkage_scheme_="node_based",
        )
        hs_structure_unchanged = (
            int(model.estimator_.get_depth()) == base_depth
            and int(model.estimator_.get_n_leaves()) == base_leaves
        )
    training_seconds = time.perf_counter() - started
    predict_started = time.perf_counter()
    train_prediction = model.predict(parts["X_train"])
    test_prediction = model.predict(parts["X_test"])
    prediction_seconds = time.perf_counter() - predict_started
    train_scores = score_predictions(parts["y_train"], train_prediction)
    test_scores = score_predictions(parts["y_test"], test_prediction)
    train_accuracy = train_scores["accuracy"]
    test_accuracy = test_scores["accuracy"]
    metrics = {
        "status": "completed",
        "dataset": dataset_name,
        "model": model_name,
        "base_parameters": base_parameters,
        "reg_param": None if reg_param is None else float(reg_param),
        "shrinkage_scheme": None if reg_param is None else "node_based",
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "error_rate": float(1.0 - test_accuracy),
        "precision_macro": float(
            precision_score(parts["y_test"], test_prediction, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(parts["y_test"], test_prediction, average="macro", zero_division=0)
        ),
        "f1_macro": test_scores["f1_macro"],
        "generalization_gap": float(train_accuracy - test_accuracy),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "tree_depth": int(
            model.estimator_.get_depth() if reg_param is not None else model.get_depth()
        ),
        "leaf_count": int(
            model.estimator_.get_n_leaves() if reg_param is not None else model.get_n_leaves()
        ),
        "structure_unchanged_after_hs": hs_structure_unchanged,
        "train_samples": len(parts["X_train"]),
        "test_samples": len(parts["X_test"]),
        "features": parts["X_train"].shape[1],
    }
    return metrics, test_prediction


def json_parameters(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


pipeline_started = time.perf_counter()
all_evaluations = []
all_cv_rows = []
selection_metadata = {}
figure_artifacts = []
for dataset_name, parts in datasets.items():
    print(f"\n===== {dataset_name} =====")
    X_tune, y_tune, tuning_scope = make_tuning_data(parts)
    cv_splits = make_cv_splits(X_tune, y_tune)
    pre_candidates = [
        (f"pre_{index + 1}", parameters)
        for index, parameters in enumerate(PREPRUNE_CANDIDATES)
    ]
    pre_cv = cv_tree_candidates(X_tune, y_tune, pre_candidates, cv_splits)
    pre_best = choose_best_cv_row(pre_cv)
    ccp_candidates = make_ccp_candidates(X_tune, y_tune)
    ccp_cv = cv_tree_candidates(X_tune, y_tune, ccp_candidates, cv_splits)
    ccp_best = choose_best_cv_row(ccp_cv)
    baseline_parameters = {}
    hs_base_cv = cv_hierarchical_shrinkage(X_tune, y_tune, baseline_parameters, cv_splits)
    hs_base_best = choose_best_cv_row(hs_base_cv)
    hs_ccp_cv = cv_hierarchical_shrinkage(
        X_tune, y_tune, ccp_best["parameters"], cv_splits
    )
    hs_ccp_best = choose_best_cv_row(hs_ccp_cv)

    cv_frames = [
        pre_cv.assign(dataset=dataset_name, tuning_scope=tuning_scope),
        ccp_cv.assign(dataset=dataset_name, tuning_scope=tuning_scope),
        hs_base_cv.assign(dataset=dataset_name, tuning_scope=tuning_scope, hs_base="baseline"),
        hs_ccp_cv.assign(dataset=dataset_name, tuning_scope=tuning_scope, hs_base="ccp"),
    ]
    for frame in cv_frames:
        frame["parameters"] = frame["parameters"].map(json_parameters)
    all_cv_rows.append(pd.concat(cv_frames, ignore_index=True))

    selected_pre_parameters = pre_best["parameters"]
    selected_ccp_parameters = ccp_best["parameters"]
    selected_base_lambda = float(hs_base_best["parameters"]["reg_param"])
    selected_ccp_lambda = float(hs_ccp_best["parameters"]["reg_param"])
    selected_models = [
        ("E0 Baseline CART", baseline_parameters, None),
        ("E1 Pre-pruned CART", selected_pre_parameters, None),
        ("E2 CCP-pruned CART", selected_ccp_parameters, None),
        ("E3 HS-DT", baseline_parameters, selected_base_lambda),
        ("E4 CCP+HS", selected_ccp_parameters, selected_ccp_lambda),
    ]
    selection_metadata[dataset_name] = {
        "tuning_scope": tuning_scope,
        "cv_folds": CV_FOLDS,
        "metric": "macro-F1",
        "pre_pruning": pre_best,
        "ccp": ccp_best,
        "hs_on_baseline": hs_base_best,
        "hs_on_ccp": hs_ccp_best,
    }
    dataset_predictions = {}
    for model_name, base_parameters, reg_param in selected_models:
        metrics, test_prediction = evaluate_final_model(
            dataset_name,
            model_name,
            parts,
            base_parameters,
            reg_param,
        )
        all_evaluations.append(metrics)
        dataset_predictions[model_name] = test_prediction
        print(
            model_name,
            "test_accuracy=",
            round(metrics["test_accuracy"], 4),
            "macro_f1=",
            round(metrics["f1_macro"], 4),
            "depth=",
            metrics["tree_depth"],
            "leaves=",
            metrics["leaf_count"],
        )
    labels = np.sort(parts["y_test"].unique())
    for model_name, prediction in dataset_predictions.items():
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay.from_predictions(
            parts["y_test"],
            prediction,
            labels=labels,
            cmap="Blues",
            colorbar=False,
            values_format="d",
            ax=ax,
        )
        ax.set_title(f"{dataset_name.replace('_', ' ').title()} - {model_name}")
        fig.tight_layout()
        figure_path = FIGURES_DIR / (
            f"{EXPERIMENT_ID}__{dataset_name}__{model_name[:2].lower()}__confusion_matrix.png"
        )
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figure_artifacts.append(figure_path)
    lambda_frame = pd.concat(
        [
            hs_base_cv.assign(hs_base="baseline"),
            hs_ccp_cv.assign(hs_base="ccp"),
        ],
        ignore_index=True,
    )
    lambda_frame["reg_param"] = lambda_frame["parameters"].map(
        lambda value: value["reg_param"]
        if isinstance(value, dict)
        else json.loads(value)["reg_param"]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for hs_base, frame in lambda_frame.groupby("hs_base"):
        ax.plot(
            frame["reg_param"],
            frame["cv_f1_macro_mean"],
            marker="o",
            label="CCP base" if hs_base == "ccp" else "Baseline base",
        )
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set(
        title=f"HS regularization selection - {dataset_name}",
        xlabel="lambda",
        ylabel="CV macro-F1",
    )
    ax.legend()
    fig.tight_layout()
    lambda_figure = FIGURES_DIR / f"{EXPERIMENT_ID}__{dataset_name}__lambda_cv.png"
    fig.savefig(lambda_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figure_artifacts.append(lambda_figure)
    del dataset_predictions, cv_splits, X_tune, y_tune
    gc.collect()

evaluations = pd.DataFrame(all_evaluations)
cv_selection = pd.concat(all_cv_rows, ignore_index=True)
display(
    evaluations[
        [
            "dataset", "model", "test_accuracy", "f1_macro", "generalization_gap",
            "training_seconds", "prediction_seconds", "tree_depth", "leaf_count",
            "reg_param", "structure_unchanged_after_hs",
        ]
    ].round(4)
)

fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5), squeeze=False)
for ax, (dataset_name, frame) in zip(axes[0], evaluations.groupby("dataset"), strict=True):
    plot_frame = frame.melt(
        id_vars=["model"],
        value_vars=["test_accuracy", "f1_macro"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=plot_frame, x="model", y="score", hue="metric", ax=ax)
    ax.set(
        title=dataset_name.replace("_", " ").title(),
        xlabel="",
        ylabel="Score",
        ylim=(0, 1),
    )
    ax.tick_params(axis="x", rotation=35)
fig.suptitle("Baseline, structural regularization and Hierarchical Shrinkage")
fig.tight_layout()
performance_figure = FIGURES_DIR / f"{EXPERIMENT_ID}__performance_by_dataset.png"
fig.savefig(performance_figure, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
figure_artifacts.append(performance_figure)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(
    data=evaluations,
    x="leaf_count",
    y="f1_macro",
    hue="model",
    style="dataset",
    s=100,
    ax=axes[0],
)
axes[0].set(xscale="log", title="Macro-F1 vs tree complexity", xlabel="Leaves (log scale)")
sns.barplot(
    data=evaluations,
    x="dataset",
    y="generalization_gap",
    hue="model",
    errorbar=None,
    ax=axes[1],
)
axes[1].set(title="Train-test gap", xlabel="", ylabel="Train accuracy - test accuracy")
axes[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
complexity_figure = FIGURES_DIR / f"{EXPERIMENT_ID}__performance_vs_complexity.png"
fig.savefig(complexity_figure, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
figure_artifacts.append(complexity_figure)

summary_path = RESULTS_DIR / f"{EXPERIMENT_ID}__summary.csv"
evaluations.to_csv(summary_path, index=False)
cv_path = RESULTS_DIR / f"{EXPERIMENT_ID}__cv_selection.csv"
cv_selection.to_csv(cv_path, index=False)
selection_path = RESULTS_DIR / f"{EXPERIMENT_ID}__selection.json"
selection_path.write_text(
    json.dumps(selection_metadata, ensure_ascii=False, indent=2, default=str) + "\n",
    encoding="utf-8",
)
pipeline_seconds = time.perf_counter() - pipeline_started
result = {
    "schema_version": "1.0",
    "experiment_id": EXPERIMENT_ID,
    "dataset": "multiple",
    "datasets": selected_dataset_names,
    "model": "DecisionTreeClassifier + Hierarchical Shrinkage",
    "split": {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": True,
        "test_used_for_tuning": False,
    },
    "selection": selection_metadata,
    "metrics": evaluations.to_dict(orient="records"),
    "data_loading_seconds": float(data_loading_seconds),
    "pipeline_seconds": float(pipeline_seconds),
    "hardware": HARDWARE,
    "artifacts": {
        "figure_paths": [str(path.relative_to(RUN_ROOT)) for path in figure_artifacts],
        "table_paths": [
            str(summary_path.relative_to(RUN_ROOT)),
            str(cv_path.relative_to(RUN_ROOT)),
        ],
        "selection_path": str(selection_path.relative_to(RUN_ROOT)),
    },
    "notes": (
        "E0 baseline, E1 pre-pruning, E2 cost-complexity pruning, E3 HS on baseline and "
        "E4 CCP+HS. Hierarchical Shrinkage is node-based post-hoc regularization from "
        "imodels 3.0.0; it changes prediction values but preserves the underlying tree "
        "structure. Lambda is selected by stratified CV on train only. All estimators use CPU."
    ),
    "created_at_utc": datetime.now(UTC).isoformat(),
}
result_path = RESULTS_DIR / f"{EXPERIMENT_ID}.json"
result_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
)
artifact_paths = [summary_path, cv_path, selection_path, result_path, *figure_artifacts]
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
                # Hierarchical Shrinkage Decision Tree — three datasets

                Notebook này triển khai đúng ba họ regularization trong đề xuất: **E1 pre-pruning**,
                **E2 cost-complexity pruning** và **E3 Hierarchical Shrinkage (HS)**; thêm **E4 CCP + HS**
                như thí nghiệm hỗ trợ. Cùng một notebook có thể chạy từng dataset bằng biến môi trường
                `HS_DATASET=letter_recognition`, `handwritten_digits`, `covertype` hoặc `all`.

                HS được áp dụng post-hoc: cây được fit trước, sau đó prediction ở node/leaf được kéo về
                thông tin ancestor bằng `lambda`; split, depth và số leaves không bị thay đổi. Notebook
                chọn hyperparameter bằng stratified CV trên train và chỉ dùng test cho báo cáo cuối.

                **Kaggle:** bật Internet để notebook tự cài `imodels==3.0.0` nếu kernel chưa có package.
                Chọn Accelerator: **None (CPU)**; Decision Tree/HS của scikit-learn không dùng GPU.
                """
            ),
            code(NOTEBOOK_CODE),
            markdown(
                """
                ## Cách chạy từng dataset

                Mặc định notebook chạy cả ba dataset. Để chạy riêng một bộ, đặt biến môi trường trước
                khi Run All, ví dụ `HS_DATASET=covertype`. Nếu Covertype quá lâu, có thể dùng
                `HS_MAX_TUNING_SAMPLES=120000` để chỉ dùng một tập train phân tầng cho bước chọn
                hyperparameter; test vẫn luôn là toàn bộ hold-out cố định.

                Kết quả E0–E4, bảng CV, lambda curves, confusion matrices, complexity plot và metadata
                phần cứng được đóng vào `dt_hierarchical_shrinkage__outputs.zip`.
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
