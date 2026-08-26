"""Generate the Kaggle-ready Decision Tree improvement notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "decision_tree" / ("04_letter_decision_tree_improvements.ipynb")
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


def make_notebook() -> dict[str, object]:
    return {
        "cells": [
            markdown(
                """
                # Decision Tree improvements — Letter Recognition

                Notebook phục vụ ba nhánh cải tiến: `max_depth`,
                `min_samples_split`/`min_samples_leaf` và cost-complexity pruning (`ccp_alpha`).
                Candidate được đo trên hold-out để báo cáo; cấu hình được chọn bằng stratified
                5-fold CV chỉ trên tập train, tránh dùng test để tuning.

                Chọn **Accelerator: None (CPU)**. Decision Tree của scikit-learn không dùng GPU.
                """
            ),
            code(
                r"""
                import json
                import os
                import platform
                import subprocess
                import time
                from datetime import UTC, datetime
                from pathlib import Path
                from shutil import which
                from zipfile import ZIP_DEFLATED, ZipFile

                import joblib
                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import seaborn as sns
                import sklearn
                from sklearn.metrics import (
                    ConfusionMatrixDisplay,
                    accuracy_score,
                    f1_score,
                    precision_score,
                    recall_score,
                )
                from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
                from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

                try:
                    from IPython.display import FileLink, display
                except ImportError:
                    FileLink = None

                    def display(value):
                        print(value)

                RANDOM_STATE = 42
                PIPELINE_STARTED = time.perf_counter()
                TEST_SIZE = 0.20
                CV_FOLDS = 5
                EXPERIMENT_ID = "dt_letter_improvements"
                TARGET = "letter"
                FEATURES = [
                    "x_box", "y_box", "width", "high", "onpix", "x_bar", "y_bar", "x2bar",
                    "y2bar", "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx",
                ]
                IS_KAGGLE = Path("/kaggle/working").exists()
                OUTPUT_ROOT = Path("/kaggle/working") if IS_KAGGLE else Path.cwd()
                FIGURES_DIR = OUTPUT_ROOT / "figures"
                RESULTS_DIR = OUTPUT_ROOT / "results"
                MODELS_DIR = OUTPUT_ROOT / "models"
                for directory in (FIGURES_DIR, RESULTS_DIR, MODELS_DIR):
                    directory.mkdir(parents=True, exist_ok=True)
                sns.set_theme(style="whitegrid")

                def detect_hardware():
                    info = {
                        "environment": "kaggle" if IS_KAGGLE else "local",
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
                        )
                        if query.returncode == 0 and query.stdout.strip():
                            info["gpu_available"] = True
                            for line in query.stdout.strip().splitlines():
                                name, memory_mb, driver = [
                                    part.strip() for part in line.split(",", maxsplit=2)
                                ]
                                info["gpus"].append(
                                    {
                                        "name": name,
                                        "memory_mb": int(float(memory_mb)),
                                        "driver_version": driver,
                                    }
                                )
                    return info

                HARDWARE_INFO = detect_hardware()
                print({"python": platform.python_version(), "sklearn": sklearn.__version__})
                print("Hardware:", HARDWARE_INFO)
                print("Decision Tree compute device: CPU")

                data_started = time.perf_counter()
                REQUIRED_COLUMNS = set(FEATURES + [TARGET])
                SEARCH_ROOTS = [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent]

                def csv_has_columns(path):
                    try:
                        return REQUIRED_COLUMNS.issubset(pd.read_csv(path, nrows=1).columns)
                    except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
                        return False

                split_paths = None
                for root in SEARCH_ROOTS:
                    if not root.exists():
                        continue
                    for candidate in sorted(root.rglob("train.csv")):
                        test_candidate = candidate.with_name("test.csv")
                        if (
                            test_candidate.exists()
                            and csv_has_columns(candidate)
                            and csv_has_columns(test_candidate)
                        ):
                            split_paths = (candidate, test_candidate)
                            break
                    if split_paths:
                        break

                if split_paths:
                    train_path, test_path = split_paths
                    train_df = pd.read_csv(train_path)
                    test_df = pd.read_csv(test_path)
                    data_source = f"canonical split: {train_path.parent}"
                else:
                    raw_path = None
                    for root in SEARCH_ROOTS:
                        if not root.exists():
                            continue
                        for name in ("letter_recognition.csv", "letter-recognition.data"):
                            matches = sorted(root.rglob(name))
                            if matches:
                                raw_path = matches[0]
                                break
                        if raw_path:
                            break
                    if raw_path is None:
                        raise FileNotFoundError(
                            "Missing Letter Recognition. Add canonical train.csv/test.csv."
                        )
                    if raw_path.name == "letter-recognition.data":
                        raw_df = pd.read_csv(raw_path, header=None, names=[TARGET] + FEATURES)
                    else:
                        raw_df = pd.read_csv(raw_path)
                    raw_df = raw_df[FEATURES + [TARGET]].drop_duplicates().reset_index(drop=True)
                    train_df, test_df = train_test_split(
                        raw_df,
                        test_size=TEST_SIZE,
                        random_state=RANDOM_STATE,
                        stratify=raw_df[TARGET],
                    )
                    train_df = train_df.reset_index(drop=True)
                    test_df = test_df.reset_index(drop=True)
                    data_source = f"raw CSV + notebook split: {raw_path}"

                assert set(train_df.columns) == REQUIRED_COLUMNS
                assert set(test_df.columns) == REQUIRED_COLUMNS
                assert not train_df.isna().any().any() and not test_df.isna().any().any()
                X_train, y_train = train_df[FEATURES], train_df[TARGET]
                X_test, y_test = test_df[FEATURES], test_df[TARGET]
                data_loading_seconds = time.perf_counter() - data_started
                print("Source:", data_source)
                print("Train/test:", X_train.shape, X_test.shape)

                depth_values = [2, 3, 5, 8, 12, 16, 20, 24, None]
                min_split_values = [2, 10, 50, 100]
                min_leaf_values = [1, 2, 5, 10]
                candidates = []
                for depth in depth_values:
                    candidates.append(
                        {
                            "candidate_id": f"depth_{depth if depth is not None else 'none'}",
                            "branch": "max_depth",
                            "parameters": {"max_depth": depth},
                        }
                    )
                for min_split in min_split_values:
                    for min_leaf in min_leaf_values:
                        candidates.append(
                            {
                                "candidate_id": f"min_split_{min_split}__min_leaf_{min_leaf}",
                                "branch": "min_samples",
                                "parameters": {
                                    "min_samples_split": min_split,
                                    "min_samples_leaf": min_leaf,
                                },
                            }
                        )
                path_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
                path_model.fit(X_train, y_train)
                raw_alphas = path_model.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
                positive_alphas = np.asarray(raw_alphas)[np.asarray(raw_alphas) > 0]
                if len(positive_alphas) > 10:
                    alpha_values = np.unique(
                        np.concatenate(([0.0], np.quantile(positive_alphas, np.linspace(0, 1, 10))))
                    )
                else:
                    alpha_values = np.unique(raw_alphas)
                for alpha in alpha_values:
                    candidates.append(
                        {
                            "candidate_id": f"ccp_alpha_{alpha:.12g}",
                            "branch": "ccp_alpha",
                            "parameters": {"ccp_alpha": float(alpha)},
                        }
                    )

                def evaluate_holdout(item):
                    model = DecisionTreeClassifier(random_state=RANDOM_STATE, **item["parameters"])
                    fit_started = time.perf_counter()
                    model.fit(X_train, y_train)
                    training_seconds = time.perf_counter() - fit_started
                    prediction_started = time.perf_counter()
                    train_prediction = model.predict(X_train)
                    test_prediction = model.predict(X_test)
                    prediction_seconds = time.perf_counter() - prediction_started
                    train_accuracy = accuracy_score(y_train, train_prediction)
                    test_accuracy = accuracy_score(y_test, test_prediction)
                    return {
                        "candidate_id": item["candidate_id"],
                        "branch": item["branch"],
                        "parameters": item["parameters"],
                        "train_accuracy": float(train_accuracy),
                        "test_accuracy": float(test_accuracy),
                        "error_rate": float(1 - test_accuracy),
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
                        "tree_depth": int(model.get_depth()),
                        "leaf_count": int(model.get_n_leaves()),
                    }

                holdout_df = pd.DataFrame([evaluate_holdout(item) for item in candidates])
                print("Hold-out candidates:", len(holdout_df))
                display(holdout_df.sort_values("f1_macro", ascending=False).head(10).round(4))

                cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                cv_rows = []
                for item in candidates:
                    cv_output = cross_validate(
                        DecisionTreeClassifier(random_state=RANDOM_STATE, **item["parameters"]),
                        X_train,
                        y_train,
                        cv=cv,
                        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
                        return_train_score=True,
                        n_jobs=-1,
                    )
                    cv_rows.append(
                        {
                            "candidate_id": item["candidate_id"],
                            "branch": item["branch"],
                            "parameters": item["parameters"],
                            "cv_train_accuracy_mean": float(cv_output["train_accuracy"].mean()),
                            "cv_test_accuracy_mean": float(cv_output["test_accuracy"].mean()),
                            "cv_test_accuracy_std": float(cv_output["test_accuracy"].std()),
                            "cv_f1_macro_mean": float(cv_output["test_f1_macro"].mean()),
                            "cv_f1_macro_std": float(cv_output["test_f1_macro"].std()),
                            "cv_fit_seconds_mean": float(cv_output["fit_time"].mean()),
                        }
                    )
                cv_df = pd.DataFrame(cv_rows)
                selected_id = cv_df.loc[cv_df["cv_f1_macro_mean"].idxmax(), "candidate_id"]
                selected_cv = cv_df.loc[cv_df["candidate_id"] == selected_id].iloc[0]
                selected_holdout = holdout_df.loc[
                    holdout_df["candidate_id"] == selected_id
                ].iloc[0]
                selected_candidate = next(item for item in candidates if item["candidate_id"] == selected_id)
                best_model = DecisionTreeClassifier(
                    random_state=RANDOM_STATE, **selected_candidate["parameters"]
                )
                best_model.fit(X_train, y_train)
                best_test_prediction = best_model.predict(X_test)
                print("Selected by training-only CV macro-F1:", selected_id)
                print("Selected parameters:", selected_candidate["parameters"])

                depth_df = holdout_df.loc[holdout_df["branch"] == "max_depth"].copy()
                depth_df["depth_label"] = depth_df["parameters"].map(
                    lambda parameters: str(parameters["max_depth"])
                )
                fig, ax = plt.subplots(figsize=(10, 5))
                for column, label in (
                    ("train_accuracy", "Train accuracy"),
                    ("test_accuracy", "Test accuracy"),
                    ("f1_macro", "Test macro-F1"),
                ):
                    ax.plot(depth_df["depth_label"], depth_df[column], marker="o", label=label)
                ax.set(title="Depth sweep: underfitting to overfitting", xlabel="max_depth", ylabel="Score")
                ax.set_ylim(0, 1.05)
                ax.legend()
                fig.tight_layout()
                depth_figure = FIGURES_DIR / f"{EXPERIMENT_ID}__depth_sweep.png"
                fig.savefig(depth_figure, dpi=200, bbox_inches="tight")
                plt.show()

                min_samples_df = holdout_df.loc[holdout_df["branch"] == "min_samples"].copy()
                min_samples_df["min_split"] = min_samples_df["parameters"].map(
                    lambda parameters: parameters["min_samples_split"]
                )
                min_samples_df["min_leaf"] = min_samples_df["parameters"].map(
                    lambda parameters: parameters["min_samples_leaf"]
                )
                heatmap = min_samples_df.pivot(index="min_leaf", columns="min_split", values="f1_macro")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(heatmap, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
                ax.set(title="min_samples grid: test macro-F1", xlabel="min_samples_split", ylabel="min_samples_leaf")
                fig.tight_layout()
                min_samples_figure = FIGURES_DIR / f"{EXPERIMENT_ID}__min_samples_heatmap.png"
                fig.savefig(min_samples_figure, dpi=200, bbox_inches="tight")
                plt.show()

                pruning_df = holdout_df.loc[holdout_df["branch"] == "ccp_alpha"].copy()
                pruning_df["ccp_alpha_value"] = pruning_df["parameters"].map(
                    lambda parameters: parameters["ccp_alpha"]
                )
                pruning_df = pruning_df.sort_values("ccp_alpha_value")
                fig, ax = plt.subplots(figsize=(10, 5))
                plot_x = pruning_df["ccp_alpha_value"].to_numpy()
                plot_x = np.where(
                    plot_x > 0,
                    plot_x,
                    max(pruning_df["ccp_alpha_value"].max() / 100, 1e-12),
                )
                ax.plot(plot_x, pruning_df["test_accuracy"], marker="o", label="Test accuracy")
                ax.plot(plot_x, pruning_df["f1_macro"], marker="o", label="Test macro-F1")
                ax.set_xscale("log")
                ax.set(title="Cost-complexity pruning path", xlabel="ccp_alpha (log scale)", ylabel="Score")
                ax.set_ylim(0, 1.05)
                ax.legend()
                fig.tight_layout()
                pruning_figure = FIGURES_DIR / f"{EXPERIMENT_ID}__pruning_path.png"
                fig.savefig(pruning_figure, dpi=200, bbox_inches="tight")
                plt.show()

                fig, ax = plt.subplots(figsize=(13, 10))
                ConfusionMatrixDisplay.from_predictions(
                    y_test, best_test_prediction, labels=best_model.classes_, cmap="Blues", colorbar=False, ax=ax
                )
                ax.set_title(f"Selected Decision Tree - {selected_id}")
                fig.tight_layout()
                confusion_figure = FIGURES_DIR / f"{EXPERIMENT_ID}__selected_confusion_matrix.png"
                fig.savefig(confusion_figure, dpi=200, bbox_inches="tight")
                plt.show()

                fig, ax = plt.subplots(figsize=(24, 11))
                plot_tree(
                    best_model, feature_names=FEATURES, class_names=best_model.classes_,
                    max_depth=3, filled=True, rounded=True, fontsize=7, ax=ax
                )
                ax.set_title(f"Selected Decision Tree - first four levels ({selected_id})")
                fig.tight_layout()
                tree_figure = FIGURES_DIR / f"{EXPERIMENT_ID}__selected_tree_top_levels.png"
                fig.savefig(tree_figure, dpi=200, bbox_inches="tight")
                plt.show()

                importance = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
                fig, ax = plt.subplots(figsize=(9, 6))
                importance.plot.barh(ax=ax, color="#0F766E")
                ax.set(title="Selected Decision Tree - feature importance", xlabel="Gini importance", ylabel="")
                fig.tight_layout()
                importance_figure = FIGURES_DIR / f"{EXPERIMENT_ID}__selected_feature_importance.png"
                fig.savefig(importance_figure, dpi=200, bbox_inches="tight")
                plt.show()

                holdout_path = RESULTS_DIR / f"{EXPERIMENT_ID}__holdout_candidates.csv"
                cv_path = RESULTS_DIR / f"{EXPERIMENT_ID}__cross_validation.csv"
                holdout_export = holdout_df.copy()
                holdout_export["parameters"] = holdout_export["parameters"].map(json.dumps)
                holdout_export.to_csv(holdout_path, index=False)
                cv_export = cv_df.copy()
                cv_export["parameters"] = cv_export["parameters"].map(json.dumps)
                cv_export.to_csv(cv_path, index=False)
                rules_path = RESULTS_DIR / f"{EXPERIMENT_ID}__selected_rules.txt"
                rules_path.write_text(export_text(best_model, feature_names=FEATURES, max_depth=5), encoding="utf-8")
                model_path = MODELS_DIR / f"{EXPERIMENT_ID}__selected.joblib"
                joblib.dump(best_model, model_path)
                pipeline_seconds = time.perf_counter() - PIPELINE_STARTED
                result = {
                    "schema_version": "1.0",
                    "experiment_id": EXPERIMENT_ID,
                    "dataset": "letter_recognition",
                    "model": "DecisionTreeClassifier",
                    "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratify": True},
                    "selection": {
                        "method": "5-fold stratified CV on training data only",
                        "metric": "macro-F1",
                        "selected_candidate_id": selected_id,
                        "selected_parameters": selected_candidate["parameters"],
                        "selected_cv": {
                            key: value for key, value in selected_cv.to_dict().items() if key != "parameters"
                        },
                    },
                    "metrics": {
                        key: float(selected_holdout[key])
                        for key in (
                            "train_accuracy", "test_accuracy", "error_rate", "precision_macro",
                            "recall_macro", "f1_macro", "generalization_gap", "training_seconds",
                            "prediction_seconds",
                        )
                    } | {
                        "tree_depth": int(selected_holdout["tree_depth"]),
                        "leaf_count": int(selected_holdout["leaf_count"]),
                        "data_loading_seconds": float(data_loading_seconds),
                        "pipeline_seconds": float(pipeline_seconds),
                    },
                    "candidate_count": len(candidates),
                    "hardware": HARDWARE_INFO,
                    "artifacts": {
                        "figure_paths": [
                            str(depth_figure), str(min_samples_figure), str(pruning_figure),
                            str(confusion_figure), str(tree_figure), str(importance_figure),
                        ],
                        "table_paths": [str(holdout_path), str(cv_path)],
                        "rules_path": str(rules_path),
                        "model_path": str(model_path),
                    },
                    "notes": (
                        "Branches: max_depth, min_samples_split/min_samples_leaf and ccp_alpha. "
                        "Test set is used for reporting only; CV selection uses train."
                    ),
                    "created_at_utc": datetime.now(UTC).isoformat(),
                }
                result_path = RESULTS_DIR / f"{EXPERIMENT_ID}.json"
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                files_to_download = [
                    depth_figure, min_samples_figure, pruning_figure, confusion_figure,
                    tree_figure, importance_figure, holdout_path, cv_path, rules_path, model_path, result_path,
                ]
                archive_path = OUTPUT_ROOT / f"{EXPERIMENT_ID}__outputs.zip"
                with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
                    for artifact_path in files_to_download:
                        archive.write(artifact_path, arcname=artifact_path.relative_to(OUTPUT_ROOT))
                print(f"Created ZIP ({archive_path.stat().st_size / 1024**2:.1f} MB): {archive_path}")
                if FileLink is not None:
                    display(FileLink(str(archive_path)))
                """
            ),
            markdown(
                """
                ## Cách đọc kết quả

                `cv_test_f1_macro_mean` là tiêu chí chọn cấu hình; các cột test hold-out dùng
                để báo cáo sau cùng. Hãy đối chiếu depth, số lá và generalization gap để giải
                thích đánh đổi giữa hiệu năng và độ phức tạp.
                """
            ),
        ],
        "metadata": {
            "kaggle": {"accelerator": "none", "isGpuEnabled": False, "internet": False},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(make_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
