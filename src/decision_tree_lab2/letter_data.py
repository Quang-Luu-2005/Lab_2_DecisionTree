"""Audit, clean, and split the UCI Letter Recognition dataset."""

from __future__ import annotations

from string import ascii_uppercase
from typing import Any

import pandas as pd

from .config import RANDOM_STATE, STRATIFY_SPLIT, TEST_SIZE
from .split import split_train_test

LETTER_TARGET = "letter"
LETTER_FEATURES = (
    "x_box",
    "y_box",
    "width",
    "high",
    "onpix",
    "x_bar",
    "y_bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x_ege",
    "xegvy",
    "y_ege",
    "yegvx",
)
LETTER_COLUMNS = (*LETTER_FEATURES, LETTER_TARGET)
LETTER_CLASSES = tuple(ascii_uppercase)
FEATURE_MIN = 0
FEATURE_MAX = 15


def _validate_columns(frame: pd.DataFrame) -> None:
    missing = sorted(set(LETTER_COLUMNS) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(LETTER_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            f"Invalid Letter Recognition columns: missing={missing}, unexpected={unexpected}"
        )


def _validate_numeric_features(frame: pd.DataFrame) -> None:
    non_numeric = [
        column for column in LETTER_FEATURES if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Letter Recognition features must be numeric: {non_numeric}")


def audit_letter_recognition(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-safe quality and descriptive-statistics audit."""

    _validate_columns(frame)
    ordered = frame.loc[:, list(LETTER_COLUMNS)]
    _validate_numeric_features(ordered)

    features = ordered.loc[:, list(LETTER_FEATURES)]
    target = ordered[LETTER_TARGET]
    missing_by_column = ordered.isna().sum()
    duplicate_feature_mask = ordered.duplicated(subset=list(LETTER_FEATURES), keep=False)
    duplicate_groups = ordered.loc[duplicate_feature_mask].groupby(
        list(LETTER_FEATURES), dropna=False
    )
    conflicting_groups = int((duplicate_groups[LETTER_TARGET].nunique() > 1).sum())

    below_range = features.lt(FEATURE_MIN)
    above_range = features.gt(FEATURE_MAX)
    out_of_range = below_range | above_range
    non_integer = features.notna() & features.ne(features.round())

    feature_summary: dict[str, dict[str, int | float]] = {}
    total_iqr_outlier_cells = 0
    for column in LETTER_FEATURES:
        series = features[column]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        iqr_outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        total_iqr_outlier_cells += iqr_outliers
        feature_summary[column] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q1": q1,
            "median": float(series.median()),
            "q3": q3,
            "max": float(series.max()),
            "unique": int(series.nunique(dropna=True)),
            "iqr_outlier_count": iqr_outliers,
        }

    class_counts = target.value_counts(dropna=False).sort_index()
    class_distribution = {
        str(label): {
            "count": int(count),
            "percentage": float(count / len(ordered) * 100),
        }
        for label, count in class_counts.items()
    }

    return {
        "rows": len(ordered),
        "columns": int(ordered.shape[1]),
        "feature_count": len(LETTER_FEATURES),
        "target": LETTER_TARGET,
        "class_count": int(target.nunique(dropna=True)),
        "class_labels": sorted(str(value) for value in target.dropna().unique()),
        "class_distribution": class_distribution,
        "missing_cells": int(missing_by_column.sum()),
        "missing_by_column": {column: int(count) for column, count in missing_by_column.items()},
        "exact_duplicate_rows": int(ordered.duplicated().sum()),
        "rows_in_duplicate_feature_groups": int(duplicate_feature_mask.sum()),
        "conflicting_feature_groups": conflicting_groups,
        "out_of_range_cells": int(out_of_range.sum().sum()),
        "out_of_range_rows": int(out_of_range.any(axis=1).sum()),
        "non_integer_feature_cells": int(non_integer.sum().sum()),
        "iqr_outlier_cells": total_iqr_outlier_cells,
        "feature_summary": feature_summary,
    }


def clean_letter_recognition(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate values and remove exact duplicates without mutating raw data."""

    _validate_columns(frame)
    cleaned = frame.loc[:, list(LETTER_COLUMNS)].copy()

    if cleaned.isna().any().any():
        raise ValueError("Missing values found; no imputation policy is defined for this dataset")

    for column in LETTER_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    _validate_numeric_features(cleaned)

    features = cleaned.loc[:, list(LETTER_FEATURES)]
    if features.ne(features.round()).any().any():
        raise ValueError("Letter Recognition features must contain integer values")
    if (features.lt(FEATURE_MIN) | features.gt(FEATURE_MAX)).any().any():
        raise ValueError(f"Letter Recognition features must be in [{FEATURE_MIN}, {FEATURE_MAX}]")

    cleaned.loc[:, list(LETTER_FEATURES)] = features.astype("int64")
    cleaned[LETTER_TARGET] = cleaned[LETTER_TARGET].astype("string").str.strip().str.upper()
    invalid_labels = sorted(set(cleaned[LETTER_TARGET]) - set(LETTER_CLASSES))
    if invalid_labels:
        raise ValueError(f"Unexpected Letter Recognition labels: {invalid_labels}")

    duplicate_feature_mask = cleaned.duplicated(subset=list(LETTER_FEATURES), keep=False)
    conflicting_groups = (
        cleaned.loc[duplicate_feature_mask].groupby(list(LETTER_FEATURES))[LETTER_TARGET].nunique()
    )
    conflicting_count = int((conflicting_groups > 1).sum())
    if conflicting_count:
        raise ValueError(f"Found {conflicting_count} feature groups with conflicting labels")

    rows_before = len(cleaned)
    cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
    rows_removed = rows_before - len(cleaned)

    summary = {
        "rows_before": int(rows_before),
        "rows_after": len(cleaned),
        "exact_duplicates_removed": int(rows_removed),
        "removal_percentage": float(rows_removed / rows_before * 100),
        "missing_value_action": "none_required",
        "outlier_action": "validate_expected_range_only",
        "scaling_action": "none_for_decision_tree",
        "feature_range": [FEATURE_MIN, FEATURE_MAX],
    }
    return cleaned, summary


def split_letter_recognition(
    cleaned: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = STRATIFY_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create the shared split and report leakage and class-balance checks."""

    _validate_columns(cleaned)
    if cleaned.duplicated().any():
        raise ValueError("Cleaned data still contains exact duplicates; clean before splitting")

    X = cleaned.loc[:, list(LETTER_FEATURES)]
    y = cleaned[LETTER_TARGET]
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train = X_train.assign(**{LETTER_TARGET: y_train}).loc[:, list(LETTER_COLUMNS)]
    test = X_test.assign(**{LETTER_TARGET: y_test}).loc[:, list(LETTER_COLUMNS)]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_keys = set(map(tuple, train.loc[:, list(LETTER_FEATURES)].to_numpy()))
    test_keys = set(map(tuple, test.loc[:, list(LETTER_FEATURES)].to_numpy()))
    feature_overlap = len(train_keys & test_keys)

    overall_share = cleaned[LETTER_TARGET].value_counts(normalize=True)
    train_share = train[LETTER_TARGET].value_counts(normalize=True).reindex(overall_share.index)
    test_share = test[LETTER_TARGET].value_counts(normalize=True).reindex(overall_share.index)

    report = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "stratify": bool(stratify),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_class_count": int(train[LETTER_TARGET].nunique()),
        "test_class_count": int(test[LETTER_TARGET].nunique()),
        "feature_vectors_shared_across_splits": int(feature_overlap),
        "max_train_class_share_delta": float((train_share - overall_share).abs().max()),
        "max_test_class_share_delta": float((test_share - overall_share).abs().max()),
    }
    return train, test, report
