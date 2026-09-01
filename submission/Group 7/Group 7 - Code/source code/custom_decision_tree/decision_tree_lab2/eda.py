"""Reusable descriptive analysis for the Letter Recognition dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .letter_data import LETTER_CLASSES, LETTER_FEATURES, LETTER_TARGET


def compute_class_distribution(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Compare class counts before and after exact-deduplication."""

    output = pd.DataFrame({LETTER_TARGET: LETTER_CLASSES})
    for name, frame in (("raw", raw), ("clean", cleaned)):
        counts = frame[LETTER_TARGET].value_counts().reindex(LETTER_CLASSES, fill_value=0)
        output[f"{name}_count"] = output[LETTER_TARGET].map(counts).astype("int64")
        output[f"{name}_percentage"] = output[f"{name}_count"] / len(frame) * 100
    output["duplicates_removed"] = output["raw_count"] - output["clean_count"]
    return output


def compute_feature_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return report-ready univariate statistics for all 16 features."""

    rows: list[dict[str, int | float | str]] = []
    for feature in LETTER_FEATURES:
        values = frame[feature]
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        iqr_outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        rows.append(
            {
                "feature": feature,
                "count": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "q1": q1,
                "median": float(values.median()),
                "q3": q3,
                "max": float(values.max()),
                "iqr": iqr,
                "skewness": float(values.skew()),
                "unique_values": int(values.nunique()),
                "iqr_outlier_count": iqr_outliers,
            }
        )
    return pd.DataFrame(rows)


def compute_correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlations between the 16 numeric features."""

    return frame.loc[:, list(LETTER_FEATURES)].corr(method="pearson")


def strongest_correlation_pairs(
    correlation: pd.DataFrame,
    *,
    limit: int = 8,
) -> list[dict[str, float | str]]:
    """Return strongest unique off-diagonal feature correlations."""

    pairs: list[dict[str, float | str]] = []
    for left_index, left in enumerate(LETTER_FEATURES):
        for right in LETTER_FEATURES[left_index + 1 :]:
            value = float(correlation.loc[left, right])
            pairs.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": value,
                    "absolute_correlation": abs(value),
                }
            )
    return sorted(pairs, key=lambda pair: float(pair["absolute_correlation"]), reverse=True)[:limit]


def compute_class_feature_profiles(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute class means and standardized profiles from training data only."""

    features = train.loc[:, list(LETTER_FEATURES)]
    class_means = train.groupby(LETTER_TARGET, observed=True)[list(LETTER_FEATURES)].mean()
    class_means = class_means.reindex(LETTER_CLASSES)
    scale = features.std().replace(0, 1)
    standardized = (class_means - features.mean()) / scale
    return class_means, standardized


def build_eda_insights(
    class_distribution: pd.DataFrame,
    feature_statistics: pd.DataFrame,
    correlation: pd.DataFrame,
    standardized_profiles: pd.DataFrame,
) -> dict[str, Any]:
    """Build deterministic observations used by the report and slide."""

    strongest_pairs = strongest_correlation_pairs(correlation)
    variability = feature_statistics.sort_values("std", ascending=False)
    skewness = feature_statistics.assign(abs_skewness=feature_statistics["skewness"].abs())
    skewness = skewness.sort_values("abs_skewness", ascending=False)
    separation = standardized_profiles.var(axis=0).sort_values(ascending=False)

    raw_min = int(class_distribution["raw_count"].min())
    raw_max = int(class_distribution["raw_count"].max())
    clean_min = int(class_distribution["clean_count"].min())
    clean_max = int(class_distribution["clean_count"].max())

    return {
        "raw_class_count_min": raw_min,
        "raw_class_count_max": raw_max,
        "raw_class_imbalance_ratio": float(raw_max / raw_min),
        "clean_class_count_min": clean_min,
        "clean_class_count_max": clean_max,
        "clean_class_imbalance_ratio": float(clean_max / clean_min),
        "most_duplicates_removed": class_distribution.nlargest(5, "duplicates_removed")[
            [LETTER_TARGET, "duplicates_removed"]
        ].to_dict(orient="records"),
        "highest_variability_features": variability.head(5)[["feature", "std"]].to_dict(
            orient="records"
        ),
        "most_skewed_features": skewness.head(5)[
            ["feature", "skewness", "iqr_outlier_count"]
        ].to_dict(orient="records"),
        "strongest_correlation_pairs": strongest_pairs,
        "most_class_separating_features_train_only": [
            {"feature": feature, "between_class_profile_variance": float(value)}
            for feature, value in separation.head(5).items()
        ],
        "mean_absolute_feature_correlation": float(
            np.abs(correlation.to_numpy()[np.triu_indices_from(correlation, k=1)]).mean()
        ),
    }
