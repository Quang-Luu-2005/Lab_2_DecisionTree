import numpy as np
import pandas as pd

from decision_tree_lab2.eda import (
    build_eda_insights,
    compute_class_distribution,
    compute_class_feature_profiles,
    compute_correlation_matrix,
    compute_feature_statistics,
    strongest_correlation_pairs,
)
from decision_tree_lab2.letter_data import LETTER_CLASSES, LETTER_FEATURES, LETTER_TARGET


def make_eda_frame(rows_per_class: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for label_index, label in enumerate(LETTER_CLASSES):
        for _ in range(rows_per_class):
            values = rng.integers(0, 16, size=len(LETTER_FEATURES))
            values[label_index % len(LETTER_FEATURES)] = (label_index * 3) % 16
            rows.append(
                {
                    **dict(zip(LETTER_FEATURES, values, strict=True)),
                    LETTER_TARGET: label,
                }
            )
    return pd.DataFrame(rows)


def test_class_distribution_compares_raw_and_clean_counts():
    cleaned = make_eda_frame()
    raw = pd.concat([cleaned, cleaned.iloc[[0, 1]]], ignore_index=True)

    distribution = compute_class_distribution(raw, cleaned)

    assert distribution["raw_count"].sum() == len(raw)
    assert distribution["clean_count"].sum() == len(cleaned)
    assert distribution["duplicates_removed"].sum() == 2
    assert len(distribution) == 26


def test_feature_statistics_cover_all_features():
    statistics = compute_feature_statistics(make_eda_frame())

    assert statistics["feature"].tolist() == list(LETTER_FEATURES)
    assert statistics["min"].ge(0).all()
    assert statistics["max"].le(15).all()
    assert statistics["unique_values"].gt(1).all()


def test_correlation_pairs_are_unique_and_sorted():
    correlation = compute_correlation_matrix(make_eda_frame())
    pairs = strongest_correlation_pairs(correlation, limit=10)

    assert len(pairs) == 10
    assert all(pair["feature_1"] != pair["feature_2"] for pair in pairs)
    absolute_values = [pair["absolute_correlation"] for pair in pairs]
    assert absolute_values == sorted(absolute_values, reverse=True)


def test_profiles_and_insights_use_all_classes_and_features():
    frame = make_eda_frame()
    class_means, profiles = compute_class_feature_profiles(frame)
    distribution = compute_class_distribution(frame, frame)
    statistics = compute_feature_statistics(frame)
    correlation = compute_correlation_matrix(frame)
    insights = build_eda_insights(distribution, statistics, correlation, profiles)

    assert class_means.shape == (26, 16)
    assert profiles.shape == (26, 16)
    assert not profiles.isna().any().any()
    assert len(insights["strongest_correlation_pairs"]) == 8
    assert len(insights["most_class_separating_features_train_only"]) == 5
