import pandas as pd
import pytest

from decision_tree_lab2.letter_data import (
    LETTER_CLASSES,
    LETTER_FEATURES,
    LETTER_TARGET,
    audit_letter_recognition,
    clean_letter_recognition,
    split_letter_recognition,
)


def make_letter_frame(rows_per_class: int = 5) -> pd.DataFrame:
    rows = []
    row_number = 0
    for label in LETTER_CLASSES:
        for _ in range(rows_per_class):
            values = {
                feature: (row_number // (16**index)) % 16
                for index, feature in enumerate(LETTER_FEATURES)
            }
            rows.append({**values, LETTER_TARGET: label})
            row_number += 1
    return pd.DataFrame(rows)


def test_audit_and_clean_remove_exact_duplicates():
    original = make_letter_frame()
    duplicated = pd.concat([original, original.iloc[[0]]], ignore_index=True)

    audit = audit_letter_recognition(duplicated)
    cleaned, summary = clean_letter_recognition(duplicated)

    assert audit["exact_duplicate_rows"] == 1
    assert audit["conflicting_feature_groups"] == 0
    assert summary["exact_duplicates_removed"] == 1
    assert len(cleaned) == len(original)


def test_clean_rejects_conflicting_labels_for_same_features():
    frame = make_letter_frame()
    conflicting = frame.iloc[[0]].copy()
    conflicting[LETTER_TARGET] = "B"
    frame = pd.concat([frame, conflicting], ignore_index=True)

    with pytest.raises(ValueError, match="conflicting labels"):
        clean_letter_recognition(frame)


def test_clean_rejects_features_outside_documented_range():
    frame = make_letter_frame()
    frame.loc[0, LETTER_FEATURES[0]] = 16

    with pytest.raises(ValueError, match=r"must be in \[0, 15\]"):
        clean_letter_recognition(frame)


def test_shared_letter_split_is_reproducible_and_leak_free():
    cleaned, _ = clean_letter_recognition(make_letter_frame())

    first_train, first_test, first_report = split_letter_recognition(cleaned)
    second_train, second_test, second_report = split_letter_recognition(cleaned)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)
    assert first_report == second_report
    assert len(first_train) == 104
    assert len(first_test) == 26
    assert first_report["train_class_count"] == 26
    assert first_report["test_class_count"] == 26
    assert first_report["feature_vectors_shared_across_splits"] == 0
