"""Shared train/test split helper."""

from typing import Any

from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, STRATIFY_SPLIT, TEST_SIZE


def split_train_test(
    X: Any,
    y: Any,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = STRATIFY_SPLIT,
) -> tuple[Any, Any, Any, Any]:
    """Split data with the project-wide reproducibility defaults.

    The returned values are ``X_train, X_test, y_train, y_test``. For a
    classification task, keep ``stratify=True`` so class proportions are
    preserved in both subsets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
