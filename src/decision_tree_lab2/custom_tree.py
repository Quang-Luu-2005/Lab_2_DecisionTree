"""A small educational CART classifier implemented without scikit-learn.

The implementation intentionally focuses on the classification pieces needed by
Lab 2: binary threshold splits, Gini impurity, stopping constraints and class
probabilities.  It is not intended to replace a production tree library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class _Node:
    """One node in the fitted binary tree."""

    depth: int
    n_samples: int
    impurity: float
    class_counts: np.ndarray
    prediction_index: int
    feature_index: int | None = None
    threshold: float | None = None
    left: int | None = None
    right: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class CustomDecisionTreeClassifier:
    """A deterministic, educational CART-style decision tree classifier.

    Splits are selected by exhaustive search over every feature and every
    distinct adjacent threshold.  The implementation uses Gini impurity and
    supports the main structural controls used by ``sklearn``'s classifier.
    It deliberately does not import or delegate to scikit-learn.
    """

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be None or a non-negative integer")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        if min_impurity_decrease < 0 or not np.isfinite(min_impurity_decrease):
            raise ValueError("min_impurity_decrease must be finite and non-negative")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "CustomDecisionTreeClassifier":
        """Fit the tree on a numeric feature matrix and a 1-D target."""

        features = np.asarray(X, dtype=float)
        if features.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric array")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite numeric values")

        target = np.asarray(y)
        if target.ndim != 1:
            target = target.reshape(-1)
        if len(target) != len(features):
            raise ValueError("X and y must contain the same number of samples")
        if len(target) == 0:
            raise ValueError("y must contain at least one sample")

        classes, encoded_target = np.unique(target, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self.n_features_in_ = features.shape[1]
        self._nodes: list[_Node] = []
        self._feature_importances_raw = np.zeros(self.n_features_in_, dtype=float)

        root_indices = np.arange(features.shape[0], dtype=np.int64)
        self.root_ = self._build_node(features, encoded_target, root_indices, depth=0)

        total_importance = self._feature_importances_raw.sum()
        if total_importance > 0:
            self.feature_importances_ = self._feature_importances_raw / total_importance
        else:
            self.feature_importances_ = np.zeros_like(self._feature_importances_raw)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict the most frequent class in the reached leaf."""

        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities for each sample."""

        self._check_is_fitted()
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite numeric values")

        probabilities = np.zeros((len(features), self.n_classes_), dtype=float)
        for row_index, row in enumerate(features):
            node = self._nodes[self.root_]
            while not node.is_leaf:
                if row[node.feature_index] <= node.threshold:  # type: ignore[index]
                    node = self._nodes[node.left]  # type: ignore[index]
                else:
                    node = self._nodes[node.right]  # type: ignore[index]
            probabilities[row_index] = node.class_counts / node.class_counts.sum()
        return probabilities

    def score(self, X: Any, y: Any) -> float:
        """Return mean classification accuracy."""

        target = np.asarray(y).reshape(-1)
        prediction = self.predict(X)
        if len(target) != len(prediction):
            raise ValueError("X and y must contain the same number of samples")
        return float(np.mean(prediction == target))

    def get_depth(self) -> int:
        """Return the maximum number of edges from root to a leaf."""

        self._check_is_fitted()
        return max(node.depth for node in self._nodes)

    def get_n_leaves(self) -> int:
        """Return the number of terminal leaves."""

        self._check_is_fitted()
        return sum(node.is_leaf for node in self._nodes)

    def export_text(self, feature_names: list[str] | None = None) -> str:
        """Render the fitted tree as readable nested split text."""

        self._check_is_fitted()
        if feature_names is None:
            feature_names = [f"feature_{index}" for index in range(self.n_features_in_)]
        if len(feature_names) != self.n_features_in_:
            raise ValueError("feature_names must match the number of input features")

        lines: list[str] = []

        def visit(node_index: int, prefix: str) -> None:
            node = self._nodes[node_index]
            class_name = self.classes_[node.prediction_index]
            if isinstance(class_name, np.generic):
                class_name = class_name.item()
            if node.is_leaf:
                lines.append(
                    f"{prefix}leaf: class={class_name!r}, samples={node.n_samples}, "
                    f"impurity={node.impurity:.6f}"
                )
                return
            feature_name = feature_names[node.feature_index]  # type: ignore[index]
            lines.append(
                f"{prefix}if {feature_name} <= {node.threshold:.6g}:"  # type: ignore[union-attr]
            )
            visit(node.left, prefix + "  ")  # type: ignore[arg-type]
            lines.append(f"{prefix}else:")
            visit(node.right, prefix + "  ")  # type: ignore[arg-type]

        visit(self.root_, "")
        return "\n".join(lines) + "\n"

    def _build_node(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_indices: np.ndarray,
        *,
        depth: int,
    ) -> int:
        counts = np.bincount(y[sample_indices], minlength=self.n_classes_).astype(float)
        n_samples = len(sample_indices)
        impurity = self._gini_from_counts(counts)
        prediction_index = int(np.argmax(counts))
        node_index = len(self._nodes)
        self._nodes.append(
            _Node(
                depth=depth,
                n_samples=n_samples,
                impurity=impurity,
                class_counts=counts,
                prediction_index=prediction_index,
            )
        )

        should_stop = (
            impurity == 0.0
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < 2 * self.min_samples_leaf
        )
        if should_stop:
            return node_index

        split = self._find_best_split(X, y, sample_indices, impurity)
        if split is None:
            return node_index
        feature_index, threshold, gain, left_indices, right_indices = split
        if gain <= self.min_impurity_decrease:
            return node_index

        self._feature_importances_raw[feature_index] += n_samples * gain
        self._nodes[node_index].feature_index = feature_index
        self._nodes[node_index].threshold = threshold
        self._nodes[node_index].left = self._build_node(
            X, y, left_indices, depth=depth + 1
        )
        self._nodes[node_index].right = self._build_node(
            X, y, right_indices, depth=depth + 1
        )
        return node_index

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_indices: np.ndarray,
        parent_impurity: float,
    ) -> tuple[int, float, float, np.ndarray, np.ndarray] | None:
        n_samples = len(sample_indices)
        best: tuple[int, float, float, np.ndarray, np.ndarray] | None = None
        best_gain = 0.0
        total_counts = np.bincount(y[sample_indices], minlength=self.n_classes_).astype(float)

        for feature_index in range(self.n_features_in_):
            values = X[sample_indices, feature_index]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_labels = y[sample_indices][order]
            if sorted_values[0] == sorted_values[-1]:
                continue

            positions = np.arange(1, n_samples, dtype=np.int64)
            different_value = sorted_values[:-1] < sorted_values[1:]
            valid = (
                different_value
                & (positions >= self.min_samples_leaf)
                & (n_samples - positions >= self.min_samples_leaf)
            )
            if not valid.any():
                continue

            one_hot = np.eye(self.n_classes_, dtype=float)[sorted_labels]
            prefix_counts = np.cumsum(one_hot, axis=0)
            valid_positions = positions[valid]
            left_counts = prefix_counts[valid_positions - 1]
            right_counts = total_counts - left_counts
            left_sizes = valid_positions.astype(float)
            right_sizes = (n_samples - valid_positions).astype(float)
            left_probabilities = left_counts / left_sizes[:, None]
            right_probabilities = right_counts / right_sizes[:, None]
            left_gini = 1.0 - np.sum(left_probabilities**2, axis=1)
            right_gini = 1.0 - np.sum(right_probabilities**2, axis=1)
            weighted_child_impurity = (
                left_sizes * left_gini + right_sizes * right_gini
            ) / n_samples
            gains = parent_impurity - weighted_child_impurity
            local_index = int(np.argmax(gains))
            gain = float(gains[local_index])
            if gain <= best_gain:
                continue

            position = int(valid_positions[local_index])
            left_value = float(sorted_values[position - 1])
            right_value = float(sorted_values[position])
            threshold = left_value + (right_value - left_value) / 2.0
            if not left_value < threshold < right_value:
                threshold = left_value
            feature_values = X[sample_indices, feature_index]
            left_mask = feature_values <= threshold
            left_indices = sample_indices[left_mask]
            right_indices = sample_indices[~left_mask]
            if (
                len(left_indices) < self.min_samples_leaf
                or len(right_indices) < self.min_samples_leaf
            ):
                continue
            best = (feature_index, threshold, gain, left_indices, right_indices)
            best_gain = gain

        return best

    @staticmethod
    def _gini_from_counts(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        probabilities = counts / total
        return float(1.0 - np.sum(probabilities**2))

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_nodes"):
            raise RuntimeError("The classifier must be fitted before prediction")
