import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from decision_tree_lab2.custom_tree import CustomDecisionTreeClassifier


def test_custom_tree_learns_nested_thresholds_with_two_levels():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2])

    model = CustomDecisionTreeClassifier(max_depth=2).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)
    assert model.get_depth() == 2
    assert model.get_n_leaves() == 3
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_custom_tree_supports_string_labels_and_text_export():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array(["low", "low", "high", "high"])

    model = CustomDecisionTreeClassifier().fit(X, y)

    np.testing.assert_array_equal(model.predict([[0.05], [1.05]]), ["low", "high"])
    assert "feature_0" in model.export_text()
    assert "class='low'" in model.export_text()


def test_min_samples_leaf_is_respected():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = CustomDecisionTreeClassifier(min_samples_leaf=2).fit(X, y)

    assert model.get_n_leaves() == 2
    np.testing.assert_array_equal(model.predict(X), y)


def test_custom_tree_matches_sklearn_on_simple_unambiguous_data():
    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    custom = CustomDecisionTreeClassifier(max_depth=3).fit(X, y)
    sklearn_model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

    np.testing.assert_array_equal(custom.predict(X), sklearn_model.predict(X))
    assert custom.get_depth() == sklearn_model.get_depth()
    assert custom.get_n_leaves() == sklearn_model.get_n_leaves()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": -1},
        {"min_samples_split": 1},
        {"min_samples_leaf": 0},
        {"min_impurity_decrease": -0.1},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CustomDecisionTreeClassifier(**kwargs)
