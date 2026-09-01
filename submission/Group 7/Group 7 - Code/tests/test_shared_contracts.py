from decision_tree_lab2.config import RANDOM_STATE, STRATIFY_SPLIT, TEST_SIZE
from decision_tree_lab2.results import build_result
from decision_tree_lab2.split import split_train_test


def test_shared_split_defaults():
    assert RANDOM_STATE == 42
    assert TEST_SIZE == 0.20
    assert STRATIFY_SPLIT is True


def test_result_contract():
    result = build_result(
        experiment_id="dt_baseline",
        dataset="toy",
        model="DecisionTreeClassifier",
        metrics={"accuracy": 0.9},
    )

    assert result["schema_version"] == "1.0"
    assert result["split"] == {"test_size": 0.20, "random_state": 42, "stratify": True}
    assert result["artifacts"] == {"figure_paths": [], "model_path": None}


def test_shared_split_is_reproducible():
    X = list(range(10))
    y = [0, 1] * 5

    first = split_train_test(X, y)
    second = split_train_test(X, y)

    assert first == second
    assert len(first[0]) == 8
    assert len(first[1]) == 2
