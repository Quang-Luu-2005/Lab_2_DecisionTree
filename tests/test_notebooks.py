import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted((PROJECT_ROOT / "notebooks").rglob("*.ipynb")))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_is_valid_and_compilable(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kaggle"]["accelerator"] == "none"
    assert notebook["metadata"]["kaggle"]["isGpuEnabled"] is False

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    compile(source, str(notebook_path), "exec")

    assert "RANDOM_STATE = 42" in source
    assert "TEST_SIZE = 0.20" in source
    assert 'Path("/kaggle/working")' in source
    assert "ZipFile" in source
    assert '__outputs.zip"' in source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)

    notebook_group = notebook_path.parent.name
    if notebook_group == "decision_tree":
        assert "DecisionTreeClassifier" in source
        assert "data_loading_seconds" in source
        assert "training_seconds" in source
        assert "prediction_seconds" in source
        assert "pipeline_seconds" in source
        assert "model_compute_device" in source
        assert "nvidia-smi" in source
    elif notebook_group == "benchmark_models":
        assert "training_seconds" in source
        assert "prediction_seconds" in source
        assert "f1_macro" in source
        assert "generalization_gap" in source
        assert "nvidia-smi" in source
        if notebook_path.name == "07_three_dataset_three_model_benchmark.ipynb":
            assert "covertype" in source
            assert "RandomForestClassifier" in source
            assert "KNeighborsClassifier" in source
            assert "SVC" in source
            assert '"scope": "full_dataset_split"' in source
            assert "predict_in_batches" in source
            assert "write_checkpoint" in source
            assert "total_model_seconds" in source
        else:
            assert "letter_recognition" in source
            assert "handwritten_digits" in source
    elif notebook_group == "model_comparison":
        assert "dt_covertype_scalability.json" in source
        assert "three_dataset_three_model_benchmark.json" in source
        assert "generalization_gap" in source
        assert "best_by_dataset" in source
        assert "total_model_seconds" in source


def test_expected_notebook_suite_exists() -> None:
    names = {path.name for path in NOTEBOOKS}
    assert names == {
        "01_letter_decision_tree_baseline.ipynb",
        "02_digits_decision_tree_baseline.ipynb",
        "03_covertype_decision_tree_scalability.ipynb",
        "07_three_dataset_three_model_benchmark.ipynb",
        "08_three_dataset_model_comparison.ipynb",
    }


def test_covertype_notebook_has_scalability_protocol() -> None:
    path = (
        PROJECT_ROOT
        / "notebooks"
        / "decision_tree"
        / "03_covertype_decision_tree_scalability.ipynb"
    )
    source = path.read_text(encoding="utf-8")
    assert "CV_FOLDS = 5" in source
    assert "cross_validate" in source
    assert "generalization_gap" in source
    assert "leaf_count" in source
