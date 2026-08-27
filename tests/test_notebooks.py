import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted((PROJECT_ROOT / "notebooks").rglob("*.ipynb")))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_is_valid_and_compilable(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    is_gpu_benchmark = notebook_path.name == "05_three_dataset_three_model_benchmark.ipynb"
    kaggle_metadata = notebook["metadata"]["kaggle"]
    if is_gpu_benchmark:
        assert kaggle_metadata["accelerator"] in {"nvidiaTeslaT4", "nvidiaTeslaP100"}
        assert kaggle_metadata["isGpuEnabled"] is True
    else:
        assert kaggle_metadata["accelerator"] == "none"
        assert kaggle_metadata["isGpuEnabled"] is False

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
        if notebook_path.name == "04_hierarchical_shrinkage_three_dataset_benchmark.ipynb":
            assert "HSTreeClassifier" in source
            assert "HS_LAMBDA_VALUES" in source
            assert "E4 CCP+HS" in source
            assert "structure_unchanged_after_hs" in source
            assert "HS_DATASET" in source
    elif notebook_group == "benchmark_models":
        assert "training_seconds" in source
        assert "prediction_seconds" in source
        assert "f1_macro" in source
        assert "generalization_gap" in source
        if notebook_path.name == "05_three_dataset_three_model_benchmark.ipynb":
            assert "covertype" in source
            assert "DecisionTreeClassifier" in source
            assert "RandomForestClassifier" in source
            assert "KNeighborsClassifier" in source
            assert "SVC" in source
            assert "cupy" in source
            assert "cuml" in source
            assert '"compute_device": "GPU"' in source
            assert "predict_in_batches" in source
            assert "total_model_seconds" in source
        else:
            assert "nvidia-smi" in source
            assert "letter_recognition" in source
            assert "handwritten_digits" in source


def test_expected_notebook_suite_exists() -> None:
    names = {path.name for path in NOTEBOOKS}
    assert names == {
        "01_letter_decision_tree_baseline.ipynb",
        "02_digits_decision_tree_baseline.ipynb",
        "03_covertype_decision_tree_scalability.ipynb",
        "04_hierarchical_shrinkage_three_dataset_benchmark.ipynb",
        "05_three_dataset_three_model_benchmark.ipynb",
        "06_gini_vs_entropy_experiment.ipynb",
        "07_depth_sweep_experiment.ipynb",
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
