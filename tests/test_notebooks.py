import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    PROJECT_ROOT / "notebooks" / "01_letter_decision_tree_baseline.ipynb",
    PROJECT_ROOT / "notebooks" / "02_digits_decision_tree_baseline.ipynb",
    PROJECT_ROOT / "notebooks" / "03_covertype_decision_tree_scalability.ipynb",
)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_baseline_notebook_is_valid_and_compilable(notebook_path: Path) -> None:
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
    assert "DecisionTreeClassifier" in source
    assert 'Path("/kaggle/working")' in source
    assert "ZipFile" in source
    assert '__outputs.zip"' in source
    assert "data_loading_seconds" in source
    assert "training_seconds" in source
    assert "prediction_seconds" in source
    assert "pipeline_seconds" in source
    assert "model_compute_device" in source
    assert "nvidia-smi" in source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)

    if "covertype" in notebook_path.name:
        assert "CV_FOLDS = 5" in source
        assert "cross_validate" in source
        assert "generalization_gap" in source
        assert "leaf_count" in source
