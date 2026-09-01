"""Single source of truth for shared project settings and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
STRATIFY_SPLIT = True
RESULT_SCHEMA_VERSION = "1.0"


def ensure_project_dirs() -> None:
    """Create directories used by the project if they do not exist."""

    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
