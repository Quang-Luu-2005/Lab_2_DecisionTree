"""Build and persist experiment results using the shared JSON contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import RANDOM_STATE, RESULT_SCHEMA_VERSION, RESULTS_DIR, STRATIFY_SPLIT, TEST_SIZE


def _validate_metrics(metrics: Mapping[str, Any]) -> dict[str, int | float]:
    """Return JSON-safe metrics and reject non-finite numeric values."""

    validated: dict[str, int | float] = {}
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"Metric {name!r} must be an int or float")
        if not math.isfinite(value):
            raise ValueError(f"Metric {name!r} must be finite")
        validated[name] = value
    return validated


def build_result(
    *,
    experiment_id: str,
    dataset: str,
    model: str,
    metrics: Mapping[str, Any],
    figure_paths: list[str] | None = None,
    model_path: str | None = None,
    notes: str | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = STRATIFY_SPLIT,
) -> dict[str, Any]:
    """Create a result record following schema version ``1.0``."""

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model": model,
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "stratify": stratify,
        },
        "metrics": _validate_metrics(metrics),
        "artifacts": {
            "figure_paths": figure_paths or [],
            "model_path": model_path,
        },
        "notes": notes,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def save_result(result: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Write a result record to JSON and return the written path."""

    output_path = (
        Path(path) if path is not None else RESULTS_DIR / f"{result['experiment_id']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    return output_path
