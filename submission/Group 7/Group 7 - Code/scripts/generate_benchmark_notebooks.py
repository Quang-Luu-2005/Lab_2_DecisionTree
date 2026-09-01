"""Compatibility entry point for generating the combined Kaggle benchmark."""

from __future__ import annotations

from pathlib import Path
from runpy import run_path

GPU_GENERATOR_PATH = Path(__file__).resolve().with_name("generate_gpu_benchmark_notebook.py")


def generate_benchmark_notebook() -> None:
    run_path(str(GPU_GENERATOR_PATH), run_name="__main__")


if __name__ == "__main__":
    generate_benchmark_notebook()
