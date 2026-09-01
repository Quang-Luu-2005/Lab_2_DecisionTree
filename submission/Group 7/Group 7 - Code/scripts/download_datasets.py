"""Download and materialize the datasets selected in the project plan.

Outputs are intentionally written below ``data/raw/`` and are ignored by Git.
Run from the repository root with:

    python scripts/download_datasets.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from io import StringIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.datasets import fetch_covtype, load_digits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LETTER_DIR = RAW_DIR / "letter_recognition"
DIGITS_DIR = RAW_DIR / "handwritten_digits"
COVERTYPE_DIR = RAW_DIR / "covertype"
LETTER_ZIP = LETTER_DIR / "letter+recognition.zip"
DIGITS_CSV = DIGITS_DIR / "digits.csv"
COVERTYPE_CSV = COVERTYPE_DIR / "covertype.csv"
MANIFEST_PATH = RAW_DIR / "dataset_manifest.json"

LETTER_URL = "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip"
LETTER_SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/59/letter%2Brecognition"
LETTER_MIRROR_URL = "https://www.openml.org/data/v1/download/6/letter.arff"
LETTER_MIRROR_PAGE = "https://www.openml.org/d/6"
DIGITS_SOURCE_PAGE = (
    "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html"
)
COVERTYPE_SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/31/covertype"
LETTER_COLUMNS = [
    "x_box",
    "y_box",
    "width",
    "high",
    "onpix",
    "x_bar",
    "y_bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x_ege",
    "xegvy",
    "y_ege",
    "yegvx",
    "letter",
]
LETTER_UCI_COLUMNS = ["letter", *LETTER_COLUMNS[:-1]]


def download(url: str, destination: Path) -> None:
    """Download a URL to a local file using a descriptive user agent."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "decision-tree-lab2/0.1"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_letter_arff(path: Path) -> pd.DataFrame:
    """Parse the OpenML ARFF mirror without adding an ARFF dependency."""

    data_started = False
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.lower() == "@data":
            data_started = True
            continue
        if data_started and not stripped.startswith("@"):
            rows.append(stripped)
    frame = pd.read_csv(StringIO("\n".join(rows)), header=None, names=LETTER_COLUMNS)
    if frame.shape != (20000, 17):
        raise ValueError(f"Unexpected Letter Recognition shape: {frame.shape}")
    return frame


def materialize_letter_csv() -> Path:
    """Create a normalized CSV from either the UCI archive or OpenML ARFF."""

    csv_path = LETTER_DIR / "letter_recognition.csv"
    data_file = LETTER_DIR / "letter-recognition.data"
    arff_file = LETTER_DIR / "letter.arff"
    if data_file.exists():
        frame = pd.read_csv(data_file, header=None, names=LETTER_UCI_COLUMNS)
        frame = frame[LETTER_COLUMNS]
    elif arff_file.exists():
        frame = parse_letter_arff(arff_file)
    else:
        raise FileNotFoundError("Neither UCI .data nor OpenML .arff file was found")
    if frame.shape != (20000, 17):
        raise ValueError(f"Unexpected Letter Recognition shape: {frame.shape}")
    frame.to_csv(csv_path, index=False)
    return csv_path


def download_letter() -> dict[str, object]:
    """Download Letter Recognition, falling back to its OpenML mirror."""

    LETTER_DIR.mkdir(parents=True, exist_ok=True)
    source_used = "UCI official archive"
    download_url = LETTER_URL
    try:
        if not LETTER_ZIP.exists() or LETTER_ZIP.stat().st_size == 0:
            print(f"Downloading Letter Recognition from {LETTER_URL}", flush=True)
            download(LETTER_URL, LETTER_ZIP)
        with zipfile.ZipFile(LETTER_ZIP) as archive:
            archive.extractall(LETTER_DIR)
            extracted_files = sorted(name for name in archive.namelist() if not name.endswith("/"))
    except (OSError, URLError, TimeoutError, zipfile.BadZipFile) as error:
        source_used = "OpenML mirror of UCI Letter Recognition"
        download_url = LETTER_MIRROR_URL
        if LETTER_ZIP.exists():
            LETTER_ZIP.unlink()
        arff_file = LETTER_DIR / "letter.arff"
        print(f"Official UCI archive unavailable ({error}); using {LETTER_MIRROR_URL}", flush=True)
        download(LETTER_MIRROR_URL, arff_file)
        extracted_files = [arff_file.name]

    csv_path = materialize_letter_csv()

    return {
        "id": "letter_recognition",
        "role": "robustness",
        "source": "UCI Machine Learning Repository",
        "source_page": LETTER_SOURCE_PAGE,
        "mirror_source": source_used,
        "download_url": download_url,
        "mirror_page": LETTER_MIRROR_PAGE,
        "archive": str(LETTER_ZIP.relative_to(PROJECT_ROOT)) if LETTER_ZIP.exists() else None,
        "archive_sha256": sha256(LETTER_ZIP) if LETTER_ZIP.exists() else None,
        "extracted_files": extracted_files,
        "materialized_file": str(csv_path.relative_to(PROJECT_ROOT)),
        "materialized_sha256": sha256(csv_path),
        "samples": 20000,
        "features": 16,
        "target": "letter",
        "classes": 26,
    }


def materialize_digits() -> dict[str, object]:
    """Save the sklearn built-in Digits dataset as a portable raw CSV."""

    DIGITS_DIR.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    columns = [f"pixel_{index:02d}" for index in range(digits.data.shape[1])]
    frame = pd.DataFrame(digits.data, columns=columns)
    frame["target"] = digits.target
    frame.to_csv(DIGITS_CSV, index=False)

    return {
        "id": "handwritten_digits",
        "role": "robustness",
        "source": "scikit-learn load_digits (UCI optical handwritten digits test set)",
        "source_page": DIGITS_SOURCE_PAGE,
        "materialized_file": str(DIGITS_CSV.relative_to(PROJECT_ROOT)),
        "file_sha256": sha256(DIGITS_CSV),
        "samples": int(digits.data.shape[0]),
        "features": int(digits.data.shape[1]),
        "target": "target",
        "classes": len(digits.target_names),
        "feature_range": [0, 16],
    }


def materialize_covertype() -> dict[str, object]:
    """Download and save UCI Covertype as a portable CSV."""

    COVERTYPE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_covtype(as_frame=True, data_home=COVERTYPE_DIR / ".cache")
    frame = dataset.data.copy()
    frame["cover_type"] = dataset.target.astype("int64").to_numpy()
    frame.to_csv(COVERTYPE_CSV, index=False)

    return {
        "id": "covertype",
        "role": "primary",
        "source": "UCI Machine Learning Repository via sklearn.datasets.fetch_covtype",
        "source_page": COVERTYPE_SOURCE_PAGE,
        "materialized_file": str(COVERTYPE_CSV.relative_to(PROJECT_ROOT)),
        "file_sha256": sha256(COVERTYPE_CSV),
        "samples": int(frame.shape[0]),
        "features": int(frame.shape[1] - 1),
        "target": "cover_type",
        "classes": int(frame["cover_type"].nunique()),
        "missing_cells": int(frame.isna().sum().sum()),
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "project": "Lab 2 - Decision Tree Modeling and Improvement",
        "generated_by": "scripts/download_datasets.py",
        "datasets": [download_letter(), materialize_digits(), materialize_covertype()],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
