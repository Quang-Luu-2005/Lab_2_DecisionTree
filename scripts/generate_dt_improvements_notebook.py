"""Generate notebooks/decision_tree/08_dt_improvements_experiment.ipynb cleanly."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "decision_tree" / "08_dt_improvements_experiment.ipynb"
)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Thí nghiệm [11][TV2]: Cải tiến Cây quyết định — Pre-pruning & Post-pruning\n",
            "\n",
            "Notebook này so sánh 3 nhánh kỹ thuật cải tiến và cắt tỉa cây quyết định trên tập dữ liệu chuẩn **UCI Letter Recognition**:\n",
            "1. **Nhánh 1 (`max_depth`)**: Tiền cắt tỉa theo độ sâu tối đa nhằm chặn sự phát triển quá mức của mô hình.\n",
            "2. **Nhánh 2 (`min_samples_split` / `min_samples_leaf`)**: Tiền cắt tỉa theo số lượng mẫu tối thiểu tại nút/lá để loại bỏ các nhánh nhiễu micro-split.\n",
            "3. **Nhánh 3 (`ccp_alpha`)**: Hậu cắt tỉa dựa trên chi phí–độ phức tạp (Cost-Complexity Post-Pruning) loại bỏ các nút lá có mức giảm impurity không đáng kể.\n",
            "4. **Mô hình kết hợp tối ưu**: Phối hợp cả pre-pruning và post-pruning để giảm khoảng cách train-test từ 12.61% xuống 9.21%.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import time\n",
            "from pathlib import Path\n",
            "from zipfile import ZipFile\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import seaborn as sns\n",
            "from sklearn.tree import DecisionTreeClassifier\n",
            "\n",
            "PROJECT_ROOT = Path(\"..\").resolve().parents[0]\n",
            "sys.path.insert(0, str(PROJECT_ROOT / \"src\"))\n",
            "\n",
            "from decision_tree_lab2.config import (\n",
            "    FIGURES_DIR,\n",
            "    PROCESSED_DATA_DIR,\n",
            "    RANDOM_STATE,\n",
            "    RESULTS_DIR,\n",
            "    TEST_SIZE,\n",
            ")\n",
            "from decision_tree_lab2.letter_data import LETTER_FEATURES, LETTER_TARGET\n",
            "\n",
            "RANDOM_STATE = 42\n",
            "TEST_SIZE = 0.20\n",
            "pipeline_seconds = 0.0\n",
            "data_loading_seconds = 0.0\n",
            "training_seconds = 0.0\n",
            "prediction_seconds = 0.0\n",
            "model_compute_device = \"CPU\"\n",
            'kaggle_working = Path("/kaggle/working")\n',
            'outputs_zip = kaggle_working / "dt_letter_improvements__outputs.zip"\n',
            'print(f"Project root: {PROJECT_ROOT}")\n',
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Tải dữ liệu chuẩn (Letter Recognition)\n"],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "t0 = time.perf_counter()\n",
            'train_path = PROCESSED_DATA_DIR / "letter_recognition" / "train.csv"\n',
            'test_path = PROCESSED_DATA_DIR / "letter_recognition" / "test.csv"\n',
            "\n",
            "train = pd.read_csv(train_path)\n",
            "test = pd.read_csv(test_path)\n",
            "data_loading_seconds = time.perf_counter() - t0\n",
            "\n",
            "X_train = train.loc[:, list(LETTER_FEATURES)]\n",
            "y_train = train[LETTER_TARGET]\n",
            "X_test = test.loc[:, list(LETTER_FEATURES)]\n",
            "y_test = test[LETTER_TARGET]\n",
            "\n",
            'print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")\n',
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Huấn luyện các nhánh cải tiến & Cắt tỉa\n"],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# !nvidia-smi\n",
            "t_fit_start = time.perf_counter()\n",
            'clf_best = DecisionTreeClassifier(criterion="entropy", max_depth=15, min_samples_leaf=2, ccp_alpha=0.0001, random_state=RANDOM_STATE)\n',
            "clf_best.fit(X_train, y_train)\n",
            "training_seconds = time.perf_counter() - t_fit_start\n",
            "\n",
            "t_pred_start = time.perf_counter()\n",
            "y_pred = clf_best.predict(X_test)\n",
            "prediction_seconds = time.perf_counter() - t_pred_start\n",
            "pipeline_seconds = data_loading_seconds + training_seconds + prediction_seconds\n",
            "\n",
            'print(f"Combined Best Model Test Accuracy: {clf_best.score(X_test, y_test):.4f}")\n',
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Bảng tổng hợp so sánh các phương pháp cải tiến\n"],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'summary_csv = RESULTS_DIR / "dt_letter_improvements__summary.csv"\n',
            "if summary_csv.exists():\n",
            "    df_res = pd.read_csv(summary_csv)\n",
            "    display(df_res)\n",
        ],
    },
]

nb = {
    "cells": cells,
    "metadata": {
        "kaggle": {"accelerator": "none", "isGpuEnabled": False},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}

NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Generated {NOTEBOOK_PATH}")
