# Custom Decision Tree comparison

This bonus experiment implements a small CART classifier in
`src/decision_tree_lab2/custom_tree.py` without importing scikit-learn inside
the model implementation. It searches binary numeric thresholds by Gini impurity
and supports `max_depth`, `min_samples_split`, `min_samples_leaf` and class
probabilities.

The comparison script is:

```powershell
python scripts/compare_custom_tree.py
```

Both estimators use the same `load_digits()` dataset, stratified 80/20 split,
Gini criterion, `random_state=42`, `max_depth=8` and `min_samples_leaf=1`.
The result is stored in `results/custom_vs_sklearn_decision_tree.json`, with a
tabular copy in `results/custom_vs_sklearn_decision_tree__comparison.csv` and a
readable custom tree in `results/custom_vs_sklearn_decision_tree__custom_tree.txt`.

Small prediction differences are expected: this educational implementation and
scikit-learn can choose different equally-good tied splits. The comparison is
therefore reported by accuracy, macro-F1, tree complexity and test prediction
agreement rather than requiring bit-identical tree topology.

## End-to-end validation

The local data pipeline is validated with:

```powershell
python scripts/prepare_letter_recognition.py
python scripts/run_letter_eda.py
python scripts/run_end_to_end_check.py
```

`run_end_to_end_check.py` records the result in
`results/end_to_end_pipeline_validation.json`. It runs load, schema/preprocessing
validation, the shared split, fit, predict and metrics for Letter Recognition,
Handwritten Digits and Covertype. The optional GPU-only four-model benchmark is
kept separate because it requires a RAPIDS/cuML GPU environment.
