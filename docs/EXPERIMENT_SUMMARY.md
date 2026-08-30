# Experiment summary

## Scope

- **Primary case study:** Covertype. It is used for the complete baseline, resulting-tree analysis, error analysis and Decision Tree improvement comparison required by Lab 2.
- **Robustness datasets:** Letter Recognition and Handwritten Digits. They test whether conclusions generalize across multiclass geometric features and small image data.
- **Reference models:** Random Forest, SVM-RBF and KNN. They contextualize Decision Tree; they are not the research focus.

## Shared protocol

| Item | Value |
|---|---|
| Split | Stratified 80/20 |
| Random state | 42 |
| Hyperparameter selection | 3-fold CV on train only |
| Covertype stability check | 5-fold CV on train |
| Scaling | Train-fitted StandardScaler for SVM/KNN only |
| Final test | Never used for tuning |

## Baseline Covertype

```text
DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    ccp_alpha=0.0,
    random_state=42,
)
```

| Accuracy | Error rate | Macro-F1 | Train-test gap | Depth | Leaves |
|---:|---:|---:|---:|---:|---:|
| 93.89% | 6.11% | 90.34% | 6.11 pp | 41 | 23,956 |

Root split: `Elevation <= 3044.5`. The full tree has 23,956 leaves, so the report and slides show only the top levels and three representative real paths.

## Decision Tree improvements

The three required methods are:

1. **E1 Pre-pruning:** controls stopping conditions.
2. **E2 Cost-Complexity Pruning:** removes weak subtrees and changes topology.
3. **E3 Hierarchical Shrinkage:** smooths node predictions without changing topology.

**E4 CCP + HS** is a combined extension, not a fourth required method.

| ID | Method | Accuracy | Error | Macro-F1 | Gap | Depth | Leaves |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | Baseline | 93.89% | 6.11% | 90.34% | 6.11% | 41 | 23,956 |
| E1 | Pre-pruning | 92.97% | 7.03% | 88.85% | 4.10% | 40 | 16,637 |
| E2 | CCP | 93.92% | 6.08% | 90.45% | 5.07% | 39 | 16,361 |
| E3 | HS | **93.97%** | **6.03%** | 90.48% | 5.70% | 41 | 23,956 |
| E4 | CCP + HS | 93.93% | 6.07% | **90.53%** | **5.04%** | 39 | **16,361** |

E4 is selected because it gives the highest Macro-F1, 31.7% fewer leaves and a 17.6% smaller gap than E0, while losing only 0.04 percentage points of accuracy relative to E3.

## Reproduction map

| Result | Notebook | Main artifact |
|---|---|---|
| Letter baseline | `notebooks/decision_tree/01_letter_decision_tree_baseline.ipynb` | `dt_letter_baseline.json` |
| Digits baseline | `notebooks/decision_tree/02_digits_decision_tree_baseline.ipynb` | `dt_digits_baseline.json` |
| Covertype baseline/CV | `notebooks/decision_tree/03_covertype_decision_tree_scalability.ipynb` | `dt_covertype_scalability.json` |
| E0-E4 and train-only selection | `notebooks/decision_tree/04_hierarchical_shrinkage_three_dataset_benchmark.ipynb` | `dt_hierarchical_shrinkage__summary.csv` |
| 3 datasets x 4 models | `notebooks/benchmark_models/05_three_dataset_three_model_benchmark.ipynb` | `gpu_three_dataset_four_model_comparison__summary.csv` |

Report-ready artifacts are under `docs/report/artifacts/`. Final source files are:

- `docs/report/decision_tree_lab2_report.tex`
- `docs/slides/decision_tree_lab2_presentation.tex`

