# Decision Tree Lab 2 - viva preparation

## 1. Why use three datasets when the assignment asks for one?

Covertype is the primary dataset used for all mandatory tasks. Letter and Digits are robustness datasets used to test whether observations generalize to different data regimes.

## 2. Why is Covertype the primary dataset?

It is large, multiclass, imbalanced and produces a deep baseline tree. Both overfitting and structural regularization are therefore easier to observe quantitatively.

## 3. How does CART choose a split?

It evaluates candidate feature-threshold pairs and selects the pair that maximizes weighted Gini reduction.

## 4. Why do you say the baseline overfits?

The tree reaches 100% train accuracy but only 93.89% test accuracy. It also has depth 41 and 23,956 leaves. Zero training error, a 6.11 percentage-point gap and high structural complexity are consistent evidence of high variance.

## 5. Why is E4 preferred when E3 has higher accuracy?

E3 is only 0.04 percentage points more accurate. E4 has the highest Macro-F1, 31.7% fewer leaves and a 17.6% smaller train-test gap than E0. Macro-F1 and complexity matter because Covertype is imbalanced and interpretability is an explicit goal.

## 6. What is the difference between CCP and HS?

CCP changes topology by pruning branches. HS preserves topology and regularizes node probability estimates. Their different mechanisms motivate the E4 combined experiment.

## 7. Why did pre-pruning perform worse?

Early stopping can remove useful fine-grained structure before it is learned. On Covertype, E1 reduces gap and leaves but lowers Accuracy and Macro-F1, which is evidence of underfitting.

## 8. Why use Macro-F1?

Accuracy is dominated by large classes. Macro-F1 gives every class equal weight; this is important because Covertype classes 4 and 5 have much lower support and recall.

## 9. Why is Random Forest weak on Covertype in this experiment?

The result belongs to the tested cuML implementation and hyperparameter configuration. The benchmark was not an exhaustive tuning study, so it does not imply Decision Tree universally outperforms Random Forest.

## 10. Can you claim HS always improves Decision Trees?

No. The gain is very small on Letter and negative on Digits. The strongest supported claim is that CCP + HS gives a favorable complexity-generalization trade-off on the large Covertype tree.

## Submission blocker

Replace every `[HỌ TÊN -- MSSV]`, `[GIẢNG VIÊN]` and `[TÊN NHÓM / MÃ NHÓM]` placeholder with real information before final submission.
