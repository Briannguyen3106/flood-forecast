# Current Status

Last updated: 2026-06-06.

## Completed in Code

- Canonical target mapping changed to `Low=0, Medium=1, High=2`.
- Final-report label order changed to Low, Medium, High.
- Ablation preprocessing moved inside CV folds.
- Linear, Ridge, Lasso, and Huber support sample-weighted losses and threshold selection.
- Custom SVM supports sample-weighted soft-margin bounds.
- Imbalance ablation now includes none, SMOTE, balanced weights, and explicit weight profiles.
- Current split and processed artifacts were regenerated with the corrected target order.
- Ablation 1-3 and the first Ablation 4 pass were run with fold-local preprocessing.
- Ablation 4 confirmed setups for Random Forest, Ridge, Lasso, and SVM.
- `ablation_study.ipynb` now contains checkpointed Ablation 4b batches for the seven remaining models.
- Custom LightGBM now reuses multiclass binning, uses cumulative histograms, and vectorizes traversal.
- Ablation 4b is complete and `ablation4_all_winners.csv` contains all 11 models.
- `Trainer` now tunes from raw rows with fold-local preprocessing and supports
  model-specific none, SMOTE, balanced, and explicit-weight strategies.
- `Train_Test.ipynb` now uses raw splits and the 11 frozen Ablation 4 setups.
  It checkpoints each completed model, validates artifact metadata before
  reuse, and blocks final test evaluation until all artifacts are compatible.

## Confirmed Ablation 4 Setups

These are post-fix train-only repeated-CV results, not test results:

| Model | Selected setup | CV F2-macro | High recall |
|---|---|---:|---:|
| Random Forest | Pipeline A baseline + SMOTE | 0.8647 | 0.7808 |
| SVM | scale + skew + weights `1:2:3` | 0.7186 | 0.6931 |
| Ridge | scale-only + no balancing | 0.5887 | 0.6772 |
| Lasso | OHE-only + no balancing | 0.5743 | 0.6500 |

Random Forest has a train/CV F2 gap of 0.1353 and SVM has a gap of 0.1090,
so later tuning should include meaningful regularization.

## Ablation 4b Pending

Pending models: Decision Tree, XGBoost, XGBRF, custom LightGBM,
HistGradientBoosting, Linear Regression, and Huber Regression.

Run the final notebook cells in separate batches:

```python
ABLATION4B_BATCH = "fast"
ABLATION4B_BATCH = "decision_tree"
ABLATION4B_BATCH = "lightgbm"
```

The `fast` batch covers XGBoost, XGBRF, HistGradientBoosting, Linear, and
Huber. CSV checkpoints are written after every model.

## Must Be Regenerated

- Linear model tuning results.
- Final train/test metrics and confusion matrices.
- All serialized final model files and their configuration metadata.

Historical metrics in `Highlights.md` remain useful as a baseline but are not the final corrected results.

## Known Issues

- Regression thresholds are optimized on the same fit rows. Out-of-fold threshold optimization would provide a cleaner estimate and is a recommended improvement.
- The repository has no formal automated test suite; current validation relies on compilation and smoke tests.
- `requirements.txt` does not list every library referenced by all optional models/notebooks, such as plotting and some boosting/tuning packages.
- `main.py` is stale and should not be used as the project entry point.
- Several personal experiment notebooks still contain comments or labels for the old class mapping. Update them before rerunning those notebooks.

## Next Actions

1. Run the checkpointed tuning cell on Kaggle from raw train data.
2. Confirm all 11 compatible artifacts before entering the final-test cells.
3. Run the final test comparison once, then update historical conclusions.
