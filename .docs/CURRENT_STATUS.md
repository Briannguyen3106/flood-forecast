# Current Status

Last updated: 2026-06-07.

## Corrected Workflow Complete

- Canonical target mapping is `Low=0, Medium=1, High=2`.
- Raw splits and processed artifacts were regenerated after the mapping fix.
- Ablation preprocessing is fitted inside each training fold.
- Ablation 4 and 4b froze one preprocessing, feature, and imbalance setup for
  each of the 11 models using train data only.
- `Trainer` tunes from raw rows and supports none, SMOTE, balanced sample
  weights, and explicit class weights.
- `Train_Test.ipynb` validates artifact schema, target mapping, model class,
  frozen setup, CV settings, and random seed before reuse, while retaining the
  training revision as provenance and warning when it differs.
- The final test gate opened only after all 11 artifacts were compatible.
- All 12 notebook cells completed without an error on 2026-06-07.

## Final Model Setups

| Model | Frozen setup |
|---|---|
| Decision Tree | Pipeline A + G3 + no balancing |
| Random Forest | Pipeline A baseline + SMOTE |
| XGBoost | Pipeline A + G3 + no balancing |
| XGBRF | Pipeline A baseline + SMOTE |
| custom LightGBM | Pipeline A + G3 + no balancing |
| HistGradientBoosting | Pipeline A + G3 + estimator class weight `balanced` |
| Linear Regression | Pipeline B scale-only + weights `1:1.5:2` |
| Ridge | Pipeline B scale-only + no balancing |
| Lasso | Pipeline B OHE-only + no balancing |
| Huber | Pipeline B scale-only + weights `1:1.5:2` |
| SVM | Pipeline B scale + skew + weights `1:2:3` |

For HistGradientBoosting, the notebook records trainer imbalance as `none`
because balancing is configured inside the estimator wrapper. Applying trainer
sample weights as well would duplicate the selected strategy.

## Corrected Final Results

Final tuning used raw train rows, fold-local preprocessing, five stratified
folds, `CV_REPEATS=1`, `N_ITER=100`, and `random_state=42`. The test set has
593 rows and was used only after tuning and artifact validation completed.

| Test rank | Model | CV F2-macro | Test F2-macro | Test F1-weighted | High recall |
|---:|---|---:|---:|---:|---:|
| 1 | HistGB | 0.9338 | 0.9388 | 0.9392 | 0.9747 |
| 2 | LightGBM | 0.9098 | 0.9386 | 0.9483 | 0.9494 |
| 3 | XGBoost | 0.9020 | 0.9294 | 0.9400 | 0.9241 |
| 4 | DecisionTree | 0.8966 | 0.9271 | 0.9372 | 0.9367 |
| 5 | RandomForest | 0.9003 | 0.9160 | 0.9234 | 0.9367 |
| 6 | XGBRF | 0.9030 | 0.8984 | 0.9164 | 0.8861 |
| 7 | SVM | 0.7386 | 0.7964 | 0.8330 | 0.8228 |
| 8 | Ridge | 0.5999 | 0.6238 | 0.6639 | 0.6835 |
| 9 | LinearRegression | 0.6031 | 0.6101 | 0.6553 | 0.7722 |
| 10 | Huber | 0.6047 | 0.6098 | 0.6521 | 0.7722 |
| 11 | Lasso | 0.5778 | 0.5820 | 0.6427 | 0.7595 |

HistGB was selected by the primary train-only tuning metric and was also the
best test performer. Its test confusion matrix correctly classified 77 of 79
`High` rows. LightGBM achieved the highest test F1-weighted score, but model
selection remains based on F2-macro.

## Artifacts

Current reports:

- `results/final/final_metrics.csv`
- `results/final/per_class_metrics.csv`
- `results/final/detailed_metrics.json`
- `results/final/model_comparison.png`
- `results/final/confusion_matrices_top3.png`
- `results/final/training_checkpoint_manifest.csv`
- `saved_models/*.pkl` for all 11 models

The artifacts retain their training code revision for provenance. Revision
differences now produce a warning rather than automatic rejection. Reuse still
requires matching artifact schema, target mapping, model class, frozen config,
imbalance strategy, class weights, CV settings, and random seed. Delete an
artifact manually when a code change alters behavior without changing one of
those contract fields.

## Known Limitations

- Regression thresholds are optimized on the same fit rows. Out-of-fold
  threshold optimization would provide a cleaner estimate.
- Final tuning uses five-fold stratified CV with one repeat. Do not call this
  stage repeated CV; repeated CV was used during configuration ablation.
- The repository has no formal automated test suite.
- `requirements.txt` does not list every plotting or optional model dependency.
- `main.py` is stale and is not the project entry point.
- Personal experiment notebooks may still contain old target-mapping comments.
- The final test has now been observed. Do not revise configurations from these
  test scores and rerun against the same split as if it were unseen data.

## Next Work

1. Update the written report and presentation using the post-fix tables.
2. Preserve Section 12 of `Highlights.md` only as a labeled pre-fix baseline.
3. For a future experiment revision, use new holdout data or nested/OOF
   procedures rather than tuning decisions against the current test results.
