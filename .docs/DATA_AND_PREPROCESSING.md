# Data and Preprocessing

## Target Contract

The canonical mapping is defined in `src/core/data_preprocessing.py`:

```python
RISK_CLASS_TO_INT = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}
```

This order is required by ordinal regression. Alphabetical encoding such as `High=0, Low=1, Medium=2` is invalid for that model family.

## Split Contract

`DataSpliter` performs an 80/20 stratified split with `random_state=42`.

- `data/splits/train.csv`: all model selection and tuning.
- `data/splits/test.csv`: final evaluation only.

Do not repeatedly inspect test performance while deciding features, preprocessing, imbalance handling, or hyperparameters.

## Shared Processing

Both preprocessing pipelines perform:

- target construction;
- replacement of elevation sentinel `-3.0` with missing data;
- median numeric imputation;
- `Unknown` categorical imputation;
- clipping of extreme storm-drain proximity;
- optional domain feature engineering.

Engineered feature groups:

- G1: infrastructure vulnerability ratio.
- G2: rainfall intensity multiplied by return period.
- G3: soil group multiplied by rainfall.
- G4: very-low-elevation indicator.

## Pipeline A: Tree Models

Pipeline A keeps a compact representation suitable for tree splits. Scaling is generally unnecessary, and categorical values use compact encodings.

## Pipeline B: Linear and Distance Models

Pipeline B can apply:

- Yeo-Johnson transforms for skewed features;
- robust scaling;
- one-hot encoding or ordinal categorical encoding;
- selected engineered features.

Do not assume the full Pipeline B is optimal. Historical ablation suggested:

- scaling was essential for Ridge, Lasso, and SVM;
- Ridge preferred scale-only;
- SVM preferred scale plus skew correction;
- OHE and all engineered features together did not consistently help.

The corrected Ablation 4 and 4b runs confirmed all final configurations:

- Random Forest: Pipeline A baseline, no engineered groups, with SMOTE;
- Decision Tree: Pipeline A with G3, without balancing;
- XGBoost: Pipeline A with G3, without balancing;
- XGBRF: Pipeline A baseline with SMOTE;
- custom LightGBM: Pipeline A with G3, without balancing;
- HistGradientBoosting: Pipeline A with G3 and the estimator's balanced class weighting;
- Linear Regression: scale-only baseline with `1:1.5:2` risk weights;
- Ridge: scale-only baseline, without balancing;
- Lasso: OHE-only baseline, without balancing;
- Huber: scale-only baseline with `1:1.5:2` risk weights;
- SVM: scale plus skew correction, baseline features, with `1:2:3` risk weights.

## Leakage Rule

In CV experiments, the preprocessor must be fit inside each training fold. The following values must not be learned from validation rows:

- medians;
- clipping bounds;
- G4 elevation threshold;
- Yeo-Johnson lambda values;
- category vocabularies;
- scaler statistics.

`AblationTrainer._run_raw_cv()` implements this fold-local behavior.
