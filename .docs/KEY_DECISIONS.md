# Key Decisions

## Three-Class Target

The source target is multi-label. The project uses one priority-based class because it is easier to compare across standard multiclass algorithms and remains meaningful for risk management.

## Explicit Ordinal Encoding

The project uses `Low=0, Medium=1, High=2`. This is semantically meaningful for ordinal regression and harmless for proper classifiers, where numeric IDs are only class identifiers.

## F2-Macro as Primary Metric

False negatives for high-risk areas are more serious than additional false alarms. F2 emphasizes recall, while macro averaging prevents the majority Low class from dominating selection.

## Separate Model Pipelines

Tree models and linear/distance models respond differently to scaling, skew correction, encoding, and interaction features. Preprocessing is selected per model family and, when useful, per model.

## Fold-Local Preprocessing

Preprocessing outside CV gives validation data influence over learned transforms. Ablation therefore starts from raw train rows and fits preprocessing separately inside every fold.

## Imbalance Handling Is Tuned

SMOTE is not universally beneficial. The project compares no balancing, SMOTE, balanced sample weights, and explicit risk-priority profiles. The winning strategy may differ by model.

The first corrected Ablation 4 pass confirms this: Random Forest selected
SMOTE, Ridge and Lasso selected no balancing, while SVM selected explicit
`1:2:3` risk weights. Final training must support model-specific imbalance
handling rather than one global SMOTE pipeline.

## Final Model Selection Is Train-Only

Preprocessing, feature, imbalance, threshold, and hyperparameter decisions are
frozen using train-only CV. The test split is reported once afterward. A table
ranked by test score may describe the best test performer, but it must not be
used to revise and rerun model decisions.

## Linear Models Remain in Scope

Tree ensembles are historically strongest, but linear models are retained because this is an educational ML project. Their purpose includes demonstrating optimization, regularization, robust loss, kernels, preprocessing sensitivity, imbalance handling, and model-capacity limitations.
