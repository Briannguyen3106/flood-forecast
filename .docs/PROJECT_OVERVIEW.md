# Project Overview

## Problem

The project predicts flood-risk severity for urban spatial segments using terrain, drainage, rainfall, return-period, and categorical infrastructure/source attributes.

The original `risk_labels` field is multi-label text. It is converted into one class using priority rules:

- `High`: contains `ponding_hotspot` or `extreme_rain_history`.
- `Medium`: otherwise contains `low_lying` or `sparse_drainage`.
- `Low`: monitoring labels, event dates, empty labels, or no stronger risk indicator.

## Dataset

- Source file: `data/raw/urban_pluvial_flood_risk_dataset.xlsx`.
- Total rows: 2,963.
- Train rows: 2,370.
- Test rows: 593.
- Historical train distribution: Low 1,595, Medium 459, High 316.

The class imbalance is operationally important because `High` is the smallest and most costly class to miss.

## Model Families

Tree-based models:

- Decision Tree
- Random Forest
- XGBoost
- XGBRF
- LightGBM/custom LightGBM
- HistGradientBoosting

Linear/distance-based models:

- Ordinal Linear Regression
- Ordinal Ridge Regression
- Ordinal Lasso Regression
- Ordinal Huber Regression
- Custom kernel SVM

The regression models predict a continuous ordinal risk value and convert it to classes using two optimized thresholds. Therefore, correct target order is essential for them.

## Historical Result Snapshot

These results came from the earlier `Train_Test.ipynb` run and are retained only as a baseline:

| Model | Test F2-macro | Test F1-weighted |
|---|---:|---:|
| Random Forest | 0.9315 | 0.9340 |
| XGBoost | 0.9255 | 0.9308 |
| XGBRF | 0.9230 | 0.9291 |
| HistGradientBoosting | 0.9223 | 0.9320 |
| SVM | 0.7149 | 0.7401 |
| Huber | 0.5641 | 0.5855 |

These are pre-regeneration results. Do not cite them as the final corrected experiment until data preparation, ablation, tuning, and final evaluation have been rerun.

## Corrected Experiment Progress

Data preparation and leakage-safe ablation have been rerun. Ablation 4 has
confirmed configurations for Random Forest, Ridge, Lasso, and SVM using only
raw train data and repeated stratified CV. Seven models remain in Ablation 4b
before final tuning can begin.

See `CURRENT_STATUS.md` for the selected setups and exact handoff steps.
Historical test rankings above remain baselines only.
