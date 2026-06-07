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

These are pre-fix results retained only as a historical baseline. Do not cite
them as the final corrected experiment.

## Corrected Final Results

The corrected workflow was completed on 2026-06-07. Ablation froze one setup
per model using raw train data only. Final tuning also started from raw train
rows and fitted preprocessing independently inside each stratified CV fold.
The 593-row test split was evaluated only after all 11 artifacts passed the
notebook compatibility gate.

| Rank | Model | Tuning CV F2-macro | Test F2-macro | Test F1-weighted | High recall |
|---:|---|---:|---:|---:|---:|
| 1 | HistGradientBoosting | 0.9338 | 0.9388 | 0.9392 | 0.9747 |
| 2 | LightGBM | 0.9098 | 0.9386 | 0.9483 | 0.9494 |
| 3 | XGBoost | 0.9020 | 0.9294 | 0.9400 | 0.9241 |
| 4 | Decision Tree | 0.8966 | 0.9271 | 0.9372 | 0.9367 |
| 5 | Random Forest | 0.9003 | 0.9160 | 0.9234 | 0.9367 |
| 6 | XGBRF | 0.9030 | 0.8984 | 0.9164 | 0.8861 |
| 7 | SVM | 0.7386 | 0.7964 | 0.8330 | 0.8228 |
| 8 | Ridge | 0.5999 | 0.6238 | 0.6639 | 0.6835 |
| 9 | Linear Regression | 0.6031 | 0.6101 | 0.6553 | 0.7722 |
| 10 | Huber | 0.6047 | 0.6098 | 0.6521 | 0.7722 |
| 11 | Lasso | 0.5778 | 0.5820 | 0.6427 | 0.7595 |

HistGradientBoosting is both the model selected by train-only tuning CV and
the best test performer. LightGBM has the highest test F1-weighted score, but
the primary selection metric remains F2-macro. See `CURRENT_STATUS.md` for
artifact provenance and limitations.
