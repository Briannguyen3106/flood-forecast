# Project Documentation

Start here when joining the project.

## Reading Order

1. [Project Overview](PROJECT_OVERVIEW.md)
2. [Data and Preprocessing](DATA_AND_PREPROCESSING.md)
3. [Architecture](ARCHITECTURE.md)
4. [Experiment Workflow](EXPERIMENT_WORKFLOW.md)
5. [Key Decisions](KEY_DECISIONS.md)
6. [Current Status](CURRENT_STATUS.md)

## One-Minute Summary

- Task: classify urban segments into `Low`, `Medium`, or `High` flood risk.
- Dataset: 2,963 rows, split 80/20 into 2,370 train and 593 test rows.
- Primary metric: **F2-macro** because missing hazardous areas is costlier than issuing extra warnings.
- Target mapping: `Low=0`, `Medium=1`, `High=2`.
- Tree ensembles outperform the linear models in the corrected final run.
- HistGradientBoosting was selected by train-only tuning CV and was also the best test performer: **F2-macro 0.9388** (test) and **F1-weighted 0.9392** (test).
- Linear models remain part of the course project for comparison.

## Repository Layout (cheat-sheet)

```
flood-forecast/
├─ config/                      # persisted config + frozen estimators
├─ data/
│  ├─ raw/                      # ⛔ do not modify
│  ├─ splits/                   # train/test used for evaluation (may be regenerated)
│  └─ processed/                # feature-engineered training artifacts (tree vs linear)
├─ experiments/                 # notebooks (prepare/ablation/final run)
├─ results/
│  └─ final/                   # final metrics + plots + per-class breakdown
├─ saved_models/               # frozen model artifacts (.pkl)
├─ scripts/                    # utility scripts
└─ src/
   ├─ core/                    # leakage-safe pipeline/training/evaluation
   └─ model/                   # model implementations
```

## Current Handoff

The corrected workflow is complete through final evaluation. All frozen
model configurations were tuned from raw train rows; final reports live under
`results/final/`. See [`CURRENT_STATUS.md`](CURRENT_STATUS.md) for metrics and
artifact details.

Root-level `Highlights.md` contains the longer analysis history. `AGENTS.md`
contains the change rules and invariants for contributors.

