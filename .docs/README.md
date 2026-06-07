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
- Primary metric: F2-macro because missing hazardous areas is costlier than issuing extra warnings.
- Target mapping: `Low=0`, `Medium=1`, `High=2`.
- Tree ensembles outperform the linear models in the corrected final run.
- HistGradientBoosting was selected by train-only tuning CV and was also the best test performer: F2-macro `0.9388`, F1-weighted `0.9392`, and `High` recall `0.9747`.
- Linear models remain part of the course project for comparison and for demonstrating ordinal regression, regularization, kernels, and model-specific preprocessing.

## Current Handoff

The corrected workflow is complete through final evaluation. All 11 frozen
model configurations were tuned from raw train rows, all compatible artifacts
were present before the test gate opened, and the final reports are under
`results/final/`. See [`CURRENT_STATUS.md`](CURRENT_STATUS.md) for metrics,
artifact details, and remaining limitations.

Root-level `Highlights.md` contains the longer analysis history. The `.docs` directory contains the maintained onboarding view.
