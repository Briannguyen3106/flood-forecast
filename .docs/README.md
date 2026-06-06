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
- Tree ensembles currently outperform linear models in historical results.
- Random Forest was historically best on the test split, but final results must be regenerated after the target-order and ablation corrections.
- Linear models remain an important part of the course project and are being improved through correct ordinal encoding, model-specific preprocessing, and class/sample weighting.

## Current Handoff

Ablation 4 is complete for Random Forest, Ridge, Lasso, and SVM. The next
contributor should run the three Ablation 4b batches in
`experiments/ablation_study.ipynb`, review
`results/ablation/ablation4_all_winners.csv`, then follow
[`CURRENT_STATUS.md`](CURRENT_STATUS.md). Do not run final test evaluation yet.

Root-level `Highlights.md` contains the longer analysis history. The `.docs` directory contains the maintained onboarding view.
