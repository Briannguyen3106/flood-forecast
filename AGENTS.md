# AGENTS.md

## Purpose

This repository is a machine-learning course project for multiclass urban flood-risk prediction. Read `.docs/README.md` before changing preprocessing, evaluation, or experiment code.

## Critical Invariants

1. Risk classes use the explicit ordinal mapping:

   ```text
   Low=0, Medium=1, High=2
   ```

   Do not replace it with alphabetical `LabelEncoder` output.

2. The test split is final evaluation data. Do not use it for preprocessing selection, feature selection, threshold selection, class-weight tuning, or hyperparameter tuning.

3. During cross-validation, fit imputation, clipping thresholds, engineered-feature thresholds, Yeo-Johnson transforms, encoders, scalers, SMOTE, and model parameters on the training fold only.

4. Never apply SMOTE to validation or test rows.

5. Use F2-macro as the primary selection metric and F1-weighted as the secondary metric. Always inspect per-class recall, especially `High` recall.

6. Tree and linear model families use different preprocessing. Do not assume a preprocessing configuration that helps one family helps the other.

## Current Workflow

Run the project in this order:

1. `experiments/Prepare_Data.ipynb`
2. `experiments/ablation_study.ipynb`
3. Select preprocessing, feature, and imbalance strategies using train-only repeated CV.
4. Tune selected models on train data.
5. Run `experiments/Train_Test.ipynb` for final train/test reporting.

The corrected final artifacts were generated on 2026-06-07. Historical result sections must remain explicitly labeled pre-fix; current post-fix reports are under `results/final/`.

## Code Map

- `src/core/data_preprocessing.py`: target construction, class mapping, Pipeline A and Pipeline B.
- `src/core/ablation_trainer.py`: leakage-safe repeated-CV ablation runner.
- `src/core/trainer.py`: final model tuning from raw rows with fold-local preprocessing.
- `src/model/base_model.py`: model interface.
- `src/model/base_regression.py`: ordinal regression and threshold logic.
- `src/model/`: model implementations.
- `experiments/`: notebooks for preparation, ablation, individual experiments, and final comparison.

## Change Rules

- Keep changes scoped and preserve existing model interfaces.
- Add model-specific preprocessing only after validating it through ablation.
- When adding a model, implement `build()`, `get_param_distributions()`, sklearn-compatible `get_params()`/`set_params()`, and set `pipeline_type`.
- Keep random seeds explicit; the project standard is `random_state=42`.
- Do not overwrite historical result descriptions without stating whether they are pre-fix or post-fix results.
- Do not commit virtual environments, generated caches, or large regenerated model files unless the team explicitly wants them versioned.

## Verification

At minimum after Python changes:

```powershell
python -m compileall -q src
git diff --check
```

For preprocessing or model changes, also run a small fold-local smoke test before launching the full ablation study.

## Documentation

Update the relevant file under `.docs/` when changing:

- class definitions or target mapping;
- preprocessing behavior;
- experiment protocol;
- model selection conclusions;
- known limitations or artifact status.
