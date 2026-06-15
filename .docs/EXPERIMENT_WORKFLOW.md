# Experiment Workflow

## Recommended Sequence

### 1. Prepare data

Run `experiments/Prepare_Data.ipynb`.

It recreates the stratified raw splits and both processed datasets. Verify the output mapping is:

```text
Low=0, Medium=1, High=2
```

### 2. Run ablation

Run `experiments/ablation_study.ipynb` using raw train data.

Ablation groups:

1. Imbalance strategy:
   - none;
   - SMOTE;
   - automatically balanced sample weights;
   - explicit `Low/Medium/High` profiles: `1/1.5/2`, `1/2/3`, `1/2/4`.
2. Feature groups: baseline, G1, G2, G3, G4, and all groups.
3. Preprocessing: baseline, FE, scale, skew correction, OHE, and combinations.

After these isolated studies, run Ablation 4 as a filtered confirmation pass.
It evaluates only the shortlisted preprocessing, feature, and imbalance
combinations for each model. Use F2-macro first, then High recall,
F1-weighted, variance, and train/CV gap to freeze one configuration per model.

Use repeated stratified CV. Select configurations by mean F2-macro, then inspect variance, train/validation gap, F1-weighted, and class recall.

Current status:

- Ablation 4 and 4b are complete for all 11 models.
- The frozen winners are recorded in
  `results/ablation/ablation4_all_winners.csv`.
- Do not rerun completed ablations unless the split, target mapping,
  preprocessing implementation, or CV logic changes.

### 3. Tune models

Tune each model using the configuration selected by Ablation 4. Avoid forcing
one global preprocessing or imbalance strategy across all models.

The current `Trainer` starts from raw rows, fits the selected preprocessor
inside every tuning fold, and applies the model-specific imbalance strategy
only to fold training rows.

For linear models, compare at least:

- Linear/Ridge: scale-only and selected interactions;
- Lasso: scale-only with very small alpha values included;
- Huber: scale-only versus scale plus skew correction;
- SVM: scale plus skew, selected G1/G4 features, and weighted alternatives.

### 4. Final evaluation

Run `experiments/Train_Test.ipynb` only after model decisions are frozen.

The corrected final run completed on 2026-06-07. The notebook used five-fold
stratified tuning CV with `CV_REPEATS=1`; the earlier ablation stage used its
own repeated-CV protocol to freeze preprocessing and imbalance setups. Do not
describe the final tuning run itself as repeated CV unless `CV_REPEATS` is
increased and all artifacts are regenerated.

The notebook is intended for Kaggle, so its clone/install setup may remain.
It validates schema version, target mapping, model class, ablation
configuration, imbalance strategy, CV settings, and random seed before loading
a PKL. Code revision is retained for provenance and a mismatch produces a
warning, not rejection. Each completed model is saved atomically. To resume after a
Kaggle session is destroyed, publish the prior notebook output or upload its
`saved_models` directory as a Dataset and set `EXTERNAL_MODELS_DIR`.

Report:

- train and test F2-macro;
- train and test F1-weighted;
- train/test gaps;
- per-class precision, recall, and F1;
- confusion matrices;
- selected hyperparameters and imbalance strategy.

## Metric Policy

- Primary: F2-macro.
- Secondary: F1-weighted.
- Operational check: `High` recall.

F2-macro gives equal class influence and emphasizes recall, matching the cost of missing dangerous areas.

## Artifact Policy

Treat the following as generated artifacts:

- `data/processed/**/*.csv`;
- `results/**/*`;
- `saved_models/*.pkl`.

Record the code revision and experiment configuration whenever regenerating final artifacts.

Artifacts retain their training revision for provenance, but a revision
difference alone does not force retraining. The notebook warns and reuses the
checkpoint when all model-contract metadata matches. When model behavior or
preprocessing changes without a corresponding schema/config change, increment
`ARTIFACT_SCHEMA_VERSION` or remove the affected PKL explicitly.
