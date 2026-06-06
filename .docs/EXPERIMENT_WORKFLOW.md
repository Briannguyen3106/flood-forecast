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

- Random Forest, Ridge, Lasso, and SVM are already confirmed.
- Run Ablation 4b with `fast`, `decision_tree`, and `lightgbm` batches.
- The batch cell checkpoints after every model; the summary cell writes
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

The notebook is intended for Kaggle, so its clone/install setup may remain.
It validates schema version, code revision, target mapping, model class,
ablation configuration, imbalance strategy, CV settings, and random seed before
loading a PKL. Each completed model is saved atomically. To resume after a
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
