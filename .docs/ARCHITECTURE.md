# Architecture

## Repository Map

```text
config/                 Domain ablation configuration
data/raw/               Original dataset
data/splits/            Stratified raw train/test splits
data/processed/         Materialized tree and linear datasets
experiments/            EDA, preparation, ablation, and training notebooks
results/                Generated figures and final metrics
saved_models/           Serialized fitted pipelines/models
src/core/               Data, training, evaluation, and ablation infrastructure
src/model/              Model implementations
.docs/                  Maintained onboarding documentation
```

## Core Components

### `data_preprocessing.py`

Owns target creation, canonical class mapping, feature engineering, Pipeline A, and Pipeline B.

### `ablation_trainer.py`

Runs repeated stratified CV for:

- imbalance handling;
- feature groups;
- preprocessing steps.

Its raw-data CV path fits preprocessing independently in every fold.

`run_ablation_final_configs()` powers Ablation 4 and Ablation 4b. The notebook
splits slower scratch models into separate batches and writes a CSV checkpoint
after each model.

### `trainer.py`

Tunes a model from raw train rows with fold-local preprocessing. It supports
no balancing, SMOTE, balanced sample weights, and explicit class weights.
Each hyperparameter candidate is evaluated with stratified CV, then the chosen
configuration is refit on all raw train rows. The serialized artifact contains
the fitted preprocessor, estimator, feature names, target mapping, imbalance
configuration, CV settings, and selected hyperparameters.

### `BaseModel`

Model wrappers should provide:

- `pipeline_type`: `tree` or `linear`;
- `build(**params)`;
- `get_param_distributions()`;
- sklearn-compatible parameter access for cloning and search;
- `fit()` and `predict()` behavior.

### `BaseOrdinalRegression`

Implements mini-batch gradient descent, continuous prediction, threshold optimization, and optional sample-weighted fitting. Linear, Ridge, Lasso, and Huber provide their own loss and gradient functions.

### Custom SVM

Implements kernel SVM with one-vs-rest multiclass handling and a custom SMO solver. Sample weights alter each training row's soft-margin bound.

### Custom LightGBM

`lgbm_scratch_v2.py` implements histogram boosting, GOSS, optional EFB, and
multiclass trees. Its optimized path now:

- shares percentile edges and binned matrices across class trees in one iteration;
- builds gradient, Hessian, and count histograms with `numpy.bincount`;
- uses cumulative counts for split eligibility;
- partitions rows by node for vectorized prediction.

The optimization preserves the previous per-iteration binning semantics.
Compilation, a smoke test, and old/new traversal equivalence checks pass.

## Important Limitation

`main.py` is legacy code for a different target (`SUSCEP`) and references a nonexistent validation split. It is not the current project entry point. Use the notebooks listed in the experiment workflow.
