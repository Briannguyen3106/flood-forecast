from dataclasses import dataclass
from typing import Dict, Tuple

import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from config.domain_ablation_config import DomainAblationConfig


f2_macro_scorer = make_scorer(fbeta_score, beta=2, average="macro", zero_division=0)
f1_weighted_scorer = make_scorer(f1_score, average="weighted", zero_division=0)


@dataclass
class ExperimentResult:
    model: str
    cv_f2: float
    cv_f2_std: float
    cv_f1w: float
    train_f2: float
    gap: float
    test_f2: float
    test_f1w: float


def make_smote_pipe(estimator, config: DomainAblationConfig):
    return ImbPipeline([("smote", SMOTE(random_state=config.random_state, k_neighbors=5)), ("clf", estimator)])


def cv_score(pipe, X, y, config: DomainAblationConfig):
    skf = StratifiedKFold(n_splits=config.cv_splits_tune, shuffle=True, random_state=config.random_state)
    return cross_validate(pipe, X, y, cv=skf, scoring=f2_macro_scorer, n_jobs=-1)["test_score"].mean()


def evaluate_pipeline(pipe, X_train, y_train, X_test, y_test, model_name: str, config: DomainAblationConfig) -> ExperimentResult:
    skf = StratifiedKFold(n_splits=config.cv_splits_eval, shuffle=True, random_state=config.random_state)
    cv_res = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=skf,
        scoring={"f2_macro": f2_macro_scorer, "f1_weighted": f1_weighted_scorer},
        return_train_score=True,
        n_jobs=-1,
    )

    cv_f2 = cv_res["test_f2_macro"].mean()
    cv_f2_std = cv_res["test_f2_macro"].std()
    tr_f2 = cv_res["train_f2_macro"].mean()
    cv_f1w = cv_res["test_f1_weighted"].mean()
    gap = tr_f2 - cv_f2

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_f2 = fbeta_score(y_test, y_pred, beta=2, average="macro", zero_division=0)
    test_f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    return ExperimentResult(
        model=model_name,
        cv_f2=float(cv_f2),
        cv_f2_std=float(cv_f2_std),
        cv_f1w=float(cv_f1w),
        train_f2=float(tr_f2),
        gap=float(gap),
        test_f2=float(test_f2),
        test_f1w=float(test_f1w),
    )


def build_xgb_baseline(config: DomainAblationConfig):
    params = dict(config.xgb_baseline_params)
    params["random_state"] = config.random_state
    return XGBClassifier(**params)


def tune_rf_optuna(X_train, y_train, config: DomainAblationConfig, n_trials: int = 40) -> Tuple[ImbPipeline, float, Dict]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        space = config.rf_tuning_space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", space["n_estimators"][0], space["n_estimators"][1]),
            "max_depth": trial.suggest_categorical("max_depth", space["max_depth"]),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", space["min_samples_split"][0], space["min_samples_split"][1]
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", space["min_samples_leaf"][0], space["min_samples_leaf"][1]
            ),
            "max_features": trial.suggest_categorical("max_features", space["max_features"]),
            "class_weight": trial.suggest_categorical("class_weight", space["class_weight"]),
            "random_state": config.random_state,
            "n_jobs": -1,
        }
        pipe = make_smote_pipe(RandomForestClassifier(**params), config)
        return cv_score(pipe, X_train, y_train, config)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = RandomForestClassifier(**study.best_params, random_state=config.random_state, n_jobs=-1)
    best_pipe = make_smote_pipe(best_model, config)
    return best_pipe, study.best_value, study.best_params
