from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.metrics import f1_score, fbeta_score, recall_score
from sklearn.model_selection import ParameterSampler, RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.core.data_preprocessing import RISK_CLASS_TO_INT, assign_class
from src.model.base_model import BaseModel


DROP_COLS = ['risk_class', 'risk_class_encoded', 'latitude', 'longitude']


def _get_processed_X_y(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if feature_cols is None:
        feature_cols = [column for column in df.columns if column not in DROP_COLS]
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Processed data is missing features: {sorted(missing)}")
    return (
        df[feature_cols].to_numpy(),
        df['risk_class_encoded'].to_numpy(),
        feature_cols,
    )


def _raw_target(df: pd.DataFrame) -> np.ndarray:
    if 'risk_labels' not in df.columns:
        raise ValueError(
            "Trainer expects raw rows containing 'risk_labels'; globally "
            "processed CSV files are not valid tuning input."
        )
    return df['risk_labels'].apply(assign_class).map(RISK_CLASS_TO_INT).to_numpy()


@dataclass
class RawModelPipeline:
    """Serializable fitted preprocessor and estimator for raw-row prediction."""

    preprocessor: object
    estimator: object
    feature_cols: list[str]
    metadata: dict

    @property
    def named_steps(self) -> dict:
        return {'preprocessor': self.preprocessor, 'model': self.estimator}

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        processed = self.preprocessor.transform(raw_df.copy())
        X, _, _ = _get_processed_X_y(processed, self.feature_cols)
        return self.estimator.predict(X)


class Trainer:
    """Leakage-safe hyperparameter tuning from raw training rows.

    For every CV fold, preprocessing is fitted on the training fold only.
    SMOTE or sample weights are also applied only to that fold's training rows.
    """

    VALID_IMBALANCE = {'none', 'smote', 'balanced', 'weighted'}

    def __init__(
        self,
        model: BaseModel,
        preprocessor_factory: Callable[[], object],
        imbalance: str = 'none',
        class_weights: Optional[dict[int, float]] = None,
        n_iter: int = 50,
        random_state: int = 42,
        cv: int = 5,
        n_repeats: int = 1,
        config_name: Optional[str] = None,
    ):
        if not isinstance(model, BaseModel):
            raise TypeError("model must inherit BaseModel")
        if not callable(preprocessor_factory):
            raise TypeError("preprocessor_factory must be callable")
        if imbalance not in self.VALID_IMBALANCE:
            raise ValueError(f"Unknown imbalance strategy: {imbalance}")
        if imbalance == 'weighted' and not class_weights:
            raise ValueError("class_weights are required for weighted strategy")
        if cv < 2 or n_repeats < 1:
            raise ValueError("cv must be >= 2 and n_repeats must be >= 1")

        self.model = model
        self.preprocessor_factory = preprocessor_factory
        self.imbalance = imbalance
        self.class_weights = class_weights
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        self.n_repeats = n_repeats
        self.config_name = config_name

        self.pipeline: Optional[RawModelPipeline] = None
        self.best_params: Optional[dict] = None
        self.best_cv_score: Optional[float] = None
        self.cv_results_: list[dict] = []
        self.train_metrics: dict = {}
        self.test_metrics: dict = {}

    def _fit_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        fit_weights = None
        if self.imbalance == 'smote':
            minority_count = int(np.bincount(y).min())
            if minority_count < 2:
                raise ValueError("SMOTE needs at least two minority-class rows")
            X, y = SMOTE(
                k_neighbors=min(5, minority_count - 1),
                random_state=self.random_state,
            ).fit_resample(X, y)
        elif self.imbalance == 'balanced':
            fit_weights = compute_sample_weight('balanced', y)
        elif self.imbalance == 'weighted':
            fit_weights = np.asarray(
                [self.class_weights.get(int(label), 1.0) for label in y],
                dtype=float,
            )
        return X, y, fit_weights

    @staticmethod
    def _fit_estimator(estimator, X, y, fit_weights):
        if fit_weights is None:
            estimator.fit(X, y)
        else:
            try:
                estimator.fit(X, y, sample_weight=fit_weights)
            except TypeError as exc:
                raise TypeError(
                    f"{type(estimator).__name__} does not support sample_weight, "
                    "but the selected imbalance strategy requires it"
                ) from exc
        return estimator

    def _build_estimator(self, params: dict):
        wrapper = clone(self.model)
        wrapper.build(**params)
        return wrapper.model

    def _candidate_params(self) -> list[dict]:
        distributions = self.model.get_param_distributions()
        if not distributions:
            return [{}]
        return list(ParameterSampler(
            distributions,
            n_iter=self.n_iter,
            random_state=self.random_state,
        ))

    def tune(self, raw_train_df: pd.DataFrame):
        y_split = _raw_target(raw_train_df)
        cv = RepeatedStratifiedKFold(
            n_splits=self.cv,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        folds = list(cv.split(raw_train_df, y_split))
        candidates = self._candidate_params()

        

        self.cv_results_ = []
        for candidate_index, params in enumerate(candidates, start=1):
            val_f2, val_f1w, train_f2, high_recalls = [], [], [], []
            for train_idx, val_idx in folds:
                raw_fold_train = raw_train_df.iloc[train_idx].copy()
                raw_fold_val = raw_train_df.iloc[val_idx].copy()

                preprocessor = self.preprocessor_factory()
                train_processed = preprocessor.fit_transform(raw_fold_train)
                val_processed = preprocessor.transform(raw_fold_val)
                X_train, y_train, feature_cols = _get_processed_X_y(train_processed)
                X_val, y_val, _ = _get_processed_X_y(val_processed, feature_cols)
                X_train_eval, y_train_eval = X_train.copy(), y_train.copy()

                X_fit, y_fit, fit_weights = self._fit_imbalance(X_train, y_train)
                estimator = self._build_estimator(params)
                self._fit_estimator(estimator, X_fit, y_fit, fit_weights)

                train_pred = estimator.predict(X_train_eval)
                val_pred = estimator.predict(X_val)
                train_f2.append(fbeta_score(
                    y_train_eval, train_pred, beta=2, average='macro', zero_division=0
                ))
                val_f2.append(fbeta_score(
                    y_val, val_pred, beta=2, average='macro', zero_division=0
                ))
                val_f1w.append(f1_score(
                    y_val, val_pred, average='weighted', zero_division=0
                ))
                high_recalls.append(recall_score(
                    y_val, val_pred, labels=[2], average=None, zero_division=0
                )[0])

            result = {
                'params': params,
                'mean_test_f2': float(np.mean(val_f2)),
                'std_test_f2': float(np.std(val_f2)),
                'mean_test_f1_weighted': float(np.mean(val_f1w)),
                'mean_high_recall': float(np.mean(high_recalls)),
                'mean_train_f2': float(np.mean(train_f2)),
            }
            result['gap'] = result['mean_train_f2'] - result['mean_test_f2']
            self.cv_results_.append(result)
            

        self.cv_results_.sort(
            key=lambda row: (
                row['mean_test_f2'],
                row['mean_test_f1_weighted'],
                row['mean_high_recall'],
                -row['std_test_f2'],
            ),
            reverse=True,
        )
        best = self.cv_results_[0]
        self.best_params = dict(best['params'])
        self.best_cv_score = best['mean_test_f2']

        preprocessor = self.preprocessor_factory()
        processed = preprocessor.fit_transform(raw_train_df.copy())
        X_train, y_train, feature_cols = _get_processed_X_y(processed)
        X_fit, y_fit, fit_weights = self._fit_imbalance(X_train, y_train)
        estimator = self._build_estimator(self.best_params)
        self._fit_estimator(estimator, X_fit, y_fit, fit_weights)

        metadata = {
            'artifact_schema_version': 1,
            'model_class': self.model.__class__.__name__,
            'target_mapping': dict(RISK_CLASS_TO_INT),
            'config_name': self.config_name,
            'imbalance': self.imbalance,
            'class_weights': self.class_weights,
            'best_params': self.best_params,
            'best_cv_f2_macro': self.best_cv_score,
            'n_iter': self.n_iter,
            'cv_splits': self.cv,
            'cv_repeats': self.n_repeats,
            'random_state': self.random_state,
            'feature_names': list(feature_cols),
        }
        self.pipeline = RawModelPipeline(
            preprocessor=preprocessor,
            estimator=estimator,
            feature_cols=feature_cols,
            metadata=metadata,
        )
        self.model.best_params = self.best_params

        print(f"\n  Best params      : {self.best_params}")
        print(f"  Best CV F2-macro : {self.best_cv_score:.4f}")
        print(f"  CV F2 std        : {best['std_test_f2']:.4f}")
        print(f"  High recall      : {best['mean_high_recall']:.4f}")
        print(f"  Train/CV gap     : {best['gap']:.4f}")
        return self

    def evaluate(self, raw_df: pd.DataFrame, split_name: str = 'dataset') -> dict:
        if self.pipeline is None:
            raise RuntimeError("Run tune() before evaluate()")
        y = _raw_target(raw_df)
        y_pred = self.pipeline.predict(raw_df)
        recalls = recall_score(
            y, y_pred, labels=[0, 1, 2], average=None, zero_division=0
        )
        metrics = {
            'f2_macro': fbeta_score(
                y, y_pred, beta=2, average='macro', zero_division=0
            ),
            'f1_weighted': f1_score(
                y, y_pred, average='weighted', zero_division=0
            ),
            'recall_low': recalls[0],
            'recall_medium': recalls[1],
            'recall_high': recalls[2],
        }
        return metrics

    def evaluate_train(self, raw_train_df: pd.DataFrame) -> dict:
        self.train_metrics = self.evaluate(raw_train_df, split_name='Train')
        return self.train_metrics

    def evaluate_test(self, raw_test_df: pd.DataFrame) -> dict:
        self.test_metrics = self.evaluate(raw_test_df, split_name='Test')
        return self.test_metrics
