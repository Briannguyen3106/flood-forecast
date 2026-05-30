# src/core/ablation_trainer.py

import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import fbeta_score, f1_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.model.base_model import BaseModel


# ================================================================== #
#  SCORER
# ================================================================== #
f2_macro_scorer  = make_scorer(fbeta_score, beta=2, average='macro', zero_division=0)
f1_weighted_scorer = make_scorer(f1_score, average='weighted', zero_division=0)

SCORING = {
    'f2_macro'   : f2_macro_scorer,
    'f1_weighted': f1_weighted_scorer,
}


# ================================================================== #
#  RESULT DATACLASS
# ================================================================== #
@dataclass
class AblationResult:
    """Kết quả của 1 experiment"""
    experiment   : str           # Tên experiment
    config       : str           # Tên config (ví dụ: "+G1", "no_smote")
    model        : str           # Tên model
    cv_f2_mean   : float         # Mean F2-macro qua 15 folds
    cv_f2_std    : float         # Std F2-macro
    cv_f1w_mean  : float         # Mean F1-weighted
    train_f2_mean: float         # Mean train F2 (detect overfit)
    gap          : float         # train_f2 - cv_f2
    n_features   : int = 0


# ================================================================== #
#  ABLATION TRAINER
# ================================================================== #
class AblationTrainer:
    """
    Chạy ablation experiments với Repeated StratifiedKFold.

    Tại sao Repeated StratifiedKFold thay vì holdout val:
    - Dataset nhỏ (2,963 mẫu) → val set quá nhỏ
    - 5 folds × 3 repeats = 15 scores → mean ± std ổn định
    - Stratified → giữ tỷ lệ class trong mỗi fold

    Framework:
    - Outer: RepeatedStratifiedKFold(5×3) → đánh giá performance
    - Inner: RandomizedSearchCV(cv=5)     → tune hyperparameter
    """

    def __init__(self, n_splits: int = 5, n_repeats: int = 3,
                 random_state: int = 42):
        self.n_splits      = n_splits
        self.n_repeats     = n_repeats
        self.random_state  = random_state
        self.results       : list[AblationResult] = []

    # ------------------------------------------------------------------ #
    def _get_X_y(self, df: pd.DataFrame,
                 feature_cols: Optional[list] = None):
        """
        Tách X, y từ DataFrame.
        feature_cols: None → dùng tất cả features (trừ metadata)
                      list → chỉ dùng các cols chỉ định (ablation FE)
        """
        drop_cols = ['risk_class', 'risk_class_encoded',
                     'latitude', 'longitude']
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in drop_cols]

        X = df[feature_cols].values
        y = df['risk_class_encoded'].values
        return X, y, feature_cols

    # ------------------------------------------------------------------ #
    def _build_pipeline(self, model: BaseModel,
                        use_smote: bool = True) -> ImbPipeline:
        """Tạo ImbPipeline: [SMOTE →] Model"""
        model.build()
        steps = []
        if use_smote:
            steps.append(('smote', SMOTE(
                k_neighbors=5,
                random_state=self.random_state
            )))
        steps.append(('model', model.model))
        return ImbPipeline(steps)
    
    # ------------------------------------------------------------------ #
    def _run_cv(self, pipeline: ImbPipeline,
                X: np.ndarray, y: np.ndarray) -> dict:
        """
        Chạy Repeated StratifiedKFold CV.
        Trả về dict với mean/std của các metrics.
        """
        cv = RepeatedStratifiedKFold(
            n_splits   = self.n_splits,
            n_repeats  = self.n_repeats,
            random_state = self.random_state
        )

        cv_results = cross_validate(
            pipeline, X, y,
            cv                  = cv,
            scoring             = SCORING,
            return_train_score  = True,
            n_jobs              = -1
        )

        return {
            'cv_f2_mean'   : cv_results['test_f2_macro'].mean(),
            'cv_f2_std'    : cv_results['test_f2_macro'].std(),
            'cv_f1w_mean'  : cv_results['test_f1_weighted'].mean(),
            'train_f2_mean': cv_results['train_f2_macro'].mean(),
        }
    
    # ------------------------------------------------------------------ #
    def _log_result(self, result: AblationResult):
        print(f"      CV F2  : {result.cv_f2_mean:.4f} ± {result.cv_f2_std:.4f}")
        print(f"      F1w    : {result.cv_f1w_mean:.4f}")
        print(f"      Gap    : {result.gap:.4f} "
              f"{'⚠️ overfit' if result.gap > 0.1 else '✅'}")
    # ================================================================== #
    #  ABLATION 1 — IMBALANCE HANDLING
    # ================================================================== #
    def run_ablation_imbalance(self, train_tree: pd.DataFrame, train_linear: pd.DataFrame,
                                models: dict[str, BaseModel]) -> list[AblationResult]:
        """
        So sánh: Không SMOTE vs Có SMOTE
        Dùng tất cả features (không filter)

        models: {'RF': RandomForestClassifierModel(), 'Ridge': RidgeRegressionModel()}
        """
        print("\n" + "="*60)
        print("ABLATION 1 — Imbalance Handling")
        print("="*60)

        results = []
        for model_name, model in models.items():
            train_df = train_tree if model.pipeline_type == 'tree' else train_linear
            X, y, feature_cols = self._get_X_y(train_df)

            for use_smote in [False, True]:
                config = 'with_SMOTE' if use_smote else 'no_SMOTE'

                pipeline    = self._build_pipeline(model, use_smote=use_smote)
                metrics     = self._run_cv(pipeline, X, y)
                gap         = metrics['train_f2_mean'] - metrics['cv_f2_mean']

                result = AblationResult(
                    experiment  = 'imbalance',
                    config      = config,
                    model       = model_name,
                    cv_f2_mean  = round(metrics['cv_f2_mean'], 4),
                    cv_f2_std   = round(metrics['cv_f2_std'], 4),
                    cv_f1w_mean   = round(metrics['cv_f1w_mean'],   4),
                    train_f2_mean = round(metrics['train_f2_mean'], 4),
                    gap           = round(gap, 4),
                    n_features    = len(feature_cols)
                )
                results.append(result)
                self.results.append(result)
                self._log_result(result)
        return results

    # ================================================================== #
    #  ABLATION 2 — FEATURE ENGINEERING
    # ================================================================== #
    def run_ablation_features(self, raw_train_df: pd.DataFrame,
                               models: dict[str, BaseModel]) -> list[AblationResult]:
        """
        So sánh từng group feature engineering.
        Thêm từng group một để thấy đóng góp riêng lẻ.

        Configs:
          baseline : chỉ 5 numeric gốc + categorical
          +G1      : baseline + G1_infra_vuln
          +G2      : baseline + G2_rain_x_return
          +G3      : baseline + G3_soil_x_rainfall
          +G4      : baseline + G4_is_very_low_elev
          ALL      : baseline + tất cả G1-G4
        """
        from src.core.data_preprocessing import PipelineA, PipelineB
        print("\n" + "="*60)
        print("ABLATION 2 — Feature Engineering")
        print("="*60)

        fe_configs = {
            'baseline': [],
            '+G1'     : ['G1'],
            '+G2'     : ['G2'],
            '+G3'     : ['G3'],
            '+G4'     : ['G4'],
            'ALL'     : ['G1', 'G2', 'G3', 'G4'],   
        }

        results = []

        for config_name, fe_groups in fe_configs.items():
            print(f"\n  Config: {config_name} — groups={fe_groups}")
            
            for model_name, model in models.items():
                if model.pipeline_type == 'tree':
                    preprocessor    = PipelineA(fe_groups=fe_groups)
                else:
                    preprocessor    = PipelineB(fe_groups=fe_groups)
                
                train_processed = preprocessor.fit_transform(raw_train_df)
                X, y, feature_cols = self._get_X_y(train_processed)

                pipeline    = self._build_pipeline(model, use_smote=True)
                metrics     = self._run_cv(pipeline, X, y)
                gap         = metrics['train_f2_mean'] - metrics['cv_f2_mean']

                result = AblationResult(
                    experiment    = 'feature_engineering',
                    config        = config_name,
                    model         = model_name,
                    cv_f2_mean    = round(metrics['cv_f2_mean'],    4),
                    cv_f2_std     = round(metrics['cv_f2_std'],     4),
                    cv_f1w_mean   = round(metrics['cv_f1w_mean'],   4),
                    train_f2_mean = round(metrics['train_f2_mean'], 4),
                    gap           = round(gap, 4),
                    n_features    = len(feature_cols)
                )
                results.append(result)
                self.results.append(result)
                self._log_result(result)
        return results

    # ================================================================== #
    #  ABLATION 3 — PREPROCESSING PIPELINE
    # ================================================================== #
    def run_ablation_preprocessing(self,
                                raw_train_df: pd.DataFrame,
                                models      : dict[str, BaseModel]
                                ) -> list[AblationResult]:
        """
        Ablation 3 — Preprocessing Steps.
        Isolate đóng góp của từng bước preprocessing trong PipelineB.
    
        8 configs:
          Config 0: PipelineA           (baseline, không gì thêm)
          Config 1: + FE only           (feature engineering)
          Config 2: + Scale only        (RobustScaler)
          Config 3: + Skew only         (Yeo-Johnson)
          Config 4: + OHE only          (OneHotEncoder)
          Config 5: + Scale + Skew      (không OHE)
          Config 6: PipelineB full      (Scale+Skew+OHE, không FE)
          Config 7: PipelineB full + FE (tất cả)
    
        Models: Ridge (linear), SVM (distance), RF (tree — confirm không ảnh hưởng)
        raw_train_df: RAW train data (chưa preprocess) từ data/splits/train.csv
        """
        from src.core.data_preprocessing import PipelineA, PipelineB

        print("\n" + "="*60)
        print("ABLATION 3 — Preprocessing Steps")
        print(f"  CV: {self.n_splits} folds × {self.n_repeats} repeats")
        print("="*60)

        # Define 8 configs
        # (preprocessor_class, kwargs, config_name)
        configs = [
            (PipelineA, dict(fe_groups=[]),
             'Config0_baseline'),

            (PipelineA, dict(fe_groups=['G1','G2','G3','G4']),
             'Config1_+FE'),

            (PipelineB, dict(fe_groups=[], use_skew=False,
                         use_scale=True,  use_ohe=False),
             'Config2_+Scale'),

            (PipelineB, dict(fe_groups=[], use_skew=True,
                             use_scale=False, use_ohe=False),
             'Config3_+Skew'),

            (PipelineB, dict(fe_groups=[], use_skew=False,
                             use_scale=False, use_ohe=True),
             'Config4_+OHE'),

            (PipelineB, dict(fe_groups=[], use_skew=True,
                             use_scale=True,  use_ohe=False),
             'Config5_+Scale+Skew'),

            (PipelineB, dict(fe_groups=[], use_skew=True,
                             use_scale=True,  use_ohe=True),
             'Config6_PipelineB_full'),

            (PipelineB, dict(fe_groups=['G1','G2','G3','G4'],
                             use_skew=True, use_scale=True, use_ohe=True),
             'Config7_PipelineB_full+FE'),
        ]
    
        results = []
    
        for preprocessor_class, kwargs, config_name in configs:
            print(f"\n  Config: {config_name}")
    
            for model_name, model in models.items():
                # Config 0,1 dùng PipelineA → phù hợp cho tất cả model
                # Config 2-7 dùng PipelineB → tốt nhất cho linear/distance
                # RF vẫn chạy để confirm không bị ảnh hưởng
                preprocessor = preprocessor_class(**kwargs)
    
                try:
                    train_processed = preprocessor.fit_transform(raw_train_df)
                except Exception as e:
                    print(f"    [{model_name}] ERROR: {e}")
                    continue
                
                X, y, feature_cols = self._get_X_y(train_processed)
                print(f"    [{model_name}] {len(feature_cols)} features...")
    
                pipeline = self._build_pipeline(model, use_smote=False)
                metrics  = self._run_cv(pipeline, X, y)
                gap      = metrics['train_f2_mean'] - metrics['cv_f2_mean']
    
                result = AblationResult(
                    experiment    = 'preprocessing_steps',
                    config        = config_name,
                    model         = model_name,
                    cv_f2_mean    = round(metrics['cv_f2_mean'],    4),
                    cv_f2_std     = round(metrics['cv_f2_std'],     4),
                    cv_f1w_mean   = round(metrics['cv_f1w_mean'],   4),
                    train_f2_mean = round(metrics['train_f2_mean'], 4),
                    gap           = round(gap, 4),
                    n_features    = len(feature_cols)
                )
                results.append(result)
                self.results.append(result)
                self._log_result(result)
    
        return results
    
    # ================================================================== #
    #  SUMMARY
    # ================================================================== #
    def to_dataframe(self) -> pd.DataFrame:
        """Chuyển tất cả results thành DataFrame để phân tích"""
        if not self.results:
            print("Chưa có kết quả nào!")
            return pd.DataFrame()
 
        return pd.DataFrame([{
            'experiment'   : r.experiment,
            'config'       : r.config,
            'model'        : r.model,
            'cv_f2_mean'   : r.cv_f2_mean,
            'cv_f2_std'    : r.cv_f2_std,
            'cv_f1w_mean'  : r.cv_f1w_mean,
            'train_f2_mean': r.train_f2_mean,
            'gap'          : r.gap,
            'n_features'   : r.n_features,
        } for r in self.results])
    
    def print_summary(self):
        """In bảng tóm tắt đẹp theo từng experiment"""
        df = self.to_dataframe()
        if df.empty:
            return

        for exp in df['experiment'].unique():
            print(f"\n{'='*70}")
            print(f"EXPERIMENT: {exp.upper()}")
            print(f"{'='*70}")
            sub = df[df['experiment'] == exp].sort_values(
                ['model', 'cv_f2_mean'], ascending=[True, False]
            )
            print(sub[['config', 'model', 'cv_f2_mean',
                        'cv_f2_std', 'cv_f1w_mean', 'gap']].to_string(index=False))

    def save_results(self, save_path: str = 'results/ablation/ablation_results.csv'):
        """Lưu kết quả ra CSV"""
        from pathlib import Path
        df = self.to_dataframe()
        if df.empty:
            return
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path} ({len(df)} rows)")
        return df