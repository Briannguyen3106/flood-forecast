import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from src.model.base_model import BaseModel

# ================================================================== #
#  SCORER
# ================================================================== #
f2_macro_scorer = make_scorer(fbeta_score, beta=2, average='macro')

# ================================================================== #
# Base Trainer
# ================================================================== #
class Trainer:
    def __init__(self, model: BaseModel,
                 n_iter: int=50, random_state: int=42):
        assert isinstance(model, BaseModel), "model phải kế thừa BaseModel"
        self.model        = model
        self.n_iter       = n_iter
        self.random_state = random_state
        self.val_metrics  = {}

    def _get_X_y(self, df):
        drop_cols = ['SUSCEP', 'SUSCEP_encoded', 'X', 'Y']
        feature_cols = [c for c in df.colums if c not in drop_cols]
        X = df[feature_cols].values
        y = df['SUSCEP_encoded'].values
        return X, y
    
    def tune(self, train_df):
        X_train, y_train = self._get_X_y(train_df)
        
        search = RandomizedSearchCV(
            estimator           = self.model,
            param_distributions = self.param_distributions,
            n_iter              = self.n_iter,
            scoring             = f2_macro_scorer,
            cv                  = 5,
            random_state        = self.random_state,
            n_jobs              = -1,
            verbose             = 1
        )
        search.fit(X_train, y_train)

        self.model.best_params = search.best_params_
        self.model.model  = search.best_estimator_

        print(f"\nBest params   : {self.model.best_params}")
        print(f"Best CV F2-macro: {search.best_score_:.4f}")
        return self

    
    def evaluate_val(self, val_df: pd.DataFrame) -> dict:
        """Đánh giá trên val set — dùng để so sánh các model"""
        X_val, y_val = self._get_X_y(val_df)
        y_pred = self.model.predict(X_val)

        self.val_metrics = {
            'f2_macro'   : fbeta_score(y_val, y_pred, beta=2, average='macro'),
            'f1_weighted': f1_score(y_val, y_pred, average='weighted'),
        }
        print(f"\n[Val] F2-macro    : {self.val_metrics['f2_macro']:.4f}")
        print(f"[Val] F1-weighted : {self.val_metrics['f1_weighted']:.4f}")
        return self.val_metrics
    
    def retrain(self, train_val_df: pd.DataFrame):
        """Phase 2: retrain trên train+val với best_params"""
        assert self.model.best_params is not None, "Chạy tune() trước"
        X, y = self._get_X_y(train_val_df)
        self.model.build(**self.model.best_params).fit(X, y)
        print(f"Retrained on {len(train_val_df):,} rows (train+val)")
        return self
    
    def evaluate_test(self, test_df: pd.DataFrame) -> dict:
        """Đánh giá cuối cùng trên test — chỉ gọi 1 lần"""
        X_test, y_test = self._get_X_y(test_df)
        y_pred = self.model.predict(X_test)

        test_metrics = {
            'f2_macro'   : fbeta_score(y_test, y_pred, beta=2, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        }
        print(f"\n[Test] F2-macro    : {test_metrics['f2_macro']:.4f}")
        print(f"[Test] F1-weighted : {test_metrics['f1_weighted']:.4f}")
        return test_metrics
    
    # ------------------------------------------------------------------ #
    @staticmethod
    def _f2_macro_scorer():
        from sklearn.metrics import make_scorer
        return make_scorer(fbeta_score, beta=2, average='macro')
    


