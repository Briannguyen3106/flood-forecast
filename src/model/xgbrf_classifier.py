from xgboost import XGBRFClassifier

from src.model.base_model import BaseModel


class XGBRFClassifierModel(BaseModel):
    """XGBRF classifier adapted from Dung_experiments notebook."""

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state
        self.pipeline_type = "tree"

    def get_param_distributions(self) -> dict:
        return {
            "n_estimators": [100, 200, 300, 500, 600],
            "max_depth": [4, 5, 6, 8, 10, 12],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bynode": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 2, 3, 5, 7, 10],
            "reg_alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
            "reg_lambda": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
        }

    def build(self, **params):
        cfg = {
            "n_estimators": 300,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "colsample_bynode": 0.8,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        cfg.update(params)
        self.model = XGBRFClassifier(**cfg)
        return self

    def get_params(self, deep: bool = True):
        return {"random_state": self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
