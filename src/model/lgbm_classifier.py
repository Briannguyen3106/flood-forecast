from lightgbm import LGBMClassifier

from src.model.base_model import BaseModel


class LGBMClassifierModel(BaseModel):
    """LightGBM classifier adapted from Dung_experiments notebook."""

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state
        self.pipeline_type = "tree"

    def get_param_distributions(self) -> dict:
        return {
            "n_estimators": [100, 200, 300, 500, 700],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
            "num_leaves": [20, 31, 63, 127, 200],
            "max_depth": [3, 4, 5, 6, 8, 10, 12],
            "min_child_samples": [10, 20, 30, 50, 70, 100],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
            "reg_lambda": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
        }

    def build(self, **params):
        cfg = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        cfg.update(params)
        self.model = LGBMClassifier(**cfg)
        return self

    def get_params(self, deep: bool = True):
        return {"random_state": self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
