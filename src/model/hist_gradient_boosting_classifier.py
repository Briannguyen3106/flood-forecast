from sklearn.ensemble import HistGradientBoostingClassifier

from src.model.base_model import BaseModel


class HistGradientBoostingClassifierModel(BaseModel):
    """HistGradientBoosting classifier adapted from Dung_experiments notebook."""

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state
        self.pipeline_type = "tree"

    def get_param_distributions(self) -> dict:
        return {
            "max_iter": [100, 200, 300, 400, 500, 600],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
            "max_leaf_nodes": [15, 31, 63, 127, 255],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "min_samples_leaf": [5, 10, 20, 30, 40, 50],
            "l2_regularization": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
            "class_weight": ["balanced"],
        }

    def build(self, **params):
        cfg = {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 63,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "class_weight": "balanced",
            "random_state": self.random_state,
        }
        cfg.update(params)
        self.model = HistGradientBoostingClassifier(**cfg)
        return self

    def get_params(self, deep: bool = True):
        return {"random_state": self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
