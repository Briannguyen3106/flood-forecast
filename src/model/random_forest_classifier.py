from sklearn.ensemble import RandomForestClassifier

from src.model.base_model import BaseModel


class RandomForestClassifierModel(BaseModel):
    """RandomForest configuration adapted from Dung_experiments notebook."""

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state
        self.pipeline_type = "tree"

    def get_param_distributions(self) -> dict:
        return {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 8, 15, 25, 35],
            "min_samples_split": [2, 4, 6, 8, 10, 12, 15],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_features": ["sqrt", "log2", 0.5, 0.7],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }

    def build(self, **params):
        cfg = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        cfg.update(params)
        self.model = RandomForestClassifier(**cfg)
        return self

    def get_params(self, deep: bool = True):
        return {"random_state": self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
