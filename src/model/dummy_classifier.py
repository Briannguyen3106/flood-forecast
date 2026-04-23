from sklearn.dummy import DummyClassifier

from src.model.base_model import BaseModel


class DummyClassifierModel(BaseModel):
    """Baseline model: always predicts the most frequent class."""

    def __init__(self, strategy: str = "most_frequent", random_state: int = 42):
        super().__init__()
        self.strategy = strategy
        self.random_state = random_state
        self.pipeline_type = "tree"

    def get_param_distributions(self) -> dict:
        return {"strategy": ["most_frequent", "prior", "stratified"]}

    def build(self, **params):
        cfg = {
            "strategy": self.strategy,
            "random_state": self.random_state,
        }
        cfg.update(params)
        self.model = DummyClassifier(**cfg)
        return self

    def get_params(self, deep: bool = True):
        return {
            "strategy": self.strategy,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
