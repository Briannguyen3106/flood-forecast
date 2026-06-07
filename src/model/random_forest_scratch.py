import numpy as np

from src.model.base_model import BaseModel
from src.model.decision_tree_classifier import DecisionTreeClassifierModel


class RandomForestScratchModel(BaseModel):
    """RandomForestClassifier implemented from scratch.

    - Bootstrap sampling to create each tree dataset
    - Feature subsampling (max_features)
    - Each tree is a DecisionTreeClassifierModel (also from scratch)

    Notes:
    - Works with the Trainer which uses ImbPipeline(SMOTE -> model).
      SMOTE happens outside; this class only needs to implement sklearn-like API.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth=None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        max_features=None,
        bootstrap: bool = True,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.pipeline_type = "tree"
        self.trees_ = []
        self.feature_subsets_ = []

    def get_param_distributions(self) -> dict:
        return {
            "n_estimators": [20, 50, 80, 120],
            "max_depth": [None, 8, 15, 25, 35],
            "min_samples_split": [2, 4, 6, 8, 10, 15],
            "min_samples_leaf": [1, 2, 3, 4, 5, 10],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
            "bootstrap": [True, False],
        }

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features in (None, 1.0, "all"):
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, (float, int)):
            if isinstance(self.max_features, float):
                return max(1, int(self.max_features * n_features))
            return max(1, int(self.max_features))
        # fallback
        return n_features

    def build(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self.trees_ = []
        self.feature_subsets_ = []
        return self

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.feature_subsets_ = []

        m = self._resolve_max_features(n_features)

        for i in range(self.n_estimators):
            # bootstrap sampling rows
            if self.bootstrap:
                idx = rng.integers(0, n_samples, size=n_samples)
            else:
                idx = rng.choice(n_samples, size=n_samples, replace=False)

            # feature subsampling cols
            feat_idx = np.sort(rng.choice(n_features, size=m, replace=False))

            tree = DecisionTreeClassifierModel(
                random_state=int(rng.integers(0, 1_000_000_000)),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
            )
            tree.fit(X[idx][:, feat_idx], y[idx])

            self.trees_.append(tree)
            self.feature_subsets_.append(feat_idx)

        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for tree, feat_idx in zip(self.trees_, self.feature_subsets_):
            preds.append(tree.predict(X[:, feat_idx]))
        preds = np.stack(preds, axis=0)  # (n_estimators, n_samples)

        # majority vote
        final = []
        for j in range(preds.shape[1]):
            vals, counts = np.unique(preds[:, j], return_counts=True)
            final.append(vals[np.argmax(counts)])
        return np.asarray(final)

    def get_params(self, deep: bool = True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

