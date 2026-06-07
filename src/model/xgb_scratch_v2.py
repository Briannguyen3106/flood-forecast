import numpy as np

from src.model.base_model import BaseModel
from src.model.lgbm_scratch_v2 import GBDTMulticlass


class XGBScratchBaseModel(BaseModel):
    """XGBoost-from-scratch style multiclass classifier (wrapper).

    This project already contains a from-scratch GBDT implementation in
    `lgbm_scratch_v2.py` that supports:
      - softmax objective (multiclass)
      - L1 (reg_alpha)
      - min_split_gain
      - subsample / colsample_bytree
      - GOSS (use_goss)
      - EFB (use_efb)
      - early stopping (optional)

    We wrap it and expose it under XGBoost-like names and variants.

    Goal here is integration + variants for tuning.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        min_split_gain: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        max_bin: int = 255,
        use_goss: bool = False,
        top_rate: float = 0.2,
        other_rate: float = 0.1,
        use_efb: bool = False,
        max_conflict_rate: float = 0.0,
        early_stopping_rounds: int | None = None,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_split_gain = min_split_gain
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin

        self.use_goss = use_goss
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.use_efb = use_efb
        self.max_conflict_rate = max_conflict_rate
        self.early_stopping_rounds = early_stopping_rounds

        self.random_state = random_state
        self.pipeline_type = "tree"

        self.model = None

    def get_param_distributions(self) -> dict:
        return {
            "n_estimators": [80, 120, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [10, 20, 30, 50],
            "reg_lambda": [0.1, 1.0, 5.0, 10.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "min_split_gain": [0.0, 0.01, 0.1],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.7, 0.8, 0.9, 1.0],
            "max_bin": [128, 255],
            "top_rate": [0.1, 0.2, 0.3, 0.4],
            "other_rate": [0.05, 0.1, 0.15],
            "use_goss": [self.use_goss],
            "use_efb": [self.use_efb],
        }

    def build(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

        self.model = GBDTMulticlass(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            min_split_gain=self.min_split_gain,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            max_bin=self.max_bin,
            use_goss=self.use_goss,
            top_rate=self.top_rate,
            other_rate=self.other_rate,
            use_efb=self.use_efb,
            max_conflict_rate=self.max_conflict_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            objective="softmax",
            random_state=self.random_state,
        )
        return self

    def fit(self, X, y):
        # Trainer will pass only X,y for our current design
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep: bool = True):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "min_split_gain": self.min_split_gain,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "max_bin": self.max_bin,
            "use_goss": self.use_goss,
            "top_rate": self.top_rate,
            "other_rate": self.other_rate,
            "use_efb": self.use_efb,
            "max_conflict_rate": self.max_conflict_rate,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class XGBScratchV2Model(XGBScratchBaseModel):
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, use_goss=False, use_efb=False, **kwargs)


class XGBScratchGOSSV2Model(XGBScratchBaseModel):
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, use_goss=True, use_efb=False, **kwargs)


class XGBScratchEFBV2Model(XGBScratchBaseModel):
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, use_goss=False, use_efb=True, **kwargs)


class XGBScratchGOSS_EFBV2Model(XGBScratchBaseModel):
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, use_goss=True, use_efb=True, **kwargs)


# Extra variants (XGB-like / LightGBM-like names)
# - histgb: thiên về histogram-based tree building
#   (ở scratch implementation này đã là histogram tree -> chỉ cần alias + tweak)
# - catboost-like: bỏ qua vì CatBoost dùng ordered boosting + target statistics,
#   trong khi pipeline hiện tại tập trung vào histogram GBDT; vẫn có thể tạo alias
#   để bạn thử thêm search space.

class XGBHistGOSSV2Model(XGBScratchGOSSV2Model):
    """Alias: 'histgb' + GOSS (histogram tree is default in implementation)."""


class XGBHistEFBV2Model(XGBScratchEFBV2Model):
    """Alias: 'histgb' + EFB."""


class XGBHistGOSS_EFBV2Model(XGBScratchGOSS_EFBV2Model):
    """Alias: 'histgb' + GOSS + EFB."""


# For convenience: keep short aliases
XGBScratch_v2 = XGBScratchV2Model
XGBScratch_goss_v2 = XGBScratchGOSSV2Model
XGBScratch_efb_v2 = XGBScratchEFBV2Model
XGBScratch_goss_efb_v2 = XGBScratchGOSS_EFBV2Model
XGBHistGoss_v2 = XGBHistGOSSV2Model
XGBHistEfb_v2 = XGBHistEFBV2Model
XGBHistGossEfb_v2 = XGBHistGOSS_EFBV2Model

