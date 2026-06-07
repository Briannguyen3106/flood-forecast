from src.model.dummy_classifier import DummyClassifierModel
from src.model.hist_gradient_boosting_classifier import HistGradientBoostingClassifierModel
from src.model.lgbm_classifier import LGBMClassifierModel
from src.model.random_forest_classifier import RandomForestClassifierModel
from src.model.xgb_classifier import XGBClassifierModel
from src.model.xgbrf_classifier import XGBRFClassifierModel
from src.model.svm_classifier import SVMClassifierModel

from src.model.random_forest_scratch import RandomForestScratchModel
from src.model.xgb_scratch_v2 import (
    XGBScratchV2Model,
    XGBScratchGOSSV2Model,
    XGBScratchEFBV2Model,
    XGBScratchGOSS_EFBV2Model,
)


MODEL_REGISTRY = {
    "dummy": DummyClassifierModel,
    "rf": RandomForestClassifierModel,
    "xgb": XGBClassifierModel,
    "xgbrf": XGBRFClassifierModel,
    "lgbm": LGBMClassifierModel,
    "hgb": HistGradientBoostingClassifierModel,
    "svm": SVMClassifierModel,

    # --- scratch variants ---
    "rf_scratch": RandomForestScratchModel,

    "xgb_scratch": XGBScratchV2Model,
    "xgb_scratch_goss": XGBScratchGOSSV2Model,
    "xgb_scratch_efb": XGBScratchEFBV2Model,
    "xgb_scratch_goss_efb": XGBScratchGOSS_EFBV2Model,

    # extra aliases
    "histgb_goss": XGBHistGOSSV2Model,
    "histgb_efb": XGBHistEFBV2Model,
    "histgb_goss_efb": XGBHistGOSS_EFBV2Model,
}



def create_model(model_key: str, **kwargs):
    key = model_key.strip().lower()
    if key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model key: {model_key}. Valid keys: {valid}")
    return MODEL_REGISTRY[key](**kwargs)
