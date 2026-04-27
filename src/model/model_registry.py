from src.model.dummy_classifier import DummyClassifierModel
from src.model.hist_gradient_boosting_classifier import HistGradientBoostingClassifierModel
from src.model.lgbm_classifier import LGBMClassifierModel
from src.model.random_forest_classifier import RandomForestClassifierModel
from src.model.xgb_classifier import XGBClassifierModel
from src.model.xgbrf_classifier import XGBRFClassifierModel
from src.model.svm_classifier import SVMClassifierModel


MODEL_REGISTRY = {
    "dummy": DummyClassifierModel,
    "rf": RandomForestClassifierModel,
    "xgb": XGBClassifierModel,
    "xgbrf": XGBRFClassifierModel,
    "lgbm": LGBMClassifierModel,
    "hgb": HistGradientBoostingClassifierModel,
    "svm": SVMClassifierModel,
}


def create_model(model_key: str, **kwargs):
    key = model_key.strip().lower()
    if key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model key: {model_key}. Valid keys: {valid}")
    return MODEL_REGISTRY[key](**kwargs)
