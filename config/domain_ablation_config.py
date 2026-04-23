from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DomainAblationConfig:
    random_state: int = 42
    cv_splits_eval: int = 10
    cv_splits_tune: int = 5
    target_col: str = "risk_class_encoded"

    base_numeric_features: List[str] = field(
        default_factory=lambda: [
            "elevation_m",
            "drainage_density_km_per_km2",
            "storm_drain_proximity_m",
            "historical_rainfall_intensity_mm_hr",
            "return_period_years",
            "is_very_low_elev",
            "rain_x_return",
            "latitude",
            "longitude",
        ]
    )

    base_categorical_features: List[str] = field(
        default_factory=lambda: [
            "land_use",
            "soil_group",
            "storm_drain_type",
            "rainfall_source",
            "dem_source",
        ]
    )

    xgb_baseline_params: Dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "n_jobs": -1,
        }
    )

    rf_tuning_space: Dict = field(
        default_factory=lambda: {
            "n_estimators": (100, 500),
            "max_depth": [None, 8, 15, 25, 35],
            "min_samples_split": (2, 15),
            "min_samples_leaf": (1, 5),
            "max_features": ["sqrt", "log2", 0.5, 0.7],
            "class_weight": ["balanced", "balanced_subsample", None],
        }
    )

    @property
    def baseline_features(self) -> List[str]:
        return self.base_numeric_features + self.base_categorical_features
