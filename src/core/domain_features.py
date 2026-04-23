from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DomainFeatureEngineer:
    """Domain feature engineering groups extracted from notebook cells."""

    @staticmethod
    def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for col in cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    @staticmethod
    def _safe_soil_group_numeric(df: pd.DataFrame) -> pd.Series:
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "Unknown": np.nan}
        values = df["soil_group"]
        if values.dtype == "O":
            return values.map(mapping)
        return pd.to_numeric(values, errors="coerce")

    def transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
        train = train_df.copy()
        test = test_df.copy()

        numeric_for_fe = [
            "elevation_m",
            "drainage_density_km_per_km2",
            "storm_drain_proximity_m",
            "historical_rainfall_intensity_mm_hr",
            "return_period_years",
            "rain_x_return",
            "latitude",
            "longitude",
        ]
        train = self._safe_numeric(train, numeric_for_fe)
        test = self._safe_numeric(test, numeric_for_fe)

        train["G1_drainage_deficit"] = (1.0 / (train["drainage_density_km_per_km2"] + 1e-5)) * (
            1.0 / train["elevation_m"].clip(lower=0.1)
        )
        test["G1_drainage_deficit"] = (1.0 / (test["drainage_density_km_per_km2"] + 1e-5)) * (
            1.0 / test["elevation_m"].clip(lower=0.1)
        )

        train["G1_infra_vuln"] = train["storm_drain_proximity_m"] / (train["drainage_density_km_per_km2"] + 1e-5)
        test["G1_infra_vuln"] = test["storm_drain_proximity_m"] / (test["drainage_density_km_per_km2"] + 1e-5)

        train["G2_hydraulic_stress"] = train["historical_rainfall_intensity_mm_hr"] / (
            train["drainage_density_km_per_km2"] + 1e-5
        )
        test["G2_hydraulic_stress"] = test["historical_rainfall_intensity_mm_hr"] / (
            test["drainage_density_km_per_km2"] + 1e-5
        )

        train["G2_flood_accum"] = (
            train["historical_rainfall_intensity_mm_hr"] * train["return_period_years"]
        ) / (train["elevation_m"].clip(lower=0.5) ** 0.5)
        test["G2_flood_accum"] = (
            test["historical_rainfall_intensity_mm_hr"] * test["return_period_years"]
        ) / (test["elevation_m"].clip(lower=0.5) ** 0.5)

        soil_train = self._safe_soil_group_numeric(train)
        soil_test = self._safe_soil_group_numeric(test)
        train["G3_effective_runoff"] = (train["historical_rainfall_intensity_mm_hr"] * (soil_train + 1.0)) / 4.0
        test["G3_effective_runoff"] = (test["historical_rainfall_intensity_mm_hr"] * (soil_test + 1.0)) / 4.0

        train["G3_soil_elev_risk"] = (soil_train + 1.0) / (train["elevation_m"].clip(lower=0.5) + 1.0)
        test["G3_soil_elev_risk"] = (soil_test + 1.0) / (test["elevation_m"].clip(lower=0.5) + 1.0)

        train_elev_sorted = np.sort(train["elevation_m"].values)
        train_rain_sorted = np.sort(train["historical_rainfall_intensity_mm_hr"].values)
        train_elev_pct = np.argsort(np.argsort(train["elevation_m"].values)).astype(float) / max(1, len(train) - 1)
        train_rain_pct = np.argsort(np.argsort(train["historical_rainfall_intensity_mm_hr"].values)).astype(float) / max(
            1, len(train) - 1
        )

        test_elev_pct = np.interp(test["elevation_m"].values, train_elev_sorted, np.linspace(0, 1, len(train)))
        test_rain_pct = np.interp(
            test["historical_rainfall_intensity_mm_hr"].values,
            train_rain_sorted,
            np.linspace(0, 1, len(train)),
        )

        train["G4_elev_pct_rank"] = train_elev_pct
        train["G4_rain_pct_rank"] = train_rain_pct
        train["G4_compound_extreme"] = (1.0 - train_elev_pct) * train_rain_pct

        test["G4_elev_pct_rank"] = test_elev_pct
        test["G4_rain_pct_rank"] = test_rain_pct
        test["G4_compound_extreme"] = (1.0 - test_elev_pct) * test_rain_pct

        train["G5_is_tropical"] = (train["latitude"].abs() < 23.5).astype(int)
        test["G5_is_tropical"] = (test["latitude"].abs() < 23.5).astype(int)
        train["G5_abs_latitude"] = train["latitude"].abs()
        test["G5_abs_latitude"] = test["latitude"].abs()

        train["G6_log_elevation"] = np.log1p(train["elevation_m"].clip(lower=0))
        test["G6_log_elevation"] = np.log1p(test["elevation_m"].clip(lower=0))
        train["G6_log_rainfall"] = np.log1p(train["historical_rainfall_intensity_mm_hr"])
        test["G6_log_rainfall"] = np.log1p(test["historical_rainfall_intensity_mm_hr"])
        train["G6_log_rain_x_return"] = np.log1p(train["rain_x_return"])
        test["G6_log_rain_x_return"] = np.log1p(test["rain_x_return"])

        fe_groups = {
            "G1": ["G1_drainage_deficit", "G1_infra_vuln"],
            "G2": ["G2_hydraulic_stress", "G2_flood_accum"],
            "G3": ["G3_effective_runoff", "G3_soil_elev_risk"],
            "G4": ["G4_elev_pct_rank", "G4_rain_pct_rank", "G4_compound_extreme"],
            "G5": ["G5_is_tropical", "G5_abs_latitude"],
            "G6": ["G6_log_elevation", "G6_log_rainfall", "G6_log_rain_x_return"],
        }
        return train, test, fe_groups
