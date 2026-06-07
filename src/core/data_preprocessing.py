# src/core/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from scipy.stats import yeojohnson

# ================================================================== #
#  TARGET ENCODING
# ================================================================== #
def assign_class(labels_str):
    if pd.isna(labels_str) or labels_str.strip() == '':
        return 'Low'
    labels = [l.strip() for l in labels_str.split('|')]
    if any(l in ['ponding_hotspot', 'extreme_rain_history'] for l in labels):
        return 'High'
    if any(l in ['low_lying', 'sparse_drainage'] for l in labels):
        return 'Medium'
    return 'Low'


RISK_CLASS_TO_INT = {'Low': 0, 'Medium': 1, 'High': 2}
RISK_INT_TO_CLASS = {value: key for key, value in RISK_CLASS_TO_INT.items()}
RISK_CLASS_NAMES = [RISK_INT_TO_CLASS[i] for i in range(len(RISK_INT_TO_CLASS))]


# ================================================================== #
#  BASE PREPROCESSOR
# ================================================================== #
class BasePreprocessor:
    def __init__(self, clip_proximity_uper=0.95, fe_groups: list = None):
        self.clip_proximity_uper = clip_proximity_uper
        self.fe_groups = fe_groups if fe_groups is not None else ['G1', 'G2', 'G3', 'G4']

        self.numerical_medians     = {}
        self.proximity_clip_uppers = None
        self.elev_low_threshold    = None

        self.numeric_cols = [
            'elevation_m',
            'drainage_density_km_per_km2',
            'storm_drain_proximity_m',
            'historical_rainfall_intensity_mm_hr',
            'return_period_years'
        ]
        self.categorical_cols = [
            'land_use', 'soil_group',
            'storm_drain_type', 'rainfall_source', 'dem_source'
        ]
        self.drop_cols = [
            'segment_id', 'city_name', 'admin_ward',
            'catchment_id', 'risk_labels'
        ]
        self.target_col = 'risk_class'

    # ------------------------------------------------------------------ #
    # TARGET
    # ------------------------------------------------------------------ #
    def _apply_target(self, df):
        df = df.copy()
        df['risk_class'] = df['risk_labels'].apply(assign_class)
        return df

    # ------------------------------------------------------------------ #
    # SENTINEL VALUE
    # ------------------------------------------------------------------ #
    def _apply_sentinel(self, df):
        df = df.copy()
        df['elevation_m'] = df['elevation_m'].replace(-3.0, np.nan)
        return df

    # ------------------------------------------------------------------ #
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------ #
    def _fit_feature_engineering(self, df):
        if 'G4' in self.fe_groups:
            self.elev_low_threshold = df['elevation_m'].quantile(0.10)

    def _apply_feature_engineering(self, df):
        df = df.copy()

        if 'G1' in self.fe_groups:
            drain_safe = df['drainage_density_km_per_km2'].fillna(
                df['drainage_density_km_per_km2'].median()
            ) + 1e-5
            prox_temp = df['storm_drain_proximity_m'].fillna(
                df['storm_drain_proximity_m'].median()
            )
            df['G1_infra_vuln'] = prox_temp / drain_safe

        if 'G2' in self.fe_groups:
            df['G2_rain_x_return'] = (
                df['historical_rainfall_intensity_mm_hr']
                * df['return_period_years']
            )

        if 'G3' in self.fe_groups:
            soil_map = {'Unknown': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
            soil_encoded_temp = df['soil_group'].fillna('Unknown').map(soil_map)
            df['G3_soil_x_rainfall'] = (
                soil_encoded_temp * df['historical_rainfall_intensity_mm_hr']
            )

        if 'G4' in self.fe_groups:
            threshold = self.elev_low_threshold if self.elev_low_threshold is not None else 5.0
            df['G4_is_very_low_elev'] = (
                df['elevation_m'].fillna(999) < threshold
            ).astype(int)

        return df

    # ------------------------------------------------------------------ #
    # IMPUTE
    # ------------------------------------------------------------------ #
    def _fit_impute(self, df):
        for col in self.numeric_cols:
            self.numerical_medians[col] = df[col].median()

    def _apply_impute(self, df):
        df = df.copy()
        for col in self.numeric_cols:
            df[col] = df[col].fillna(self.numerical_medians[col])
        for col in self.categorical_cols:
            df[col] = df[col].fillna('Unknown')
        return df

    # ------------------------------------------------------------------ #
    # CLIP
    # ------------------------------------------------------------------ #
    def _fit_clip(self, df):
        self.proximity_clip_uppers = df['storm_drain_proximity_m'].quantile(
            self.clip_proximity_uper
        )

    def _apply_clip(self, df):
        df = df.copy()
        df['storm_drain_proximity_m'] = np.clip(
            df['storm_drain_proximity_m'],
            df['storm_drain_proximity_m'].min(),
            self.proximity_clip_uppers
        )
        return df

    # ------------------------------------------------------------------ #
    # LABEL ENCODER
    # ------------------------------------------------------------------ #
    def _fit_label_encoder(self, df):
        unknown = set(df[self.target_col].dropna().unique()) - set(RISK_CLASS_TO_INT)
        if unknown:
            raise ValueError(f"Unknown risk classes: {sorted(unknown)}")

    def _apply_label_encoder(self, df):
        encoded = df[self.target_col].map(RISK_CLASS_TO_INT)
        if encoded.isna().any():
            unknown = sorted(df.loc[encoded.isna(), self.target_col].unique())
            raise ValueError(f"Unknown risk classes: {unknown}")
        return encoded.astype(int).to_numpy()

    # ------------------------------------------------------------------ #
    # METADATA
    # ------------------------------------------------------------------ #
    def _attach_metadata(self, df_processed, df_original):
        df_processed = df_processed.copy()
        df_processed['risk_class']         = df_original[self.target_col].values
        df_processed['risk_class_encoded'] = self._apply_label_encoder(df_original)
        df_processed['latitude']           = df_original['latitude'].values
        df_processed['longitude']          = df_original['longitude'].values
        return df_processed

    # ------------------------------------------------------------------ #
    # COMMON FLOW
    # ------------------------------------------------------------------ #
    def _common_fit(self, df):
        df = self._apply_target(df)
        df = self._apply_sentinel(df)
        self._fit_feature_engineering(df)
        df = self._apply_feature_engineering(df)
        self._fit_impute(df)
        df = self._apply_impute(df)
        self._fit_clip(df)
        df = self._apply_clip(df)
        self._fit_label_encoder(df)
        return df

    def _common_transform(self, df):
        df = self._apply_target(df)
        df = self._apply_sentinel(df)
        df = self._apply_feature_engineering(df)
        df = self._apply_impute(df)
        df = self._apply_clip(df)
        return df

    def fit(self, df):
        raise NotImplementedError
    def transform(self, df):
        raise NotImplementedError
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def _get_engineered_cols(self) -> list:
        mapping = {
            'G1': 'G1_infra_vuln',
            'G2': 'G2_rain_x_return',
            'G3': 'G3_soil_x_rainfall',
            'G4': 'G4_is_very_low_elev',
        }
        return [mapping[g] for g in ['G1', 'G2', 'G3', 'G4']
                if g in self.fe_groups]


# ================================================================== #
#  PIPELINE A — TREE-BASED
# ================================================================== #
class PipelineA(BasePreprocessor):
    def __init__(self, fe_groups: list = None):
        super().__init__(fe_groups=fe_groups)
        self.ordinal_encoder  = None
        self.feature_cols_out = None

    def _fit_encoder(self, df):
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.ordinal_encoder.fit(df[self.categorical_cols])

    def _apply_encoder(self, df):
        df = df.copy()
        df[self.categorical_cols] = self.ordinal_encoder.transform(
            df[self.categorical_cols]
        )
        return df

    def fit(self, df: pd.DataFrame):
        print(f"[PipelineA] fitting on {len(df):,} rows, fe_groups={self.fe_groups}...")
        df_common = self._common_fit(df)
        self._fit_encoder(df_common)
        engineered_cols       = self._get_engineered_cols()
        self.feature_cols_out = self.numeric_cols + engineered_cols + self.categorical_cols
        print(f"[PipelineA] done — {len(self.feature_cols_out)} features")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_common  = self._common_transform(df)
        df_encoded = self._apply_encoder(df_common)
        df_out     = df_encoded[self.feature_cols_out].copy()
        df_out     = self._attach_metadata(df_out, df_common)
        return df_out


# ================================================================== #
#  PIPELINE B — LINEAR / DISTANCE-BASED
# ================================================================== #
class PipelineB(BasePreprocessor):

    ORDINAL_COLS  = ['soil_group']
    ORDINAL_ORDER = [['Unknown', 'A', 'B', 'C', 'D']]
    OHE_COLS      = ['land_use', 'storm_drain_type', 'rainfall_source', 'dem_source']

    BASE_SKEW_COLS = [
        'elevation_m',
        'storm_drain_proximity_m',
        'historical_rainfall_intensity_mm_hr',
        'return_period_years',
    ]
    ENGINEERED_SKEW_COLS = [
        'G1_infra_vuln',
        'G2_rain_x_return',
        'G3_soil_x_rainfall',
    ]

    def __init__(self,
                 skew_threshold : float = 0.5,
                 fe_groups      : list  = None,
                 use_skew       : bool  = True,
                 use_scale      : bool  = True,
                 use_ohe        : bool  = True):
        super().__init__(fe_groups=fe_groups)
        self.skew_threshold = skew_threshold
        self.use_skew       = use_skew
        self.use_scale      = use_scale
        self.use_ohe        = use_ohe

        self.skew_transforms   = {}
        self.ordinal_encoder   = None
        self.ohe_encoder       = None
        self.scaler            = None
        self.ohe_feature_names = []
        self.scale_cols        = None
        self.feature_cols_out  = None

    # ── Skew ────────────────────────────────────────────────────────
    def _fit_skew(self, df):
        self.skew_transforms = {}
        if not self.use_skew:
            return

        skew_cols = self.BASE_SKEW_COLS + [
            c for c in self.ENGINEERED_SKEW_COLS if c in df.columns
        ]
        print("  Yeo-Johnson transforms:")
        for col in skew_cols:
            if col not in df.columns:
                continue
            skew = df[col].skew()
            if abs(skew) < self.skew_threshold:
                self.skew_transforms[col] = {'method': 'none'}
            else:
                _, lmbda = yeojohnson(df[col].dropna())
                self.skew_transforms[col] = {
                    'method'     : 'yeojohnson',
                    'lambda'     : lmbda,
                    'skew_before': round(skew, 3)
                }
                print(f"    '{col}': skew={skew:.3f} → λ={lmbda:.4f}")

    def _apply_skew(self, df):
        df = df.copy()
        if not self.use_skew:
            return df
        for col, info in self.skew_transforms.items():
            if info['method'] == 'yeojohnson' and col in df.columns:
                df[col] = yeojohnson(df[col], lmbda=info['lambda'])
        return df

    # ── Encoders ─────────────────────────────────────────────────────
    def _fit_encoders(self, df):
        """
        Fit encoders từ train set.
        use_ohe=True  → OrdinalEncoder(soil) + OneHotEncoder(others)
        use_ohe=False → OrdinalEncoder(soil) + OrdinalEncoder(others)
        Cả 2 trường hợp đều fit trên train → không data leak khi transform test
        """
        # OrdinalEncoder cho soil_group — luôn fit
        self.ordinal_encoder = OrdinalEncoder(
            categories=self.ORDINAL_ORDER,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.ordinal_encoder.fit(df[self.ORDINAL_COLS])

        if self.use_ohe:
            # OneHotEncoder cho OHE_COLS
            self.ohe_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore'
            )
            self.ohe_encoder.fit(df[self.OHE_COLS])
            self.ohe_feature_names = self.ohe_encoder.get_feature_names_out(
                self.OHE_COLS
            ).tolist()
        else:
            # OrdinalEncoder cho OHE_COLS — fit trên train ← tránh leak
            self.ohe_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            self.ohe_encoder.fit(df[self.OHE_COLS])  # ← fit trên train
            self.ohe_feature_names = []

    def _apply_encoders(self, df):
        """
        Apply encoders đã fit từ train.
        Không fit lại → không data leak với test set.
        """
        df = df.copy()
        # Luôn transform soil_group bằng ordinal_encoder
        df[self.ORDINAL_COLS] = self.ordinal_encoder.transform(df[self.ORDINAL_COLS])

        if self.use_ohe and isinstance(self.ohe_encoder, OneHotEncoder):
            # OneHotEncoder → expand thành nhiều cols
            ohe_array = self.ohe_encoder.transform(df[self.OHE_COLS])
            ohe_df    = pd.DataFrame(
                ohe_array, columns=self.ohe_feature_names, index=df.index
            )
            df = pd.concat([df.drop(columns=self.OHE_COLS), ohe_df], axis=1)
        else:
            # OrdinalEncoder → transform (không fit lại) ← tránh leak
            df[self.OHE_COLS] = self.ohe_encoder.transform(df[self.OHE_COLS])
        return df

    # ── Scaler ───────────────────────────────────────────────────────
    def _fit_scaler(self, df, scale_cols):
        if not self.use_scale:
            return
        self.scaler = RobustScaler()
        self.scaler.fit(df[scale_cols])

    def _apply_scaler(self, df, scale_cols):
        if not self.use_scale or self.scaler is None:
            return df
        df = df.copy()
        df[scale_cols] = self.scaler.transform(df[scale_cols])
        return df

    # ── Fit ──────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        print(f"\n[PipelineB] fitting on {len(df):,} rows...")
        print(f"  use_skew={self.use_skew}, use_scale={self.use_scale}, "
              f"use_ohe={self.use_ohe}, fe_groups={self.fe_groups}")

        df_common   = self._common_fit(df)
        self._fit_skew(df_common)
        df_unskewed = self._apply_skew(df_common)

        self._fit_encoders(df_unskewed)
        df_encoded = self._apply_encoders(df_unskewed)

        # Dynamic scale_cols:
        # use_ohe=True  → ORDINAL_COLS + ohe_feature_names (binary cols)
        # use_ohe=False → ORDINAL_COLS + OHE_COLS (ordinal encoded, cần scale)
        engineered_cols = self._get_engineered_cols()
        if self.use_ohe:
            categorical_encoded_cols = self.ORDINAL_COLS + self.ohe_feature_names
        else:
            categorical_encoded_cols = self.ORDINAL_COLS + self.OHE_COLS

        self.scale_cols = (
            self.numeric_cols
            + engineered_cols
            + categorical_encoded_cols
        )
        self._fit_scaler(df_encoded, self.scale_cols)
        self.feature_cols_out = self.scale_cols

        print(f"[PipelineB] done — {len(self.feature_cols_out)} features")
        return self

    # ── Transform ────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_common   = self._common_transform(df)
        df_unskewed = self._apply_skew(df_common)
        df_encoded  = self._apply_encoders(df_unskewed)
        df_scaled   = self._apply_scaler(df_encoded, self.scale_cols)
        df_out      = df_scaled[self.feature_cols_out].copy()
        df_out      = self._attach_metadata(df_out, df_common)
        return df_out
