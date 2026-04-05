# src/core/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from scipy.stats import yeojohnson

#=========================================================== #
# Target Encoding
#============================================================ #
def assign_class(labels_str):
    if pd.isna(labels_str) or labels_str.strip() =='':
        return 'Low'
    labels = [l.strip() for l in labels_str.split('|')]
    if any(l in ['ponding_hotspot', 'extreme_rain_history'] for l in labels):
        return 'High'
    if any(l in ['low_lying', 'sparse_drainage'] for l in labels):
        return 'Medium'
    return 'Low'

# ================================================================== #
#  BASE PREPROCESSOR
# ================================================================== #
class BasePreprocessor:
    def __init__(self, clip_proximity_uper = 0.95):
        self.clip_proximity_uper = clip_proximity_uper

        self.numerical_medians = {}
        self.proximity_clip_uppers = None
        self.label_encoders = LabelEncoder()

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
    def _apply_feature_engineering(self, df):
        df = df.copy()
        df['is_very_low_elev'] = (df['elevation_m'].fillna(999)<5).astype(int)
        df['rain_x_return'] = (df['historical_rainfall_intensity_mm_hr'] * df['return_period_years'])
        return df
    
    # ------------------------------------------------------------------ #
    # Impute
    # ------------------------------------------------------------------ #
    def _fit_impute(self, df):
        for col in self.numeric_cols:
            self.numerical_medians[col] = df[col].median()
    def _apply_impute(self,df):
        df = df.copy()
        for col in self.numeric_cols:
            df[col] = df[col].fillna(self.numerical_medians[col])
        
        for col in self.categorical_cols:
            df[col] = df[col].fillna('Unknown')
        return df
    
    # ------------------------------------------------------------------ #
    # CLIP PROXIMITY
    # ------------------------------------------------------------------ #
    def _fit_clip(self, df):
        self.proximity_clip_uppers = df['storm_drain_proximity_m'].quantile(self.clip_proximity_uper)
    def _apply_clip(self, df):
        df = df.copy()
        df['storm_drain_proximity_m'] = np.clip(df['storm_drain_proximity_m'], df['storm_drain_proximity_m'].min(), self.proximity_clip_uppers)
        return df
    
    # ------------------------------------------------------------------ #
    # ENCODE
    # ------------------------------------------------------------------ #
    def _fit_label_encoder(self, df):
        self.label_encoders.fit(df[self.target_col])
    def _apply_label_encoder(self, df):
        return self.label_encoders.transform(df[self.target_col])
    
    # ------------------------------------------------------------------ #
    # METADATA
    # ------------------------------------------------------------------ #
    def _attach_metadata(self, df_processed, df_original):
        df_processed = df_processed.copy()
        df_processed['risk_class'] = df_original[self.target_col].values
        df_processed['risk_class_encoded'] = self._apply_label_encoder(df_original)
        df_processed['latitude'] = df_original['latitude'].values
        df_processed['longitude'] = df_original['longitude'].values
        return df_processed
    
    # ------------------------------------------------------------------ #
    #Common fit flow
    # ------------------------------------------------------------------ #
    def _common_fit(self, df):
        df = self._apply_target(df)
        df = self._apply_sentinel(df)
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
        raise NotImplementedError("Subclasses should implement this method.")
    def transform(self, df):
        raise NotImplementedError("Subclasses should implement this method.")
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


# ================================================================== #
#  PIPELINE A — TREE-BASED
#  Decision Tree, Random Forest, XGBoost, LightGBM
# ================================================================== #
class PipelineA(BasePreprocessor):
    def __init__(self):
        super().__init__()
        self.ordinal_encoder = None
        self.feature_cols_out = None

    def _fit_encoder(self, df):
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.ordinal_encoder.fit(df[self.categorical_cols])
    def _apply_encoder(self, df):
        df = df.copy()
        df[self.categorical_cols] = self.ordinal_encoder.transform(df[self.categorical_cols])
        return df
    
    
    
    def fit(self, df: pd.DataFrame):
        print(f"[TreePreprocessor] fitting on {len(df):,} rows...")
        df_common = self._common_fit(df)
        self._fit_encoder(df_common)
        # Lưu feature cols để dùng trong transform
        new_cols = ['is_very_low_elev', 'rain_x_return']
        self.feature_cols_out = self.numeric_cols + new_cols + self.categorical_cols
        print(f"[TreePreprocessor] fit done — {len(self.feature_cols_out)} features")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_common  = self._common_transform(df)
        df_encoded = self._apply_encoder(df_common)
        df_out     = df_encoded[self.feature_cols_out].copy()
        df_out     = self._attach_metadata(df_out, df_common)
        return df_out
    
# ================================================================== #
#  PIPELINE B — LINEAR / DISTANCE-BASED
#  Logistic Regression, SVM, KNN
# ================================================================== #
class PipelineB(BasePreprocessor):

    # Cols bị skew nặng → áp dụng Yeo-Johnson
    SKEW_COLS = [
        'elevation_m',
        'storm_drain_proximity_m',
        'historical_rainfall_intensity_mm_hr',
        'return_period_years',
        'rain_x_return'   # Derived feature cũng có thể skew
    ]

    # soil_group có thứ tự tự nhiên A < B < C < D
    ORDINAL_COLS   = ['soil_group']
    ORDINAL_ORDER  = [['Unknown', 'A', 'B', 'C', 'D']]

    # Categorical không có thứ tự → OneHotEncoder
    OHE_COLS = ['land_use', 'storm_drain_type', 'rainfall_source', 'dem_source']


    def __init__(self, skew_threshold=0.5):
        super().__init__()
        self.skew_threshold = skew_threshold
        self.skew_transforms = {}
        self.ordinal_encoder = None
        self.ohe_encoder = None
        self.scaler = None
        self.ohe_feature_names = None
        self.feature_cols_out = None

    def _fit_skew(self, df: pd.DataFrame):
        self.skew_transforms = {}
        print("  Yeo-Johnson transforms:")
        for col in self.SKEW_COLS:
            if col not in df.columns:
                continue
            skew = df[col].skew()
            if abs(skew) < self.skew_threshold:
                self.skew_transforms[col] = {'method': 'none'}
            else:
                _, lmbda = yeojohnson(df[col].dropna())
                self.skew_transforms[col] = {
                    'method'      : 'yeojohnson',
                    'lambda'      : lmbda,
                    'skew_before' : round(skew, 3)
                }
                print(f"    '{col}': skew={skew:.3f} → λ={lmbda:.4f}")
    
    def _apply_skew(self, df):
        df = df.copy()
        for col, info in self.skew_transforms.items():
            if info['method'] == 'yeojohnson' and col in df.columns:
                df[col] = yeojohnson(df[col], lmbda=info['lambda'])
        return df
    
    def _fit_encoders(self, df):
        # Ordinal Encoder
        self.ordinal_encoder = OrdinalEncoder(
            categories=self.ORDINAL_ORDER,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.ordinal_encoder.fit(df[self.ORDINAL_COLS])

        # One-Hot Encoder
        self.ohe_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        )
        self.ohe_encoder.fit(df[self.OHE_COLS])
        self.ohe_feature_names = self.ohe_encoder.get_feature_names_out(self.OHE_COLS).tolist()

    def _apply_encoders(self, df):
        df = df.copy()
        # Ordinal Encoding
        df[self.ORDINAL_COLS] = self.ordinal_encoder.transform(df[self.ORDINAL_COLS])

        # One-Hot Encoding
        ohe_array = self.ohe_encoder.transform(df[self.OHE_COLS])
        ohe_df = pd.DataFrame(ohe_array, columns=self.ohe_feature_names, index=df.index)
        df = pd.concat([df.drop(columns=self.OHE_COLS), ohe_df], axis=1)
        return df
    
    def _fit_scaler(self, df, scale_cols):
        self.scaler = RobustScaler()
        self.scaler.fit(df[scale_cols])
    def _apply_scaler(self, df, scale_cols):
        df = df.copy()
        df[scale_cols] = self.scaler.transform(df[scale_cols])
        return df
    
    def fit(self, df: pd.DataFrame):
        print(f"\n[LinearPreprocessor] fitting on {len(df):,} rows...")

        df_common = self._common_fit(df)

        # Thêm rain_x_return vào numeric_cols để xử lý
        all_numeric = self.numeric_cols + ['is_very_low_elev', 'rain_x_return']

        # Yeo-Johnson
        self._fit_skew(df_common)
        df_unskewed = self._apply_skew(df_common)

        # Encoders
        self._fit_encoders(df_unskewed)
        df_encoded = self._apply_encoders(df_unskewed)

        # Scale cols = numeric + ordinal + OHE
        self.scale_cols = (
            self.numeric_cols
            + ['is_very_low_elev', 'rain_x_return']
            + self.ORDINAL_COLS
            + self.ohe_feature_names
        )
        self._fit_scaler(df_encoded, self.scale_cols)

        # Lưu feature cols out
        self.feature_cols_out = self.scale_cols
        print(f"[LinearPreprocessor] fit done — {len(self.feature_cols_out)} features")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_common   = self._common_transform(df)
        df_unskewed = self._apply_skew(df_common)
        df_encoded  = self._apply_encoders(df_unskewed)
        df_scaled   = self._apply_scaler(df_encoded, self.scale_cols)
        df_out      = df_scaled[self.feature_cols_out].copy()
        df_out      = self._attach_metadata(df_out, df_common)
        return df_out   