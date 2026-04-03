# src/core/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from scipy.stats import yeojohnson


# ================================================================== #
#  BASE CLASS
# ================================================================== #
class BasePreprocessor:
    """
    Chứa logic chung cho cả 2 pipeline:
    - Aspect encoding (sin/cos)
    - KNNImputer
    - Clip outlier
    - LabelEncoder cho target
    """

    def __init__(self, n_neighbors: int = 5,
                 clip_lower: float = 0.01, clip_upper: float = 0.99):
        self.n_neighbors = n_neighbors
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

        self.imputer = None
        self.label_encoder = None
        self.clip_bounds = {}

        # Aspect sẽ được encode thành sin/cos TRƯỚC khi impute
        self.raw_cols = ['Slope', 'Curvature', 'Aspect', 'TWI', 'FA', 'Drainage', 'Rainfall']
        self.numeric_cols = ['Slope', 'Curvature', 'Aspect_sin', 'Aspect_cos',
                             'TWI', 'FA', 'Drainage', 'Rainfall']
        self.clip_cols = ['Slope', 'Curvature', 'TWI', 'FA']
        # Aspect_sin/cos không clip vì luôn trong [-1, 1]
        self.target_col = 'SUSCEP'

    # ------------------------------------------------------------------ #
    # ASPECT ENCODING
    # ------------------------------------------------------------------ #
    def _apply_aspect_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển Aspect (0-360°) → sin/cos để giữ tính chu kỳ.
        0° và 360° có sin/cos giống nhau ✅
        """
        df = df.copy()
        aspect_rad = df['Aspect'] * np.pi / 180
        df['Aspect_sin'] = np.sin(aspect_rad)
        df['Aspect_cos'] = np.cos(aspect_rad)
        df.drop(columns=['Aspect'], inplace=True)
        return df

    # ------------------------------------------------------------------ #
    # IMPUTER
    # ------------------------------------------------------------------ #
    def _fit_imputer(self, df: pd.DataFrame):
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer.fit(df[self.numeric_cols])

    def _apply_imputer(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed = self.imputer.transform(df[self.numeric_cols])
        return pd.DataFrame(imputed, columns=self.numeric_cols, index=df.index)

    # ------------------------------------------------------------------ #
    # CLIP
    # ------------------------------------------------------------------ #
    def _fit_clip(self, df: pd.DataFrame):
        for col in self.clip_cols:
            self.clip_bounds[col] = (
                df[col].quantile(self.clip_lower),
                df[col].quantile(self.clip_upper)
            )

    def _apply_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (lo, hi) in self.clip_bounds.items():
            df[col] = np.clip(df[col], lo, hi)
        return df

    # ------------------------------------------------------------------ #
    # LABEL ENCODER
    # ------------------------------------------------------------------ #
    def _fit_label_encoder(self, df: pd.DataFrame):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df[self.target_col])

    def _apply_label_encoder(self, df: pd.DataFrame) -> pd.Series:
        return self.label_encoder.transform(df[self.target_col])

    # ------------------------------------------------------------------ #
    # METADATA
    # ------------------------------------------------------------------ #
    def _attach_metadata(self, df_processed: pd.DataFrame,
                         df_original: pd.DataFrame) -> pd.DataFrame:
        """Gắn lại X, Y, SUSCEP, SUSCEP_encoded"""
        df_processed = df_processed.copy()
        df_processed['SUSCEP_encoded'] = self._apply_label_encoder(df_original)
        df_processed['SUSCEP'] = df_original[self.target_col].values
        df_processed[['X', 'Y']] = df_original[['X', 'Y']].values
        return df_processed

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ================================================================== #
#  PIPELINE A — TREE-BASED
#  Decision Tree, Random Forest, XGBoost, LightGBM
# ================================================================== #
class TreePreprocessor(BasePreprocessor):
    """
    Flow:
    Raw → Aspect sin/cos → Impute → Clip(1%-99%) → Done

    Tree không cần scale, fix skew, drop multicollinearity.
    """

    def __init__(self, n_neighbors: int = 5):
        super().__init__(
            n_neighbors=n_neighbors,
            clip_lower=0.01,
            clip_upper=0.99
        )

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        print(f"[TreePreprocessor] fitting on {len(df):,} rows...")

        # 1. Aspect encoding
        df_encoded = self._apply_aspect_encoding(df)

        # 2. Fit imputer
        self._fit_imputer(df_encoded)
        df_imp = self._apply_imputer(df_encoded)

        # 3. Fit clip bounds
        self._fit_clip(df_imp)

        # 4. Fit label encoder
        self._fit_label_encoder(df)

        print(f"[TreePreprocessor] fit done ✅")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = self._apply_aspect_encoding(df)
        df_imp     = self._apply_imputer(df_encoded)
        df_clipped = self._apply_clip(df_imp)
        df_out     = self._attach_metadata(df_clipped, df)
        return df_out


# ================================================================== #
#  PIPELINE B — LINEAR / DISTANCE-BASED
#  Logistic Regression, SVM, KNN
# ================================================================== #
class LinearPreprocessor(BasePreprocessor):
    """
    Flow:
    Raw → Aspect sin/cos → Impute → Clip(5%-95%)
        → Yeo-Johnson → VIF drop → RobustScaler → Done

    Lý do từng bước:
    - Clip chặt hơn : Linear nhạy cảm với outlier hơn Tree
    - Yeo-Johnson   : Linear giả định phân bố gần normal
    - VIF drop      : Loại multicollinearity (TWI ↔ FA = 0.864)
    - RobustScaler  : Đồng đều scale để tránh feature áp đảo nhau
    """

    def __init__(self, n_neighbors: int = 5,
                 skew_threshold: float = 0.5,
                 vif_threshold: float = 10.0):
        super().__init__(
            n_neighbors=n_neighbors,
            clip_lower=0.05,
            clip_upper=0.95
        )
        self.skew_threshold = skew_threshold
        self.vif_threshold  = vif_threshold

        self.skew_transforms = {}
        self.scaler          = None
        self.cols_to_drop    = []
        self.final_cols      = None

    # ------------------------------------------------------------------ #
    # SKEW
    # ------------------------------------------------------------------ #
    def _fit_skew(self, df: pd.DataFrame):
        self.skew_transforms = {}
        print("  Skew transforms:")
        for col in self.numeric_cols:
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

    def _apply_skew(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.numeric_cols:
            info = self.skew_transforms.get(col, {'method': 'none'})
            if info['method'] == 'yeojohnson':
                df[col] = yeojohnson(df[col], lmbda=info['lambda'])
        return df

    # ------------------------------------------------------------------ #
    # VIF
    # ------------------------------------------------------------------ #
    def _fit_vif(self, df: pd.DataFrame):
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        self.cols_to_drop = []
        cols = list(self.numeric_cols)
        print("  VIF analysis:")

        while True:
            X    = df[cols].values.astype(float)
            vifs = [variance_inflation_factor(X, i) for i in range(len(cols))]
            max_vif = max(vifs)
            if max_vif < self.vif_threshold:
                break
            drop_col = cols[vifs.index(max_vif)]
            print(f"    Drop '{drop_col}' (VIF={max_vif:.2f})")
            self.cols_to_drop.append(drop_col)
            cols.remove(drop_col)

        self.final_cols = cols
        print(f"    Features giữ lại: {self.final_cols}")

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame):
        df = df.copy()
        print(f"\n[LinearPreprocessor] fitting on {len(df):,} rows...")

        # 1. Aspect encoding
        df_encoded = self._apply_aspect_encoding(df)

        # 2. Impute
        self._fit_imputer(df_encoded)
        df_imp = self._apply_imputer(df_encoded)

        # 3. Clip
        self._fit_clip(df_imp)
        df_clipped = self._apply_clip(df_imp)

        # 4. Skew
        self._fit_skew(df_clipped)
        df_unskewed = self._apply_skew(df_clipped)

        # 5. VIF drop
        self._fit_vif(df_unskewed)

        # 6. Scaler — chỉ fit trên final_cols
        self.scaler = RobustScaler()
        self.scaler.fit(df_unskewed[self.final_cols])

        # 7. Label encoder
        self._fit_label_encoder(df)

        print(f"[LinearPreprocessor] fit done ✅")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded  = self._apply_aspect_encoding(df)
        df_imp      = self._apply_imputer(df_encoded)
        df_clipped  = self._apply_clip(df_imp)
        df_unskewed = self._apply_skew(df_clipped)

        # Chỉ giữ final_cols sau VIF drop
        df_final = df_unskewed[self.final_cols].copy()

        scaled    = self.scaler.transform(df_final)
        df_scaled = pd.DataFrame(scaled, columns=self.final_cols, index=df.index)

        df_out = self._attach_metadata(df_scaled, df)
        return df_out