# src/core/trainer.py

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.model.base_model import BaseModel


# ================================================================== #
#  SCORER
# ================================================================== #
f2_macro_scorer = make_scorer(fbeta_score, beta=2, average='macro')


# ================================================================== #
#  TRAINER
# ================================================================== #
class Trainer:
    """
    Trách nhiệm:
    1. Nhận data đã preprocess (train set)
    2. Dùng ImbPipeline: SMOTE → Model
       → SMOTE chỉ apply trên train fold trong CV
       → Val fold giữ nguyên phân bố thực tế
       → Tránh data leak hoàn toàn
    3. Tune hyperparameter bằng RandomizedSearchCV CV=10
    4. Đánh giá trên test set
    """

    def __init__(self, model: BaseModel,
                 n_iter: int = 50,
                 random_state: int = 42,
                 cv: int = 10):
        assert isinstance(model, BaseModel), "model phải kế thừa BaseModel"
        self.model        = model
        self.n_iter       = n_iter
        self.random_state = random_state
        self.cv           = cv
        self.pipeline     = None   # ImbPipeline sau khi tune
        self.test_metrics = {}

    # ------------------------------------------------------------------ #
    def _get_X_y(self, df: pd.DataFrame):
        """Tách features và target"""
        drop_cols    = ['risk_class', 'risk_class_encoded', 'latitude', 'longitude']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].values
        y = df['risk_class_encoded'].values
        return X, y

    # ------------------------------------------------------------------ #
    def _build_pipeline(self) -> ImbPipeline:
        """
        Tạo ImbPipeline: SMOTE → Model

        Tại sao dùng ImbPipeline thay vì sklearn Pipeline:
        - sklearn Pipeline không support resamplers
        - ImbPipeline xử lý đúng: SMOTE chỉ fit trên train fold
          val fold không bị ảnh hưởng → CV score khách quan
        """
        # Tính k_neighbors an toàn
        # Sẽ được override trong tune() nếu cần
        smote = SMOTE(
            k_neighbors  = 5,
            random_state = self.random_state
        )
        self.model.build()

        return ImbPipeline([
            ('smote', smote),
            ('model', self.model.model)
        ])

    # ------------------------------------------------------------------ #
    def _prefix_params(self, param_distributions: dict) -> dict:
        """
        RandomizedSearchCV cần prefix 'model__' cho params của model
        khi dùng trong Pipeline.

        Ví dụ:
            {'n_estimators': [100, 200]}
            → {'model__n_estimators': [100, 200]}
        """
        return {
            f"model__{k}": v
            for k, v in param_distributions.items()
        }

    # ------------------------------------------------------------------ #
    def tune(self, train_df: pd.DataFrame):
        """
        Flow bên trong mỗi CV fold:
        ┌─────────────────────────────────────┐
        │  Train fold (9/10)                  │
        │    → SMOTE fit+transform            │
        │    → Model fit trên resampled data  │
        ├─────────────────────────────────────┤
        │  Val fold (1/10)                    │
        │    → KHÔNG apply SMOTE             │
        │    → Model predict → F2-macro      │
        └─────────────────────────────────────┘
        → Lặp 10 lần → average F2-macro
        → Thử n_iter bộ params → chọn best
        """
        print(f"\n[Trainer] Tuning {self.model.__class__.__name__}...")
        print(f"  CV={self.cv}, n_iter={self.n_iter}, scorer=F2-macro")

        X_train, y_train = self._get_X_y(train_df)

        # Kiểm tra số mẫu minority class
        unique, counts = np.unique(y_train, return_counts=True)
        min_samples    = counts.min()
        k_neighbors    = min(5, min_samples - 1)

        print(f"  Class distribution: {dict(zip(unique, counts))}")
        print(f"  SMOTE k_neighbors : {k_neighbors}")

        # Build pipeline với k_neighbors phù hợp
        pipeline = ImbPipeline([
            ('smote', SMOTE(k_neighbors=k_neighbors,
                            random_state=self.random_state)),
            ('model', self.model.build().model)
        ])

        # Prefix params với 'model__'
        param_distributions = self._prefix_params(
            self.model.get_param_distributions()
        )

        # RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator           = pipeline,
            param_distributions = param_distributions,
            n_iter              = self.n_iter,
            scoring             = f2_macro_scorer,
            cv                  = self.cv,
            random_state        = self.random_state,
            n_jobs              = -1,
            verbose             = 1,
            return_train_score  = True   # Để detect overfit
        )
        search.fit(X_train, y_train)

        # Lưu lại pipeline tốt nhất
        self.pipeline   = search.best_estimator_
        self.model      = self.pipeline.named_steps['model']
        self.model.best_params = {
            # Bỏ prefix 'model__' để dễ đọc
            k.replace('model__', ''): v
            for k, v in search.best_params_.items()
        }

        

        # Log kết quả
        best_cv_score   = search.best_score_
        best_train_score = search.cv_results_['mean_train_score'][search.best_index_]

        print(f"\n  Best params     : {self.model.best_params}")
        print(f"  Best CV F2-macro: {best_cv_score:.4f}")
        print(f"  Train F2-macro  : {best_train_score:.4f}")

        # Cảnh báo overfit
        gap = best_train_score - best_cv_score
        if gap > 0.1:
            print(f"  ⚠️  Overfit gap={gap:.4f} — cân nhắc tăng regularization")
        else:
            print(f"  ✅ Overfit gap={gap:.4f} — ổn")

        return self

    # ------------------------------------------------------------------ #
    def evaluate_test(self, test_df: pd.DataFrame) -> dict:
        """
        Đánh giá CUỐI CÙNG trên test set.

        Dùng pipeline đã tune (bao gồm SMOTE bên trong)
        nhưng khi predict, SMOTE không được gọi —
        chỉ model.predict() được dùng.
        KHÔNG apply SMOTE trên test → phân bố thực tế được giữ nguyên.
        """
        assert self.pipeline is not None, "Chạy tune() trước"

        X_test, y_test = self._get_X_y(test_df)

        # Pipeline.predict chỉ dùng model step, không dùng SMOTE
        y_pred = self.pipeline.predict(X_test)

        self.test_metrics = {
            'f2_macro'   : fbeta_score(y_test, y_pred, beta=2, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        }

        print(f"\n[Test] F2-macro    : {self.test_metrics['f2_macro']:.4f}")
        print(f"[Test] F1-weighted : {self.test_metrics['f1_weighted']:.4f}")
        return self.test_metrics