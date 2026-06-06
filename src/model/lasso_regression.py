# src/model/lasso_regression.py

import numpy as np
from src.model.base_regression import BaseOrdinalRegression


class LassoRegressionModel(BaseOrdinalRegression):
    """
    Ordinal Lasso Regression (L1 Regularization).

    Loss: MSE + λ × Σ|wᵢ|

    Gradient:
    ∂Loss/∂W = (2/n) × Xᵀ(ŷ - y) + λ × sign(W)
    ∂Loss/∂b = (2/n) × Σ(ŷ - y)

    Tại sao Lasso phù hợp với dataset này:
    - EDA cho thấy nhiều features có predictive power thấp:
      drainage_density (corr=-0.02), storm_drain_proximity (corr=-0.12)
      các categorical features (land_use, dem_source...)
    - Lasso đẩy weight của features không quan trọng về 0
      → tự động feature selection
    - Kết quả: model chỉ giữ elevation, rainfall, rain_x_return
      là những features có signal mạnh nhất

    Lưu ý kỹ thuật:
    - L1 không differentiable tại w=0 → dùng subgradient sign(w)
    - sign(0) = 0 để tránh oscillation khi weight gần 0
    """

    def __init__(self, alpha: float = 1.0,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-6,
                 random_state: int = 42):
        super().__init__(learning_rate, max_iter, tol, random_state)
        self.alpha = alpha

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray,
                      weights: np.ndarray, sample_weight=None) -> float:
        mse = np.average((y_pred - y) ** 2, weights=sample_weight)
        l1  = self.alpha * np.sum(np.abs(weights))
        return mse + l1

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray,
                          y_pred: np.ndarray,
                          weights: np.ndarray, sample_weight=None) -> tuple[np.ndarray, float]:
        sample_weight = np.ones(len(y)) if sample_weight is None else sample_weight
        weight_sum = np.sum(sample_weight)
        residuals = y_pred - y
        # Subgradient của |w|: sign(w), sign(0)=0
        weighted_residuals = sample_weight * residuals
        grad_w = (2 / weight_sum) * X.T @ weighted_residuals + self.alpha * np.sign(weights)
        grad_b = (2 / weight_sum) * np.sum(weighted_residuals)
        return grad_w, grad_b

    def get_sparsity(self) -> dict:
        """
        Báo cáo feature nào bị Lasso đẩy về 0.
        Hữu ích để phân tích feature importance.
        """
        if self.weights is None:
            return {}
        threshold = 1e-4
        n_zero    = np.sum(np.abs(self.weights) < threshold)
        return {
            'n_features_total'  : len(self.weights),
            'n_features_nonzero': len(self.weights) - n_zero,
            'n_features_zero'   : n_zero,
            'sparsity_%'        : round(n_zero / len(self.weights) * 100, 2)
        }

    def get_params(self, deep=True) -> dict:
        params = super().get_params()
        params['alpha'] = self.alpha
        return params

    def get_param_distributions(self) -> dict:
        return {
            'alpha'        : [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'max_iter'     : [500, 1000, 2000],
        }
    
