# src/model/huber_regression.py

import numpy as np
from src.model.base_regression import BaseOrdinalRegression


class HuberRegressionModel(BaseOrdinalRegression):
    """
    Ordinal Huber Regression.

    Huber Loss kết hợp MSE và MAE:
    L(r) = { r²/2            nếu |r| ≤ δ  (MSE vùng gần 0)
           { δ(|r| - δ/2)    nếu |r| > δ  (MAE vùng xa)

    Gradient:
    ∂L/∂ŷ = { r              nếu |r| ≤ δ
            { δ × sign(r)    nếu |r| > δ

    Tại sao Huber phù hợp với dataset này:
    - Dataset có outlier trong features (giữ nguyên theo EDA):
      historical_rainfall: 120-150 mm/hr (signal của High risk)
      elevation: 200-266m (signal của Low risk)
    - Những điểm này tạo residual lớn trong regression
    - MSE penalize residual lớn theo bình phương → outlier ảnh hưởng mạnh
    - Huber chuyển sang MAE khi residual > δ → ít bị ảnh hưởng hơn
    - Kết quả: weights ổn định hơn, không bị kéo lệch bởi outlier

    Tham số δ (delta):
    - δ nhỏ → gần MAE → robust nhưng hội tụ chậm
    - δ lớn → gần MSE → nhanh nhưng kém robust
    - Tune δ theo IQR của residual là heuristic tốt
    """

    def __init__(self, delta: float = 1.0,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-6,
                 random_state: int = 42):
        super().__init__(learning_rate, max_iter, tol, random_state)
        self.delta = delta

    def _huber_loss(self, residuals: np.ndarray) -> np.ndarray:
        """Tính Huber loss cho từng residual"""
        abs_r = np.abs(residuals)
        loss  = np.where(
            abs_r <= self.delta,
            0.5 * residuals ** 2,                    # MSE vùng gần 0
            self.delta * (abs_r - 0.5 * self.delta)  # MAE vùng xa
        )
        return loss

    def _huber_gradient(self, residuals: np.ndarray) -> np.ndarray:
        """Tính gradient của Huber loss theo residual"""
        abs_r = np.abs(residuals)
        grad  = np.where(
            abs_r <= self.delta,
            residuals,                    # Gradient MSE
            self.delta * np.sign(residuals)  # Gradient MAE
        )
        return grad

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray,
                      weights: np.ndarray) -> float:
        residuals = y_pred - y
        return np.mean(self._huber_loss(residuals))

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray,
                          y_pred: np.ndarray,
                          weights: np.ndarray) -> tuple[np.ndarray, float]:
        n         = len(y)
        residuals = y_pred - y
        # Huber gradient thay thế cho 2×residuals trong MSE
        huber_grad = self._huber_gradient(residuals)
        grad_w = (1 / n) * X.T @ huber_grad
        grad_b = (1 / n) * np.sum(huber_grad)
        return grad_w, grad_b

    def get_params(self, deep=True) -> dict:
        params = super().get_params()
        params['delta'] = self.delta
        return params

    def get_param_distributions(self) -> dict:
        return {
            'delta'        : [0.1, 0.5, 1.0, 1.5, 2.0, 3.0],
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'max_iter'     : [500, 1000, 2000],
        }