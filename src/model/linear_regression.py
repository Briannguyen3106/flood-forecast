import numpy as np
from src.model.base_regression import BaseOrdinalRegression

class LinearRegressionModel(BaseOrdinalRegression):
    """
    Ordinal Linear Regression.

    Loss: MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

    Gradient:
    ∂Loss/∂W = (2/n) × Xᵀ(ŷ - y)
    ∂Loss/∂b = (2/n) × Σ(ŷ - y)
    """
    def __init__(self, learning_rate = 0.01,
                 max_iter = 1000, tol = 1e-6,
                 random_state = 42):
        super().__init__(learning_rate, max_iter, tol, random_state)

    def _compute_loss(self, y, y_pred, weights):
        return np.mean((y_pred - y)**2)
    
    def _compute_gradient(self, X, y, y_pred, weights):
        n = len(y)
        residuals = y_pred-y
        grad_w = (2/n) * X.T @ residuals
        grad_b = (2/n) * np.sum(residuals)
        return grad_w, grad_b
    
    def get_param_distributions(self) -> dict:
        return {
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'max_iter'     : [500, 1000, 2000],
            'tol'          : [1e-4, 1e-5, 1e-6],
        }
    

