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

    def _compute_loss(self, y, y_pred, weights, sample_weight=None):
        return np.average((y_pred - y)**2, weights=sample_weight)
    
    def _compute_gradient(self, X, y, y_pred, weights, sample_weight=None):
        sample_weight = np.ones(len(y)) if sample_weight is None else sample_weight
        weight_sum = np.sum(sample_weight)
        residuals = y_pred-y
        weighted_residuals = sample_weight * residuals
        grad_w = (2/weight_sum) * X.T @ weighted_residuals
        grad_b = (2/weight_sum) * np.sum(weighted_residuals)
        return grad_w, grad_b
    
    def get_param_distributions(self) -> dict:
        return {
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'max_iter'     : [500, 1000, 2000],
            'tol'          : [1e-4, 1e-5, 1e-6],
        }
    

