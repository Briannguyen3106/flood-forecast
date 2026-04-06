import numpy as np
from src.model.base_regression import BaseOrdinalRegression

class RidgeRegressionModel(BaseOrdinalRegression):
    """
    Ordinal Ridge Regression (L2 Regularization).

    Loss: MSE + λ × Σwᵢ²

    Gradient:
    ∂Loss/∂W = (2/n) × Xᵀ(ŷ - y) + 2λ × W
    ∂Loss/∂b = (2/n) × Σ(ŷ - y)   ← bias không regularize
    """
     
    def __init__(self, alpha = 1.0,
                 learning_rate = 0.01, max_iter = 1000,
                 tol = 1e-6, random_state = 42):
        super().__init__(learning_rate, max_iter, tol, random_state)
        self.alpha = alpha
    
    def _compute_loss(self, y, y_pred, weights):
        residuals = y_pred-y
        l2 = self.alpha * np.sum(weights **2)
        mse = np.mean(residuals**2)

        return mse+l2
    
    def _compute_gradient(self, X, y, y_pred, weights):
        n=len(y)
        residulas = y_pred-y
        grad_w = (2/n) * X.T @ residulas + 2*self.alpha*weights
        grad_b = (2/n) * np.sum(residulas)

        return grad_w, grad_b
    def get_params(self, deep=True) -> dict:
        params = super().get_params()
        params['alpha'] = self.alpha
        return params
    
    def get_param_distributions(self) -> dict:
        return {
            'alpha'        : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'max_iter'     : [500, 1000, 2000],
        }
    
 
