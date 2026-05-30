import numpy as np
from itertools import product
from sklearn.metrics import fbeta_score
from src.model.base_model import BaseModel

class BaseOrdinalRegression(BaseModel):
    def __init__(self, learning_rate = 0.01,
                 max_iter: int=1000,
                 tol: float = 1e-6,
                 random_state: int=42):
        super().__init__()
        self.learning_rate  = learning_rate
        self.max_iter       = max_iter
        self.tol            = tol
        self.random_state   = random_state
        self.pipeline_type  = 'linear'

        self.weights        = None
        self.bias           = None
        self.threshold1     = None
        self.threshold2     = None
        self.loss_history   = []

    def _predict_continuous(self, X):
        return np.dot(X, self.weights) + self.bias
        
    def _compute_loss(self, y, y_pred, weights):
        raise NotImplementedError
    def _compute_gradient(self, X, y, y_pred, weights):
        raise NotImplementedError
        
    def _gradient_descent(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        self.weights = np.random.randn(n_features)*0.01
        self.bias = 0.0
        self.loss_history = []

        bacth_size = min(32, n_samples)
        prev_loss = np.inf

        for iteration in range(self.max_iter):

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start  in range(0, n_samples, bacth_size):
                end = min(start + bacth_size, n_samples)
                X_batch = X_shuffled[start: end]
                y_batch = y_shuffled[start: end]

                y_pred_batch = self._predict_continuous(X_batch)
                grad_w, grad_b = self._compute_gradient(X_batch, y_batch, y_pred_batch, self.weights)

                self.weights -= self.learning_rate * grad_w
                self.bias   -= self.learning_rate * grad_b

            y_pred_full = self._predict_continuous(X)
            loss = self._compute_loss(y, y_pred_full, self.weights)
            self.loss_history.append(loss)

            if abs(prev_loss - loss)< self.tol:
                break
            prev_loss = loss
            if (iteration + 1) % 100 == 0:
                print(f"    Iter {iteration+1}/{self.max_iter}, loss={loss:.6f}")

    def _optimize_thresholds(self, X, y):
        y_pred_continuous  = self._predict_continuous(X)

        y_min, y_max = y_pred_continuous.min(), y_pred_continuous.max()
        candidates = np.linspace(y_min, y_max, 50)

        best_score = -np.inf
        best_1, best_2 = 0.5, 1.5
        for t1, t2 in product(candidates, candidates):
            if t1>=t2:
                continue
            y_class = self._apply_threshold(y_pred_continuous, t1, t2)
            score = fbeta_score(y, y_class, beta=2, average='macro', zero_division=0)

            if score>best_score:
                best_score = score
                best_1, best_2 = t1, t2
        self.threshold1 = best_1
        self.threshold2 = best_2
        print(f"    Threshold optimized: t1={best_1:.4f}, t2={best_2:.4f}")
        print(f"    Train F2-macro: {best_score:.4f}")

    def _apply_threshold(self, y_pred_continuous, t1, t2):
        y_class = np.zeros(len(y_pred_continuous), dtype=int)
        y_class[y_pred_continuous >= t1] = 1
        y_class[y_pred_continuous >= t2] = 2
        return y_class


    def fit(self, X: np.ndarray, y: np.ndarray):
        print(f"  Fitting {self.__class__.__name__}...")
        print(f"  X: {X.shape}, y distribution: {np.bincount(y.astype(int))}")
        self._gradient_descent(X, y.astype(float))
        self._optimize_thresholds(X, y)
        return self
 
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred_continuous = self._predict_continuous(X)
        return self._apply_threshold(
            y_pred_continuous, self.threshold1, self.threshold2
        )
 
    def predict_continuous(self, X: np.ndarray) -> np.ndarray:
        """Dùng để visualize regression output"""
        return self._predict_continuous(X)  
    
    #==================================================
    #Cho imbPipeline
    #==================================================
    
    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'max_iter'     : self.max_iter,
            'tol'          : self.tol,
            'random_state' : self.random_state
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = self
        return self
    
    def get_param_distributions(self):
        raise NotImplementedError
    
    
    def build(self, **params):
        # Reset weights để fit lại từ đầu
        self.weights    = None
        self.bias       = None
        self.threshold1 = None
        self.threshold2 = None
        # Set params mới
        for key, value in params.items():
            setattr(self, key, value)
        self.model = self  # Chính nó là model → ImbPipeline dùng được
        return self
        