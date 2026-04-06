# src/model/base_model.py

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Interface mà tất cả model phải implement.
    Trainer chỉ làm việc với BaseModel — không biết model cụ thể là gì.
    """

    def __init__(self):
        self.model       = None   # Model thực tế bên trong
        self.best_params = None
        self.pipeline_type = None # 'tree' hoặc 'linear' → biết dùng data nào

    @abstractmethod
    def get_param_distributions(self) -> dict:
        """Trả về không gian tìm kiếm hyperparameter"""
        raise NotImplementedError

    @abstractmethod
    def build(self, **params):
        """Khởi tạo model với params cụ thể"""
        raise NotImplementedError

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self
    
    def get_params(self, deep: bool=True):
        return {}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self    