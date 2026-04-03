# src/core/evaluator.py

import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, f1_score, classification_report


class Evaluator:
    """
    Trách nhiệm:
    1. Nhận kết quả val metrics từ nhiều model
    2. So sánh và chọn top model
    3. Báo cáo kết quả cuối cùng trên test
    """

    def __init__(self, top_k: int = 3):
        self.top_k       = top_k
        self.val_results = {}   # {model_name: metrics_dict}
        self.test_results = {}  # {model_name: metrics_dict}

    # ------------------------------------------------------------------ #
    def add_val_result(self, model_name: str, metrics: dict):
        """
        Mỗi thành viên sau khi tune xong gọi hàm này
        để đăng ký kết quả val của model mình
        """
        self.val_results[model_name] = metrics
        print(f"Registered val result: {model_name} | "
              f"F2-macro={metrics['f2_macro']:.4f} | "
              f"F1-weighted={metrics['f1_weighted']:.4f}")

    # ------------------------------------------------------------------ #
    def compare_val(self) -> pd.DataFrame:
        """
        So sánh tất cả model trên val set.
        Sắp xếp theo F2-macro (ưu tiên số 1).
        """
        assert self.val_results, "Chưa có kết quả nào, gọi add_val_result() trước"

        df = pd.DataFrame(self.val_results).T
        df = df.sort_values('f2_macro', ascending=False)
        df.index.name = 'model'

        # Đánh dấu top k
        df['top'] = ''
        df.iloc[:self.top_k, df.columns.get_loc('top')] = '✅'

        print("\n========== VAL SET COMPARISON ==========")
        print(df.to_string())
        print(f"\nTop {self.top_k} models: {list(df.index[:self.top_k])}")
        return df

    # ------------------------------------------------------------------ #
    def get_top_models(self) -> list[str]:
        """Trả về tên top k model để retrain + evaluate test"""
        df = self.compare_val()
        return list(df.index[:self.top_k])

    # ------------------------------------------------------------------ #
    def add_test_result(self, model_name: str, metrics: dict):
        """
        Chỉ top k model mới được gọi hàm này.
        Đăng ký kết quả test cuối cùng.
        """
        self.test_results[model_name] = metrics
        print(f"Registered test result: {model_name} | "
              f"F2-macro={metrics['f2_macro']:.4f} | "
              f"F1-weighted={metrics['f1_weighted']:.4f}")

    # ------------------------------------------------------------------ #
    def compare_test(self) -> pd.DataFrame:
        """
        Báo cáo kết quả cuối cùng trên test set.
        Chỉ hiển thị top k model.
        """
        assert self.test_results, "Chưa có kết quả test nào"

        df = pd.DataFrame(self.test_results).T
        df = df.sort_values('f2_macro', ascending=False)
        df.index.name = 'model'

        print("\n========== TEST SET — FINAL RESULTS ==========")
        print(df.to_string())
        print(f"\nBest model: {df.index[0]} "
              f"(F2-macro={df.iloc[0]['f2_macro']:.4f})")
        return df

    # ------------------------------------------------------------------ #
    def full_report(self) -> dict:
        """Trả về cả val và test report để dùng trong notebook/main"""
        return {
            'val' : self.compare_val(),
            'test': self.compare_test()
        }