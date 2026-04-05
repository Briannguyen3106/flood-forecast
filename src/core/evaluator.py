# src/core/evaluator.py

import pandas as pd
import numpy as np
from sklearn.metrics import (
    fbeta_score, f1_score,
    classification_report, confusion_matrix
)


class Evaluator:
    """
    Trách nhiệm:
    1. Thu thập kết quả test từ nhiều model
    2. So sánh và xếp hạng theo F2-macro
    3. Báo cáo kết quả cuối cùng
    """

    def __init__(self, top_k: int = 3):
        self.top_k        = top_k
        self.test_results = {}  # {model_name: metrics_dict}

    # ------------------------------------------------------------------ #
    def add_result(self, model_name: str, metrics: dict):
        """
        Mỗi thành viên sau khi evaluate_test() xong
        gọi hàm này để đăng ký kết quả.
        """
        self.test_results[model_name] = metrics
        print(f"  Registered: {model_name:20s} | "
              f"F2-macro={metrics['f2_macro']:.4f} | "
              f"F1-weighted={metrics['f1_weighted']:.4f}")

    # ------------------------------------------------------------------ #
    def compare(self) -> pd.DataFrame:
        """
        So sánh tất cả model.
        Ưu tiên: F2-macro (số 1) → F1-weighted (số 2).
        """
        assert self.test_results, "Chưa có kết quả, gọi add_result() trước"

        df = pd.DataFrame(self.test_results).T
        df = df.sort_values(
            ['f2_macro', 'f1_weighted'],
            ascending=False
        )
        df.index.name = 'model'

        # Đánh dấu top k
        df['rank'] = range(1, len(df) + 1)
        df['top']  = ''
        df.iloc[:self.top_k, df.columns.get_loc('top')] = '✅'

        print("\n========== TEST SET — FINAL RESULTS ==========")
        print(df.to_string())
        print(f"\nBest model: {df.index[0]} "
              f"(F2-macro={df.iloc[0]['f2_macro']:.4f})")
        return df

    # ------------------------------------------------------------------ #
    def get_top_models(self) -> list[str]:
        """Trả về tên top k model"""
        df = self.compare()
        return list(df.index[:self.top_k])