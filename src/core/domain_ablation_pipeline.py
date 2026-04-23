import json
from pathlib import Path
from typing import Dict

import pandas as pd

from config.domain_ablation_config import DomainAblationConfig
from src.core.domain_features import DomainFeatureEngineer
from src.model.domain_training import build_xgb_baseline, evaluate_pipeline, make_smote_pipe


class DomainAblationRunner:
    def __init__(self, config: DomainAblationConfig | None = None):
        self.config = config or DomainAblationConfig()

    def run_ablation_fe_groups(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        fe_train, fe_test, fe_groups = DomainFeatureEngineer().transform(train_df, test_df)
        y_train = train_df[self.config.target_col].values
        y_test = test_df[self.config.target_col].values

        baseline = self.config.baseline_features
        variants = [("none", baseline)]
        for group_name, cols in fe_groups.items():
            variants.append((group_name, baseline + cols))
        variants.append(("all", baseline + [feature for cols in fe_groups.values() for feature in cols]))

        rows = []
        for tag, cols in variants:
            X_train = fe_train[cols]
            X_test = fe_test[cols]
            xgb = build_xgb_baseline(self.config)
            pipe = make_smote_pipe(xgb, self.config)
            result = evaluate_pipeline(pipe, X_train, y_train, X_test, y_test, f"XGB_FE_{tag}", self.config)
            rows.append(
                {
                    "variant": tag,
                    "num_features": len(cols),
                    "cv_f2": result.cv_f2,
                    "test_f2": result.test_f2,
                    "gap": result.gap,
                }
            )
        return pd.DataFrame(rows)


def run_from_csv(train_csv: str, test_csv: str, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    runner = DomainAblationRunner()
    fe_ablation = runner.run_ablation_fe_groups(train_df, test_df)
    fe_path = out / "fe_group_ablation.csv"
    fe_ablation.to_csv(fe_path, index=False)

    report = {
        "fe_group_ablation": str(fe_path),
        "best_variant": fe_ablation.sort_values("test_f2", ascending=False).iloc[0]["variant"],
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report": str(report_path), **report}

