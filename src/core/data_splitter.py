import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataSpliter:
    def __init__(self, target_col, test_size: float = 0.15,
                 val_size: float = 0.15, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.train_df = None
        self.val_df = None
        self.test_df = None

    def split(self, df: pd.DataFrame):
        train_val, test = train_test_split(
            df, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_col]
        )

        val_ratio = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=train_val[self.target_col]
        )

        self.train_df =  train.reset_index(drop=True)
        self.val_df = val.reset_index(drop=True)
        self.test_df = test.reset_index(drop=True)
        self._log(df)
        return self
    
    def get_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert self.train_df is not None, "Chạy split() trước"
        return self.train_df, self.val_df, self.test_df
    
    def get_train_val(self) -> pd.DataFrame:
        assert self.train_df is not None, "Chạy split() trước"
        return pd.concat([self.train_df, self.val_df]).reset_index(drop=True)
    
    def save(self, save_dir: Path):
        assert self.train_df is not None, "Chạy split() trước"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.train_df.to_csv(save_dir / 'train.csv', index=False)
        self.val_df.to_csv(save_dir / 'val.csv', index=False)
        self.test_df.to_csv(save_dir / 'test.csv', index=False)

    def load(self, load_dir: Path):
        load_dir = Path(load_dir)
        self.train_df = pd.read_csv(load_dir / 'train.csv')
        self.val_df = pd.read_csv(load_dir / 'val.csv')
        self.test_df = pd.read_csv(load_dir / 'test.csv')

        self._log(pd.concat([self.train_df, self.val_df, self.test_df]))
        return self
    
    def _log(self, df: pd.DataFrame):
        total = len(df)
        print(f"Total samples: {total}")
        print(f"Train samples: {len(self.train_df)} ({len(self.train_df)/total:.2%})")
        print(f"Val samples: {len(self.val_df)} ({len(self.val_df)/total:.2%})")
        print(f"Test samples: {len(self.test_df)} ({len(self.test_df)/total:.2%})") 
    
