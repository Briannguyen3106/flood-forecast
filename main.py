import pandas as pd
from src.core.data_splitter import DataSpliter

train = pd.read_csv("data\\splits\\train.csv")
val = pd.read_csv("data\\splits\\val.csv")
test = pd.read_csv("data\\splits\\test.csv")
print(train.shape)
print(val.shape)
print(test.shape)

print("Phân bố của target SUSCEP:\n")

# Count
df_count = pd.DataFrame({
    'train': train['SUSCEP'].value_counts(),
    'val': val['SUSCEP'].value_counts(),
    'test': test['SUSCEP'].value_counts()
}).fillna(0).astype(int)

print("Số lượng:")
print(df_count.to_string())

# Percentage
df_pct = pd.DataFrame({
    'train': train['SUSCEP'].value_counts(normalize=True) * 100,
    'val': val['SUSCEP'].value_counts(normalize=True) * 100,
    'test': test['SUSCEP'].value_counts(normalize=True) * 100
}).fillna(0)

print("\nTỷ lệ (%):")
print(df_pct.round(2).to_string())
