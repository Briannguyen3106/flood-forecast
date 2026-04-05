
Bạn có thể lấy data ở đây: [Tải dataset]
(https://drive.google.com/file/d/1WcbXS7YceyNIe2y6LZU0V2sMibMnpsIC/view?usp=sharing)

# 🚀 Hướng Dẫn Phát Triển Model

## 📁 Cấu Trúc Project

```
ML_PROJECT/
├── config/
│   └── config.yaml                  # Cấu hình toàn bộ pipeline
│
├── data/
│   ├── raw/                         # ⛔ KHÔNG chạm vào
│   ├── splits/                      # Train/test chưa preprocess
│   │   ├── train.csv                # 80% (~2370 mẫu)
│   │   └── test.csv                 # 20% (~593 mẫu)
│   │
│   └── processed/                   # Data đã preprocess — dùng để train
│       ├── tree/                    # Dành cho: DT, RF, XGBoost, LightGBM
│       │   ├── train.csv
│       │   └── test.csv
│       └── linear/                  # Dành cho: Logistic, SVM, KNN
│           ├── train.csv
│           └── test.csv
│
├── src/
│   ├── core/
│   │   ├── data_splitter.py         # Chia train/test (80/20)
│   │   ├── data_preprocessing.py    # TreePreprocessor + LinearPreprocessor
│   │   ├── trainer.py               # Trainer + RandomSearchTuner + OptunaTuner
│   │   ├── evaluator.py             # So sánh các model
│   │   └── pipeline.py              # Điều phối toàn bộ flow
│   │
│   └── model/
│       ├── base_model.py            # ⭐ Interface chung — đọc trước
│       ├── decision_tree.py         # Ví dụ tham khảo
│       └── your_model.py            # 👈 Bạn tạo file mới ở đây
│
├── experiments/
│   ├── EDA.ipynb                    # Phân tích dữ liệu
│   ├── EDA_comparison.ipynb         # So sánh Raw vs Tree vs Linear
│   ├── prepare_data.ipynb           # Tạo processed data
│   └── your_name_experiment.ipynb   # 👈 Notebook thử nghiệm của bạn
│
├── requirements.txt                 # Danh sách thư viện
├── .gitignore                       # Các file không commit
├── GUIDE.md                         # File này
└── HIGHLIGHTS.md                    # Highlight quá trình phân tích
```

---

## 🛠️ Cài Đặt Môi Trường

### Yêu cầu
- Python 3.10 trở lên
- Git

### Các bước

**1. Clone project về:**
```bash
git clone <repo_url>
cd ML_PROJECT
```

**2. Tạo môi trường ảo:**
```bash
python -m venv venv
```

**3. Kích hoạt môi trường ảo:**
```bash
# Windows
venv\Scripts\activate

# MacOS/Linux
source venv/bin/activate
```

**4. Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

**5. Kiểm tra cài đặt:**
```bash
python -c "import sklearn; import xgboost; import lightgbm; import optuna; import imblearn; print('Cài đặt thành công!')"
```

> ⚠️ Mỗi lần mở terminal mới, nhớ kích hoạt lại môi trường ảo (bước 3).

> ⚠️ Data không được lưu trên git. Liên hệ người phụ trách để lấy data và đặt vào `data/raw/`.

---

## 🔄 Flow Tổng Quan

```
Raw data (2963 mẫu)
        │
        ▼
DataSplitter (stratified)
        │
        ├── train.csv (80% ~ 2370 mẫu)
        └── test.csv  (20% ~  593 mẫu)  ← cất đi, chỉ dùng cuối cùng
                │
                ▼
        TreePreprocessor.fit_transform(train)    LinearPreprocessor.fit_transform(train)
        TreePreprocessor.transform(test)         LinearPreprocessor.transform(test)
                │
                ▼
        data/processed/tree/                     data/processed/linear/
        ├── train.csv                            ├── train.csv
        └── test.csv                             └── test.csv
                │
                ▼
        Trainer.tune(train)
          └── ImbPipeline: SMOTE → Model
              CV=10 (thay thế val set)
              SMOTE chỉ apply trong train fold → tránh data leak
              Tuner: RandomSearch hoặc Optuna (bạn tự chọn)
                │
                ▼
        Trainer.evaluate_test(test)  ← chỉ chạy 1 lần duy nhất
                │
                ▼
        Evaluator.add_result()
        Evaluator.compare()          ← bảng so sánh cuối cùng
```

---

## ⭐ Vai Trò Các File Core

| File | Vai trò | Bạn có cần sửa? |
|------|---------|-----------------|
| `data_splitter.py` | Chia train/test stratified | ❌ Không |
| `data_preprocessing.py` | Xử lý data (impute, encode, scale...) | ❌ Không |
| `trainer.py` | Tune hyperparameter + đánh giá | ❌ Không |
| `evaluator.py` | So sánh các model | ❌ Không |
| `pipeline.py` | Điều phối toàn bộ flow | ❌ Không |
| `base_model.py` | Interface chung cho tất cả model | ❌ Không |
| **`your_model.py`** | **Model của bạn** | **✅ Bạn tự viết** |

---

## 👨‍💻 Hướng Dẫn Từng Bước

### Bước 1 — Tạo file model của bạn

Tạo file mới trong `src/model/`, kế thừa `BaseModel`:

```python
# src/model/your_model.py

from sklearn.xxx import YourAlgorithm
from src.model.base_model import BaseModel

class YourModel(BaseModel):
    def __init__(self):
        super().__init__()

        # ⚠️ QUAN TRỌNG: Chọn đúng pipeline type
        # 'tree'   → DT, RF, XGBoost, LightGBM
        # 'linear' → Logistic Regression, SVM, KNN
        self.pipeline_type = 'tree'

    def get_param_distributions(self) -> dict:
        """Cho RandomSearchTuner"""
        return {
            'param_1': [value_1, value_2, value_3],
            'param_2': [value_1, value_2],
        }

    def get_optuna_params(self, trial) -> dict:
        """Cho OptunaTuner — dùng trial.suggest_*"""
        return {
            'param_1': trial.suggest_int('param_1', min_val, max_val),
            'param_2': trial.suggest_float('param_2', min_val, max_val),
            'param_3': trial.suggest_categorical('param_3', [val_1, val_2]),
        }

    def build(self, **params):
        """Khởi tạo model với params"""
        self.model = YourAlgorithm(random_state=42, **params)
        return self
```

### Bước 2 — Đăng ký model vào config

```yaml
# config/config.yaml
models:
  tree:                          # Nếu pipeline_type = 'tree'
    - name: "YourModelName"
      module: "src.model.your_model"
      class: "YourModel"
```

### Bước 3 — Thử nghiệm trong notebook

```python
# experiments/your_name_experiment.ipynb

import sys
sys.path.append('..')

import pandas as pd
from src.core.trainer import Trainer, RandomSearchTuner, OptunaTuner
from src.model.your_model import YourModel

# Load data đúng pipeline type
train = pd.read_csv('../data/processed/tree/train.csv')   # hoặc linear/
test  = pd.read_csv('../data/processed/tree/test.csv')

model = YourModel()

# ── Chọn 1 trong 2 tuner ──────────────────────────────

# Option A: RandomSearch (đơn giản, nhanh)
tuner = RandomSearchTuner(n_iter=50, cv=10)

# Option B: Optuna (thông minh hơn, tốt cho model nhiều params)
# tuner = OptunaTuner(n_trials=100, cv=10)

trainer = Trainer(model=model, tuner=tuner)

# Tune
trainer.tune(train)
print(f"Best params: {trainer.model.best_params}")

# Evaluate test — chỉ chạy 1 lần sau khi hài lòng với model
test_metrics = trainer.evaluate_test(test)
```

### Bước 4 — Cải thiện model

```python
# 1. Mở rộng search space
def get_param_distributions(self):
    return {
        'max_depth': [3, 5, 10, 15, 20, None],  # Thêm giá trị
    }

# 2. Dùng Optuna với nhiều trials hơn
tuner = OptunaTuner(n_trials=200, cv=10)

# 3. Xem feature importance (tree models)
import matplotlib.pyplot as plt
best_model = trainer.pipeline.named_steps['model']
importances = best_model.feature_importances_
# ... plot

# 4. Xem overfit gap trong log
# Train F2-macro >> CV F2-macro → overfit → tăng regularization
```

---

## ⚠️ Những Điều KHÔNG Được Làm

```
❌ KHÔNG dùng test set để tune hay cải thiện model
   → test set chỉ chạy 1 lần duy nhất sau khi chọn xong model

❌ KHÔNG fit preprocessor lại
   → data đã xử lý sẵn trong data/processed/

❌ KHÔNG apply SMOTE trên test set
   → SMOTE đã được handle bên trong ImbPipeline

❌ KHÔNG sửa các file trong src/core/
   → thảo luận với cả nhóm trước khi sửa

❌ KHÔNG commit data lên git
   → liên hệ người phụ trách để lấy data

❌ KHÔNG commit venv/ lên git
   → đã có trong .gitignore
```

---

## 💡 Tips

```
✅ Chạy thử với n_iter=10 hoặc n_trials=10 trước
   → Sau khi không có lỗi mới tăng lên 50-100

✅ Với Optuna, dùng trial.suggest_int() cho integer params
   trial.suggest_float() cho float, trial.suggest_categorical() cho list

✅ Lưu lại best_params sau khi tune
   → print(trainer.model.best_params)

✅ Kiểm tra overfit gap trong log
   → Train F2 >> CV F2: overfit → tăng regularization

✅ Tham khảo decision_tree.py như ví dụ hoàn chỉnh

✅ Mỗi người tạo notebook riêng trong experiments/
   → Đặt tên: ten_ban_experiment.ipynb
```

---

## 📦 Thư Viện Chính

| Thư viện | Dùng cho |
|---|---|
| `scikit-learn` | Preprocessing, models, metrics |
| `imbalanced-learn` | SMOTE, ImbPipeline |
| `optuna` | Bayesian hyperparameter tuning |
| `xgboost` | XGBoost model |
| `lightgbm` | LightGBM model |
| `scipy` | Yeo-Johnson transform |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |

---

## 📞 Liên Hệ

Nếu gặp lỗi hoặc cần thêm tính năng:
- **Pipeline/Preprocessing**: [Nguyễn Danh Bảo]
- **Tài liệu tham khảo**: `HIGHLIGHTS.md` — ghi lại các quyết định thiết kế
