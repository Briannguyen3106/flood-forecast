```markdown

# 🚀 Hướng Dẫn Phát Triển Model

## 📁 Cấu Trúc Project

Bạn có thể lấy data ở đây: [Tải dataset](https://drive.google.com/file/d/1hTGvNImpJcetxsGXTYvGlu1NENWsntFe/view?usp=sharing)

ML_PROJECT/
├── config/
│   └── config.yaml              # Cấu hình toàn bộ pipeline
│
├── data/
│   ├── raw/                     # ⛔ KHÔNG chạm vào
│   ├── splits/                  # Train/val/test chưa preprocess
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── processed/               # Data đã preprocess — dùng để train
│       ├── phase1/              # Fit trên train — dùng khi tune
│       │   ├── tree/            # Dành cho: DT, RF, XGBoost, LightGBM
│       │   │   ├── train.csv
│       │   │   ├── val.csv
│       │   │   └── test.csv
│       │   └── linear/          # Dành cho: Logistic, SVM, KNN
│       │       ├── train.csv
│       │       ├── val.csv
│       │       └── test.csv
│       │
│       └── phase2/              # Fit trên train+val — dùng khi final model
│           ├── tree/
│           │   ├── train_val.csv
│           │   └── test.csv
│           └── linear/
│               ├── train_val.csv
│               └── test.csv
│
├── src/
│   ├── core/
│   │   ├── data_splitter.py     # Chia train/val/test
│   │   ├── data_preprocessing.py # TreePreprocessor + LinearPreprocessor
│   │   ├── trainer.py           # Tune hyperparameter + đánh giá
│   │   ├── evaluator.py         # So sánh các model
│   │   └── pipeline.py          # Điều phối toàn bộ flow
│   │
│   └── model/
│       ├── base_model.py        # ⭐ Interface chung — đọc trước
│       ├── decision_tree.py     # Ví dụ tham khảo
│       └── your_model.py        # 👈 Bạn tạo file mới ở đây
│
├── experiments/
│   ├── EDA.ipynb                # Phân tích dữ liệu
│   └── your_name_experiment.ipynb # 👈 Notebook thử nghiệm của bạn
│
├── requirements.txt             # Danh sách thư viện cần cài
├── .gitignore                   # Các file/folder không commit lên git
└── GUIDE.md                     # File này
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

**2. Tạo môi trường ảo** (khuyến khích để tránh conflict thư viện):
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

**5. Kiểm tra cài đặt thành công:**
```bash
python -c "import sklearn; import xgboost; import lightgbm; print('Cài đặt thành công!')"
```

> ⚠️ **Lưu ý:** Mỗi lần mở terminal mới, nhớ kích hoạt lại môi trường ảo (bước 3) trước khi chạy code.

---

## 🔄 Flow Tổng Quan

```
data/processed/phase1/
        │
        ├── train.csv ──→ Trainer.tune()        # Tìm best hyperparameters
        │                     │
        │               CV=5 bên trong train
        │               (tự tạo fold, không đụng val)
        │
        ├── val.csv ───→ Trainer.evaluate_val() # Đánh giá → đăng ký vào Evaluator
        │
        │           Evaluator.compare_val()     # So sánh tất cả model → chọn top 3
        │
        └── [Nếu model của bạn vào top 3]
                │
                ↓
data/processed/phase2/
        ├── train_val.csv ──→ Trainer.retrain()       # Train lại trên nhiều data hơn
        └── test.csv      ──→ Trainer.evaluate_test() # Kết quả cuối cùng 📊
```

---

## ⭐ Vai Trò Các File Core

| File | Vai trò | Bạn có cần sửa? |
|------|---------|-----------------|
| `data_splitter.py` | Chia train/val/test stratified | ❌ Không |
| `data_preprocessing.py` | Xử lý data (impute, clip, scale...) | ❌ Không |
| `trainer.py` | Tune hyperparameter + đánh giá model | ❌ Không |
| `evaluator.py` | So sánh các model, chọn top k | ❌ Không |
| `pipeline.py` | Điều phối toàn bộ flow | ❌ Không |
| `base_model.py` | Interface chung cho tất cả model | ❌ Không |
| **`your_model.py`** | **Model của bạn** | **✅ Bạn tự viết** |

---

## 👨‍💻 Hướng Dẫn Từng Bước

### Bước 1 — Tạo file model của bạn

Tạo file mới trong `src/model/`, kế thừa `BaseModel`:

```python
# src/model/your_model.py


from src.model.base_model import BaseModel

class YourModel(BaseModel):
    def __init__(self):
        super().__init__()

        # ⚠️ QUAN TRỌNG: Chọn đúng pipeline type
        # 'tree'   → Dùng nếu model là: Decision Tree, RF, XGBoost, LightGBM
        # 'linear' → Dùng nếu model là: Logistic Regression, SVM, KNN
        self.pipeline_type = 'tree'  # hoặc 'linear'

    def get_param_distributions(self) -> dict:
        """
        Định nghĩa không gian tìm kiếm hyperparameter.
        RandomizedSearchCV sẽ random sample từ đây.
        """
        return {
            'param_1': [value_1, value_2, value_3],
            'param_2': [value_1, value_2],
        }

    def build(self, **params):
        """Khởi tạo model với params cụ thể"""
        self.model = YourAlgorithm(random_state=42, **params)
        return self
```

### Bước 2 — Đăng ký model vào config (Chưa cần quan tâm vội)

Mở `config/config.yaml`, thêm model của bạn vào đúng nhóm:

```yaml
models:
  tree:                        # Nếu pipeline_type = 'tree'
    - name: "YourModelName"    # Tên hiển thị khi so sánh
      module: "src.model.your_model"   # Đường dẫn file
      class: "YourModel"               # Tên class
```

### Bước 3 — Thử nghiệm trong notebook

Tạo notebook riêng trong `experiments/` để thử nghiệm:

```python
# experiments/your_name_experiment.ipynb

import sys
sys.path.append('..')

import pandas as pd
from src.core.trainer import Trainer
from src.model.your_model import YourModel

# ── Load data ──────────────────────────────────────────
# Dùng đúng pipeline type của model bạn (tree hoặc linear)
train = pd.read_csv('../data/processed/phase1/tree/train.csv')
val   = pd.read_csv('../data/processed/phase1/tree/val.csv')

# ── Phase 1: Tune + Evaluate Val ───────────────────────
model   = YourModel()
trainer = Trainer(model, n_iter=50)

trainer.tune(train)
val_metrics = trainer.evaluate_val(val)

print(f"F2-macro   : {val_metrics['f2_macro']:.4f}")
print(f"F1-weighted: {val_metrics['f1_weighted']:.4f}")
print(f"Best params: {trainer.model.best_params}")

# ── Nếu kết quả tốt → Phase 2 ──────────────────────────
train_val = pd.read_csv('../data/processed/phase2/tree/train_val.csv')
test      = pd.read_csv('../data/processed/phase2/tree/test.csv')

trainer.retrain(train_val)
test_metrics = trainer.evaluate_test(test)
```

### Bước 4 — Cải thiện model

Một số hướng cải thiện có thể thử:

```python
# 1. Mở rộng không gian tìm kiếm hyperparameter
def get_param_distributions(self):
    return {
        'max_depth'   : [3, 5, 10, 15, 20, None],  # Thêm giá trị
        'n_estimators': [100, 200, 300, 500],
    }

# 2. Tăng n_iter để thử nhiều bộ params hơn
trainer = Trainer(model, n_iter=100)  # Mặc định 50

# 3. Thêm class_weight nếu data imbalanced
self.model = YourAlgorithm(
    random_state=42,
    class_weight='balanced',
    **params
)
```

---

## ⚠️ Những Điều KHÔNG Được Làm

```
❌ KHÔNG dùng data/processed/phase1/test.csv để tune hay cải thiện model
   → test set chỉ dùng ở phase 2, sau khi đã chọn xong model tốt nhất

❌ KHÔNG fit preprocessor lại — data đã được xử lý sẵn trong data/processed/
   → chỉ cần pd.read_csv() là dùng được

❌ KHÔNG sửa các file trong src/core/
   → nếu cần thay đổi, thảo luận với cả nhóm trước

❌ KHÔNG commit data lên git
   → liên hệ người phụ trách để lấy data

❌ KHÔNG commit môi trường ảo (venv/) lên git
   → đã được thêm vào .gitignore
```

---

## 💡 Tips

```
✅ Chạy thử với n_iter=10 trước để kiểm tra không có lỗi
   → Sau khi ổn mới tăng lên 50-100

✅ Lưu lại best_params sau khi tune xong
   → print(trainer.model.best_params)

✅ So sánh val score trước và sau khi thay đổi params
   → Để biết thay đổi có giúp ích không

✅ Tham khảo decision_tree.py như một ví dụ hoàn chỉnh

✅ Mỗi người tạo notebook riêng trong experiments/
   → Đặt tên: ten_ban_experiment.ipynb
```

---

## 📞 Liên Hệ

Nếu gặp lỗi hoặc cần thêm tính năng, liên hệ người phụ trách:
- **Pipeline/Preprocessing**: [Nguyễn Danh Bảo]
