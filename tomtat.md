# Tom Tat Train_Test.ipynb

Tai lieu nay giai thich luong xu ly trong `experiments/Train_Test.ipynb` theo thu tu cac cell.

## Muc Tieu

Notebook thuc hien bon viec chinh:

1. Tai train/test data da duoc preprocess.
2. Tai model tu file PKL neu model da ton tai, neu khong thi tune va train model moi.
3. Danh gia tung model tren ca train data va test data bang F2-macro, F1-weighted.
4. So sanh, ve bieu do va luu ket qua.

## Luong Tong Quan

```text
Processed train/test data
          |
          v
Kiem tra saved_models/<Model>.pkl
          |
     +----+----+
     |         |
 PKL ton tai   Khong co PKL
     |         |
 load model    tune + train + save PKL
     |         |
     +----+----+
          |
          v
Predict train va test
          |
          v
Tinh F2-macro va F1-weighted
          |
          v
Xep hang, ve bieu do, luu CSV/JSON
```

## Cell 0 - Kaggle Setup

Cell dau tien chua cac lenh du kien dung tren Kaggle:

```python
#!git clone ...
#%cd flood-forecast
#!pip install -r requirements.txt
```

Bo dau `#` neu can clone repository va cai dependencies trong Kaggle notebook.

## Cell 1 - Setup

Cell nay:

- Import pandas, numpy, matplotlib, seaborn, pickle va metrics.
- Import `Trainer` va tat ca model classes.
- Dat `RANDOM_STATE = 42` de ket qua co the lap lai.
- Dat `N_ITER = 100` cho `RandomizedSearchCV`.
- Tao thu muc `results/final` va `saved_models`.

`N_ITER` cang lon thi tim duoc nhieu hyperparameter combinations hon, nhung thoi gian chay cung tang.

## Cell 2 - Load Data

Notebook tai hai loai processed data:

| Data | Model su dung |
|---|---|
| `data/processed/tree` | Decision Tree, Random Forest, XGBoost, XGBRF, LightGBM, HistGB |
| `data/processed/linear` | Linear, Ridge, Lasso, Huber, SVM |

Moi loai co:

- `train.csv`: dung de tune, fit va tinh train metrics.
- `test.csv`: dung de danh gia ket qua cuoi cung.

Khong dung test data de chon hyperparameter.

## Cell 3 - Helper Functions

### `get_X_y(df)`

Loai bo metadata va target khoi features:

```python
drop_cols = [
    'risk_class', 'risk_class_encoded',
    'latitude', 'longitude'
]
```

Ham tra ve `X`, `y` va danh sach feature columns.

### `train_and_save(...)`

Duoc goi khi model chua co file PKL:

1. Tao `Trainer`.
2. Goi `trainer.tune(train_df)`.
3. Tinh metrics tren train data.
4. Tinh metrics tren test data.
5. Luu fitted pipeline vao `saved_models/<model_name>.pkl`.
6. Tra ve trainer va bon metrics train/test.

Pipeline duoc luu thay vi chi luu estimator, vi pipeline chua ca SMOTE step va fitted model.

### `load_pkl(model_name)`

Tai fitted pipeline da co. Nho vay Kaggle khong can train lai model moi lan chay notebook.

PKL chi nen duoc load tu nguon tin cay. Nen su dung cung version scikit-learn, imbalanced-learn, XGBoost va LightGBM da dung khi tao PKL.

### `evaluate_split(pipeline, df)`

Dung cung mot cach tinh metrics cho train va test:

```python
y_pred = pipeline.predict(X)
```

Sau do tinh:

- `F2-macro`: dat trong so recall cao hon precision va cho moi class trong so ngang nhau.
- `F1-weighted`: can bang precision/recall, nhung moi class duoc tinh theo so luong mau.

### `compute_metrics(...)`

Dung cho phan phan tich test data chi tiet:

- F2-macro.
- F1-weighted.
- Precision, recall va F1 cua tung class.
- Confusion matrix.
- Best parameters neu co trong saved model.

## Cell 4 - Model Configuration

`ALL_MODEL_CONFIGS` anh xa moi model voi:

```text
(model class, train dataframe, test dataframe, n_iter)
```

Vi du:

```python
'RandomForest': (
    RandomForestClassifierModel,
    train_tree,
    test_tree,
    N_ITER
)
```

Cell nay cung kiem tra model nao da co PKL va model nao can train.

## Cell 5 - Load Hoac Train Model

Day la cell dieu phoi chinh.

Voi moi model:

### Neu PKL ton tai

```python
pipeline = load_pkl(model_name)
```

Notebook dung pipeline de predict ca train va test, sau do luu:

```text
train_f2_macro
train_f1_weighted
test_f2_macro
test_f1_weighted
```

### Neu PKL khong ton tai

Notebook goi `train_and_save(...)`. Model duoc tune bang cross-validation, fit, danh gia va luu thanh PKL.

Neu mot model bi loi, `try/except` ghi loi va tiep tuc model ke tiep.

Ket qua duoc luu trong:

- `all_results`: train/test metrics.
- `all_pipelines`: fitted pipelines.
- `all_trainers`: Trainer cua model vua train; model load tu PKL co gia tri `None`.

## Cell 6 - Train Data Comparison

Cell nay tao `df_train_results` va xep hang theo:

```python
Train F2-macro
```

Bang train gom:

- Train Rank.
- Model.
- Pipeline type.
- Train F2-macro.
- Train F1-weighted.

Train metrics la in-sample metrics: model duoc danh gia tren chinh cac dong train ban dau, khong phai cac dong SMOTE tong hop.

Train score cao khong tu dong co nghia model tot. No can duoc so sanh voi test score de phat hien overfitting.

## Cell 7 - Test Data Comparison

Cell nay tao `df_test_results` va xep hang model theo Test F2-macro.

Bang test gom:

- Rank.
- Model.
- Pipeline type.
- Test F2-macro.
- Test F1-weighted.
- F2 gap.
- F1 gap.

Gap duoc tinh nhu sau:

```text
F2 gap = Train F2 - Test F2
F1 gap = Train F1 - Test F1
```

Interpretation:

- Gap nho: train va test gan nhau, kha nang generalize tot hon.
- Gap lon: model co kha nang overfit.
- Test F2 cao: model can bang recall giua High, Low va Medium tot hon.
- Test F1-weighted cao: performance tong the tot, nhung class Low co the anh huong lon vi co nhieu mau.

## Detailed Test Metrics

Cell tiep theo chay `compute_metrics()` cho moi model va tao bang per-class:

- High Recall va High F1.
- Medium Recall va Medium F1.
- Low Recall va Low F1.

Trong bai toan flood risk, can chu y `High-Recall`. False negative cua class High co the quan trong hon mot sai sot thong thuong.

## Visualization

Notebook ve:

1. Bar chart so sanh Test F2-macro va Test F1-weighted.
2. Confusion matrix cua ba model co Test F2-macro cao nhat.

Mau sac tach Tree pipeline va Linear pipeline.

## Save Results

Notebook luu:

```text
results/final/final_metrics.csv
results/final/per_class_metrics.csv
results/final/detailed_metrics.json
results/final/model_comparison.png
results/final/confusion_matrices_top3.png
```

`final_metrics.csv` chua train metrics, test metrics va train-test gaps.

`detailed_metrics.json` chua:

- Train/test F2 va F1.
- Best parameters.
- Confusion matrix.
- Precision, recall, F1 va support cua tung class.

## Best Model Summary

Model tot nhat duoc chon theo Test F2-macro, khong phai train score.

Notebook in:

- Train F2/F1.
- Test F2/F1.
- Best parameters.
- Per-class performance.
- Full ranking.

## Cach Chay Tren Kaggle

1. Clone repository va cai dependencies.
2. Bao dam processed data nam dung duong dan.
3. Upload/copy cac PKL vao `saved_models` neu muon bo qua training.
4. Restart kernel sau khi cai package neu Kaggle yeu cau.
5. Run cac cell tu tren xuong duoi.
6. Doc Train Comparison truoc de nhan dien model fit manh.
7. Doc Test Comparison de chon model cuoi cung.
8. Kiem tra F2/F1 gap va per-class recall truoc khi ket luan.

## Luu Y Quan Trong

- PKL ton tai thi notebook khong tune lai model.
- Xoa hoac doi ten PKL neu muon train lai model do.
- Test data chi nen dung cho danh gia cuoi cung.
- Train metrics khong thay the cross-validation metrics.
- Neu thu hyperparameter nhieu lan dua tren test score, test set se gian tiep tro thanh validation set va ket qua khong con khach quan.
- Nen pin versions trong `requirements.txt` de PKL chay nhat quan tren Kaggle.

## Corrected Final Workflow - Da Hoan Tat

### Trang thai notebook final sau khi cap nhat

`Train_Test.ipynb` hien tai khong con tune tren `data/processed`. Notebook:

- doc `data/splits/train.csv` va `data/splits/test.csv` dang raw;
- dung winner setup rieng cua ca 11 model;
- fit preprocessing trong tung tuning fold thong qua `Trainer`;
- checkpoint tung model vao `saved_models/<Model>.pkl` ngay khi hoan tat;
- chi load PKL khi metadata khop target mapping, model, ablation config,
  imbalance strategy va CV settings; code revision khac chi tao canh bao;
- chan cac cell final test neu chua du 11 artifact hop le.

Tren Kaggle, file trong `/kaggle/working` chi ton tai trong session hien tai.
De tiep tuc sau khi session bi huy, can Save Version de luu output hoac upload
`saved_models` thanh Kaggle Dataset, sau do dat `EXTERNAL_MODELS_DIR` trong
cell Setup den thu muc `saved_models` da mount.

Phan mo ta o tren phan anh `Train_Test.ipynb` hien tai. Corrected final run da
hoan tat ngay 2026-06-07 voi 11/11 artifact hop le va 12/12 cell khong co loi.

### Winner setup da khoa

Ket qua nay duoc chon bang train-only repeated CV, khong dung test set:

| Model | Preprocessing | Feature group | Imbalance |
|---|---|---|---|
| Random Forest | Pipeline A | Baseline, khong FE | SMOTE |
| Ridge | Pipeline B scale-only | Baseline | None |
| Lasso | Pipeline B OHE-only | Baseline | None |
| SVM | Pipeline B scale + skew | Baseline | Weights `Low:Medium:High = 1:2:3` |

Bay model con lai cung da duoc khoa:

| Model | Preprocessing | Feature group | Imbalance |
|---|---|---|---|
| Decision Tree | Pipeline A | G3 | None |
| XGBoost | Pipeline A | G3 | None |
| XGBRF | Pipeline A | Baseline | SMOTE |
| LightGBM | Pipeline A | G3 | None |
| HistGradientBoosting | Pipeline A | G3 | Estimator `class_weight='balanced'` |
| Linear Regression | Pipeline B scale-only | Baseline | Weights `1:1.5:2` |
| Huber Regression | Pipeline B scale-only | Baseline | Weights `1:1.5:2` |

Trainer ghi HistGB voi `imbalance='none'` vi balanced class weight nam ben
trong estimator wrapper; khong duoc ap them sample weight lan thu hai.

### Ket qua final post-fix

Final tuning dung raw train, fold-local preprocessing, stratified 5-fold CV,
`CV_REPEATS=1`, `N_ITER=100` va `random_state=42`. Day khong phai repeated CV
o giai doan final tuning; repeated CV duoc dung trong ablation de khoa setup.

| Rank | Model | CV F2 | Test F2 | Test F1-weighted | High recall |
|---:|---|---:|---:|---:|---:|
| 1 | HistGB | 0.9338 | 0.9388 | 0.9392 | 0.9747 |
| 2 | LightGBM | 0.9098 | 0.9386 | 0.9483 | 0.9494 |
| 3 | XGBoost | 0.9020 | 0.9294 | 0.9400 | 0.9241 |
| 4 | DecisionTree | 0.8966 | 0.9271 | 0.9372 | 0.9367 |
| 5 | RandomForest | 0.9003 | 0.9160 | 0.9234 | 0.9367 |
| 6 | XGBRF | 0.9030 | 0.8984 | 0.9164 | 0.8861 |
| 7 | SVM | 0.7386 | 0.7964 | 0.8330 | 0.8228 |
| 8 | Ridge | 0.5999 | 0.6238 | 0.6639 | 0.6835 |
| 9 | Linear Regression | 0.6031 | 0.6101 | 0.6553 | 0.7722 |
| 10 | Huber | 0.6047 | 0.6098 | 0.6521 | 0.7722 |
| 11 | Lasso | 0.5778 | 0.5820 | 0.6427 | 0.7595 |

HistGB duoc chon bang train-only tuning CV va cung la best test performer.
Model du doan dung 77/79 dong `High`. LightGBM co Test F1-weighted cao nhat,
nhung F2-macro van la metric chinh.

### Quy tac su dung test set

- Test set khong duoc dung de chon preprocessing, features, imbalance strategy,
  class weights, thresholds hoac hyperparameters.
- Setup va model selection phai duoc khoa bang train-only CV truoc khi xem test.
- Co the xep hang ket qua test de bao cao, nhung khong duoc dua vao bang xep
  hang do de quay lai sua model va chay test lai.
- Cum tu `best model` trong bao cao final nen duoc giai thich ro la model da
  duoc chon bang CV hay chi la `best test performer`.

### Kaggle

Giu cell `git clone`, `%cd` va cai dependencies vi `Train_Test.ipynb` se chay
tren Kaggle. Khong bat buoc clone dung revision da train PKL, nhung artifact
van phai co schema, target mapping, model va config metadata tuong thich.

## Artifact Status

Ket qua ablation duoc checkpoint tai:

```text
results/ablation/ablation4_remaining_fast.csv
results/ablation/ablation4_remaining_decision_tree.csv
results/ablation/ablation4_remaining_lightgbm.csv
results/ablation/ablation4_all_winners.csv
```

Final reports nam trong `results/final/`; 11 model nam trong `saved_models/`.
Artifact metadata van ghi revision luc train de truy vet. Neu revision hien tai
khac, notebook chi canh bao va van load PKL khi schema, target mapping, model,
config, imbalance, class weights va CV settings deu khop. Neu code thay doi hanh
vi model, can tang `ARTIFACT_SCHEMA_VERSION` hoac xoa PKL bi anh huong.

Test split da duoc xem trong final run. Khong duoc sua config dua tren bang test
nay roi chay lai va coi ket qua la danh gia tren du lieu chua tung thay.
