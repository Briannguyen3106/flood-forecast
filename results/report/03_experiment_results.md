# Hyperparameter Tuning Và Kết Quả Thực Nghiệm

## 1. Final hyperparameter tuning

Final tuning chỉ bắt đầu sau khi Ablation 4 đã khóa preprocessing, feature và
imbalance setup cho từng model. Giai đoạn này không tiếp tục lựa chọn lại các
setup đó mà chỉ tìm model hyperparameters bên trong cấu hình đã chọn.

Protocol:

- `ParameterSampler` trên search space rời rạc.
- `N_ITER=100` candidate samples cho mỗi model.
- Stratified 5-fold CV, `CV_REPEATS=1`.
- `random_state=42`.
- Preprocessing và imbalance handling được fit lại trong từng fold train.
- Candidate xếp theo mean CV F2-macro; tie-break bằng F1-weighted, High recall
  và CV standard deviation thấp hơn.
- Winner được refit trên toàn bộ 2.370 train rows.
- Test set chưa được đọc trong bước tuning.

Ablation dùng 5 folds x 3 repeats, còn final tuning dùng 5 folds x 1 repeat.

## 2. Hyperparameter search spaces

| Model | Search space |
|---|---|
| Decision Tree | `max_depth={3,5,10,15,20,None}`; `min_samples_split={2,5,10,20,50}`; `min_samples_leaf={1,2,5,10,20}`; `criterion={gini,entropy}` |
| Random Forest | `n_estimators={100,200,300,400,500}`; `max_depth={None,8,15,25,35}`; `min_samples_split={2,4,6,8,10,12,15}`; `min_samples_leaf={1,2,3,4,5}`; `max_features={sqrt,log2,0.5,0.7}`; `class_weight={None,balanced,balanced_subsample}` |
| XGBoost | `n_estimators={100,200,300,500,700}`; `learning_rate={0.01,0.03,0.05,0.1,0.2,0.3}`; `max_depth={3,4,5,6,8,10}`; `min_child_weight={1,2,3,5,7,10}`; `subsample` và `colsample_bytree={0.5..1.0}`; `gamma={0,0.1,0.5,1,2,5}`; L1/L2 `{1e-4..10}` |
| XGBRF | `n_estimators={100,200,300,500,600}`; `max_depth={4,5,6,8,10,12}`; `subsample={0.6..1.0}`; `colsample_bytree` và `colsample_bynode={0.5..1.0}`; child weight `{1,2,3,5,7,10}`; L1/L2 `{1e-4..10}` |
| LightGBM scratch | `n_estimators={50,100,150}`; `learning_rate={0.05,0.1,0.2}`; `max_depth={3,4,5}`; `num_leaves={15,31}`; `min_child_samples={20,30,50}`; L2 `{0.1,1,5,10}`; L1 `{0,0.1,0.5,1}`; split gain `{0,0.01,0.1}`; row/column sampling `{0.8,1}`; GOSS `{True,False}` |
| HistGB | `max_iter={100,200,300,400,500,600}`; `learning_rate={0.01,0.03,0.05,0.1,0.2,0.3}`; `max_leaf_nodes={15,31,63,127,255}`; `max_depth={3,4,5,6,8,10}`; `min_samples_leaf={5,10,20,30,40,50}`; L2 `{1e-4..10}` |
| Linear | `learning_rate={0.001,0.005,0.01,0.05,0.1}`; `max_iter={500,1000,2000}`; `tol={1e-4,1e-5,1e-6}` |
| Ridge | `alpha={0.001,0.01,0.1,1,10,100}`; `learning_rate={0.001,0.005,0.01,0.05}`; `max_iter={500,1000,2000}` |
| Lasso | `alpha={0.0001,0.001,0.01,0.1,1,10}`; `learning_rate={0.001,0.005,0.01,0.05}`; `max_iter={500,1000,2000}` |
| Huber | `delta={0.1,0.5,1,1.5,2,3}`; `learning_rate={0.001,0.005,0.01,0.05}`; `max_iter={500,1000,2000}` |
| SVM | `kernel={linear,rbf,poly,sigmoid}`; `C={0.01,0.1,1}`; `tol={1e-3,1e-4}`; `max_passes={5,10}`; RBF/poly/sigmoid thử `gamma={scale,auto,0.01,0.1}`; poly thử degree `{2,3}`; poly/sigmoid thử `coef0={0,1}` |

## 3. Best hyperparameters

| Model | Best parameters từ train-only tuning CV |
|---|---|
| Decision Tree | `criterion=entropy`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=5` |
| Random Forest | `n_estimators=200`, `max_depth=8`, `max_features=0.7`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight=balanced_subsample` |
| XGBoost | `n_estimators=500`, `learning_rate=0.03`, `max_depth=5`, `min_child_weight=5`, `subsample=1.0`, `colsample_bytree=0.8`, `gamma=0.5`, `reg_alpha=0.0001`, `reg_lambda=0.001` |
| XGBRF | `n_estimators=600`, `max_depth=8`, `subsample=0.9`, `colsample_bytree=0.8`, `colsample_bynode=0.9`, `min_child_weight=5`, `reg_alpha=0.01`, `reg_lambda=0.1` |
| LightGBM | `n_estimators=150`, `learning_rate=0.2`, `max_depth=5`, `num_leaves=31`, `min_child_samples=20`, `reg_alpha=1`, `reg_lambda=10`, `min_split_gain=0.1`, `subsample=0.8`, `colsample_bytree=1`, `use_goss=True` |
| HistGB | `max_iter=300`, `learning_rate=0.01`, `max_depth=6`, `max_leaf_nodes=31`, `min_samples_leaf=50`, `l2_regularization=1`, `class_weight=balanced` |
| Linear | `learning_rate=0.05`, `max_iter=2000`, `tol=1e-6` |
| Ridge | `alpha=0.1`, `learning_rate=0.005`, `max_iter=500` |
| Lasso | `alpha=0.0001`, `learning_rate=0.001`, `max_iter=2000` |
| Huber | `delta=1.5`, `learning_rate=0.01`, `max_iter=500` |
| SVM | `kernel=poly`, `C=0.1`, `degree=3`, `gamma=scale`, `coef0=1`, `tol=0.001`, `max_passes=5` |

## 4. Phạm vi final evaluation

Đây là kết quả corrected post-fix từ `Train_Test.ipynb`. Toàn bộ 11 model đã
được tune trên raw train data với fold-local preprocessing. Notebook chỉ mở
final test evaluation sau khi 11/11 artifact vượt qua compatibility gate.

Model selection chính thức dựa trên **train-only tuning CV F2-macro**. Test
ranking chỉ dùng để báo cáo khả năng generalization sau khi quyết định đã khóa.

## 5. Kết quả tổng hợp

| Test rank | Model | CV F2 | Train F2 | Test F2 | Train F1w | Test F1w | F2 gap |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | HistGB | **0.9338** | 0.9639 | **0.9388** | 0.9568 | 0.9392 | 0.0251 |
| 2 | LightGBM | 0.9098 | 0.9868 | 0.9386 | 0.9886 | **0.9483** | 0.0483 |
| 3 | XGBoost | 0.9020 | 0.9772 | 0.9294 | 0.9790 | 0.9400 | 0.0477 |
| 4 | Decision Tree | 0.8966 | 0.9469 | 0.9271 | 0.9597 | 0.9372 | 0.0198 |
| 5 | Random Forest | 0.9003 | 0.9395 | 0.9160 | 0.9429 | 0.9234 | 0.0235 |
| 6 | XGBRF | 0.9030 | 0.9493 | 0.8984 | 0.9507 | 0.9164 | 0.0509 |
| 7 | SVM | 0.7386 | 0.7778 | 0.7964 | 0.8294 | 0.8330 | -0.0186 |
| 8 | Ridge | 0.5999 | 0.6043 | 0.6238 | 0.6515 | 0.6639 | -0.0194 |
| 9 | Linear Regression | 0.6031 | 0.6036 | 0.6101 | 0.6483 | 0.6553 | -0.0065 |
| 10 | Huber | 0.6047 | 0.6030 | 0.6098 | 0.6449 | 0.6521 | -0.0068 |
| 11 | Lasso | 0.5778 | 0.5903 | 0.5820 | 0.6461 | 0.6427 | 0.0083 |

## 6. Per-class test performance

| Model | Low recall | Medium recall | High precision | High recall | High F1 |
|---|---:|---:|---:|---:|---:|
| HistGB | 0.9223 | **0.9652** | 0.8370 | **0.9747** | 0.9006 |
| LightGBM | **0.9524** | 0.9304 | **0.9036** | 0.9494 | **0.9259** |
| XGBoost | 0.9424 | 0.9391 | 0.9012 | 0.9241 | 0.9125 |
| Decision Tree | 0.9348 | 0.9391 | 0.8478 | 0.9367 | 0.8916 |
| Random Forest | 0.9073 | 0.9565 | 0.7872 | 0.9367 | 0.8555 |
| XGBRF | 0.9148 | 0.9304 | 0.8140 | 0.8861 | 0.8485 |
| SVM | 0.8546 | 0.7478 | 0.6566 | 0.8228 | 0.7303 |
| Ridge | 0.6416 | 0.5826 | 0.8060 | 0.6835 | 0.7397 |
| Linear | 0.6667 | 0.4435 | 0.6040 | 0.7722 | 0.6778 |
| Huber | 0.6491 | 0.4783 | 0.5545 | 0.7722 | 0.6455 |
| Lasso | 0.6892 | 0.3304 | 0.6186 | 0.7595 | 0.6818 |

## 7. Confusion matrices của top models

Thứ tự hàng là true class, thứ tự cột là predicted class: Low, Medium, High.

### 7.1 HistGB

```text
[[368, 20, 11],
 [  0,111,  4],
 [  0,  2, 77]]
```

HistGB nhận đúng 77/79 High rows và 111/115 Medium rows. Không có Medium hoặc
High row nào bị dự đoán thành Low. Đây là đặc tính quan trọng trong bài toán
cảnh báo rủi ro: sai số chủ yếu dịch sang lớp lân cận hoặc cảnh báo cao hơn.

### 7.2 LightGBM

```text
[[380, 14,  5],
 [  5,107,  3],
 [  2,  2, 75]]
```

LightGBM cân bằng precision và recall tốt hơn HistGB, đặc biệt High precision
0.9036 và High F1 0.9259. Nó cũng giữ Low recall cao nhất trong top group.

### 7.3 XGBoost

```text
[[376, 16,  7],
 [  6,108,  1],
 [  3,  3, 73]]
```

XGBoost có profile cân bằng, High precision 0.9012 và High recall 0.9241. Nó
đứng thứ ba theo Test F2 và thứ hai theo Test F1-weighted.

## 8. Phân tích theo model family

### 8.1 Tree-based models

Cả sáu tree models đều vượt SVM và ordinal regressors theo F2-macro. Điều này
cho thấy target phụ thuộc mạnh vào nonlinear effects, threshold-like behavior
và interactions giữa rainfall, return period, soil, elevation và drainage.

HistGB có CV F2 cao nhất và test F2 cao nhất. Learning rate nhỏ 0.01, leaf size
50 và L2=1 tạo regularization đáng kể, giúp train-test gap chỉ 0.0251.

LightGBM có test F1-weighted cao nhất 0.9483 và High F1 cao nhất 0.9259. Tuy
nhiên, primary metric là F2-macro và model selection đã khóa theo CV, nên
LightGBM được xem là strong alternative chứ không thay thế HistGB sau khi xem
test.

Decision Tree đạt kết quả bất ngờ tốt với test F2 0.9271 và gap 0.0198. Best
parameters giới hạn depth=10 và leaf size=5, giảm overfitting so với tree không
giới hạn trong ablation.

Random Forest giảm từ vị trí dẫn đầu trong kết quả pre-fix xuống test rank 5.
Điều này minh họa tác động của target-order correction, fold-local preprocessing
và model-specific setup đối với kết luận cuối cùng.

XGBRF có CV tốt nhưng test F2 thấp hơn CV một chút và thấp nhất trong tree
group. High recall 0.8861 vẫn cao hơn linear family nhưng kém các boosting
alternatives.

### 8.2 Linear, ordinal regression và SVM

SVM là model tốt nhất trong linear pipeline: test F2 0.7964 và High recall
0.8228. Polynomial kernel cho phép mô hình hóa nonlinear interactions tốt hơn
linear regressors, nhưng vẫn còn khoảng cách lớn với tree family.

Ridge có F2 tốt nhất trong bốn ordinal regressors (0.6238) và High precision
cao (0.8060), nhưng High recall chỉ 0.6835. Linear và Huber ưu tiên High recall
hơn (0.7722) nhưng Medium recall thấp, làm macro metric giảm.

Lasso có kết quả thấp nhất. OHE tạo 30 features nhưng L1 shrinkage và mô hình
ordinal tuyến tính vẫn không biểu diễn đủ cấu trúc nonlinear. Medium recall
0.3304 là điểm yếu lớn nhất.

## 9. Generalization và overfitting

- HistGB có profile ổn định nhất trong các top models: CV 0.9338, test 0.9388,
  train-test gap 0.0251.
- LightGBM và XGBoost có train F2 rất cao nhưng test vẫn mạnh; gap khoảng
  0.048, cho thấy overfitting vừa phải.
- XGBRF có gap lớn nhất trong final tree results, 0.0509.
- Negative gap ở SVM/Ridge/Linear/Huber không chứng minh test tốt hơn một cách
  tổng quát; nó có thể do sampling variation trên test split 593 rows.
- Không nên chọn model chỉ bằng train-test gap. Metric chính vẫn là train-only
  CV F2 kết hợp với variance và per-class recall.

## 10. So sánh với kết quả pre-fix

Kết quả lịch sử từng chọn Random Forest. Corrected workflow thay đổi ba yếu tố
quan trọng:

1. Target mapping được sửa thành đúng thứ tự ordinal.
2. Preprocessing được chuyển vào từng CV fold.
3. Preprocessing và imbalance strategy được chọn riêng cho từng model.

Do đó, bảng corrected final supersede các kết luận pre-fix. Không nên trộn số
liệu từ hai protocol trong cùng một bảng mà không gắn nhãn rõ ràng.

## 11. Artifact đầu ra

- `results/final/final_metrics.csv`
- `results/final/per_class_metrics.csv`
- `results/final/detailed_metrics.json`
- `results/final/model_comparison.png`
- `results/final/confusion_matrices_top3.png`
- `results/final/training_checkpoint_manifest.csv`
