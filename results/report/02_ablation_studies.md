# Nghiên Cứu Ablation

## 1. Mục đích

Ablation study được dùng để trả lời bốn câu hỏi trước khi tuning final:

1. Có cần xử lý mất cân bằng lớp không, và chiến lược nào phù hợp từng model?
2. Engineered feature group nào thực sự cải thiện generalization?
3. Scaling, skew correction và categorical encoding ảnh hưởng thế nào?
4. Khi kết hợp các lựa chọn tốt nhất, cấu hình model-specific nào nên được khóa?

Toàn bộ ablation chỉ dùng raw train data. Test set không tham gia lựa chọn.

## 2. Protocol chống leakage

Ablation sử dụng `RepeatedStratifiedKFold` với:

- 5 folds.
- 3 repeats.
- 15 validation scores cho mỗi configuration.
- `random_state=42`.
- Primary metric: mean CV F2-macro.
- Secondary checks: F1-weighted, High recall, standard deviation và train/CV gap.

Trong từng fold, imputation, clipping, G4 threshold, Yeo-Johnson lambda,
categorical vocabulary, scaler, SMOTE và model fit chỉ học từ fold train.

## 3. Ablation 1: Imbalance handling

Các chiến lược được so sánh:

- None.
- SMOTE.
- Automatically balanced sample weights.
- Explicit weights `Low:Medium:High = 1:1.5:2`.
- Explicit weights `1:2:3`.
- Explicit weights `1:2:4`.

### 3.1 Kết quả F2-macro

| Strategy | Ridge | Lasso | SVM | Random Forest |
|---|---:|---:|---:|---:|
| None | 0.5625 | 0.3965 | 0.6703 | 0.8434 |
| SMOTE | 0.5217 | 0.4055 | 0.6797 | **0.8500** |
| Balanced | 0.5282 | 0.4064 | 0.7019 | 0.8405 |
| Weights 1:1.5:2 | **0.5636** | 0.4230 | 0.7034 | 0.8427 |
| Weights 1:2:3 | 0.5491 | **0.4274** | **0.7141** | 0.8431 |
| Weights 1:2:4 | 0.5531 | 0.4134 | 0.7062 | 0.8404 |

### 3.2 Diễn giải

- Không có một imbalance strategy thắng cho mọi model.
- Random Forest hưởng lợi nhẹ từ SMOTE, tăng từ 0.8434 lên 0.8500.
- SVM hưởng lợi rõ từ explicit risk weights; profile `1:2:3` tốt nhất.
- Ridge gần như không cần balancing; `1:1.5:2` chỉ nhỉnh hơn None rất ít.
- Lasso cải thiện từ 0.3965 lên 0.4274 với `1:2:3`, nhưng variance vẫn lớn và
  mức tuyệt đối thấp.
- Trọng số cực đoan `1:2:4` không tiếp tục cải thiện, cho thấy tăng ưu tiên High
  quá mạnh có thể làm mất cân bằng precision/recall của các lớp khác.

Kết luận của vòng này là imbalance handling phải được chọn theo model, không
thể ép toàn bộ pipeline dùng SMOTE.

## 4. Ablation 2: Feature engineering groups

Các cấu hình gồm baseline, thêm riêng G1-G4 và dùng tất cả feature groups.

### 4.1 Kết quả F2-macro

| Feature setup | Ridge | Lasso | SVM | Random Forest |
|---|---:|---:|---:|---:|
| Baseline | 0.565 | 0.396 | 0.667 | 0.848 |
| +G1 | 0.567 | 0.400 | 0.668 | 0.839 |
| +G2 | 0.564 | 0.399 | 0.664 | 0.835 |
| +G3 | **0.569** | **0.450** | 0.668 | **0.856** |
| +G4 | 0.564 | 0.398 | 0.663 | 0.840 |
| All | 0.562 | 0.397 | **0.670** | 0.843 |

### 4.2 Diễn giải

- G3 là engineered feature hữu ích và ổn định nhất, đặc biệt với Lasso và RF.
- G1/G2/G4 chỉ tạo thay đổi nhỏ, đôi khi giảm F2.
- Dùng tất cả engineered features không tốt hơn chọn lọc. Điều này cho thấy
  feature engineering dư thừa có thể tăng noise hoặc multicollinearity.
- SVM gần như không nhạy với các feature groups trong vòng này; lợi ích chính
  của SVM đến từ scaling, skew correction và class weights.
- Kết quả vòng này tạo shortlist baseline và G3 cho Ablation 4.

## 5. Ablation 3: Preprocessing steps

Các biến thể Pipeline B gồm baseline, feature engineering, scale, skew, OHE và
các tổ hợp. RF được dùng làm đối chứng tree model.

### 5.1 Điểm nổi bật

| Model | Baseline | Cấu hình nổi bật | F2-macro | Nhận xét |
|---|---:|---|---:|---|
| Ridge | 0.583 | Scale | **0.589** | Scaling cần thiết; skew và tổ hợp phức tạp không giúp |
| Lasso | 0.568 | OHE | **0.574** | OHE tốt nhất; scale-only giảm mạnh |
| SVM | 0.515 | Scale + skew | **0.694** | Cải thiện lớn nhất trong ablation preprocessing |
| RF | 0.848 | Scale | 0.849 | Thay đổi rất nhỏ; tree không cần scaling |

Chi tiết đáng chú ý:

- Ridge giảm xuống khoảng 0.501 khi chỉ skew correction; scale-only đạt 0.589.
- Lasso scale-only chỉ khoảng 0.427; OHE đạt 0.574.
- SVM tăng từ 0.515 baseline lên 0.638 với scale và 0.694 với scale+skew.
- RF dao động hẹp quanh 0.818-0.849, xác nhận preprocessing của linear models
  không nên áp dụng máy móc cho tree family.

## 6. Ablation 4: Filtered model-specific confirmation

Ablation 4 không chạy lại Cartesian product đầy đủ. Mỗi model chỉ thử các cấu
hình đã shortlist từ Ablation 1-3. Winner được chọn bằng mean F2-macro, sau đó
kiểm tra High recall, F1-weighted, variance và gap.

### 6.1 Winner table

| Model | Winner config | CV F2 | Std | CV F1w | High recall | Train/CV gap |
|---|---|---:|---:|---:|---:|---:|
| Decision Tree | `PA_G3_none` | 0.8903 | 0.0219 | 0.9255 | 0.8472 | 0.1097 |
| Random Forest | `PA_base_smote` | 0.8647 | 0.0178 | 0.9105 | 0.7808 | 0.1353 |
| XGBoost | `PA_G3_none` | 0.8963 | 0.0189 | 0.9266 | 0.8513 | 0.1037 |
| XGBRF | `PA_base_smote` | 0.8969 | 0.0166 | 0.9202 | 0.8536 | 0.0668 |
| LightGBM | `PA_G3_none` | 0.8987 | 0.0241 | 0.9274 | 0.8587 | 0.1013 |
| HistGB | `PA_G3_balanced` | **0.9056** | **0.0155** | **0.9309** | **0.8819** | 0.0944 |
| Linear | `scale_w1152` | 0.6005 | 0.0175 | 0.6435 | 0.7406 | 0.0040 |
| Ridge | `scale_base_none` | 0.5887 | 0.0195 | 0.6459 | 0.6772 | 0.0083 |
| Lasso | `ohe_base_none` | 0.5743 | 0.0252 | 0.6553 | 0.6500 | 0.0080 |
| Huber | `scale_w1152` | 0.5968 | 0.0192 | 0.6394 | 0.7195 | 0.0085 |
| SVM | `scale_skew_w123` | 0.7186 | 0.0189 | 0.7886 | 0.6931 | 0.1090 |

### 6.2 Selected model-specific setups

Bảng dưới đây là đầu ra của ablation, không phải giả định có sẵn trong
experiment setup. Đây là các cấu hình được chuyển sang final hyperparameter
tuning:

| Model | Selected preprocessing | Selected feature | Selected imbalance handling |
|---|---|---|---|
| Decision Tree | Pipeline A | G3 | None |
| Random Forest | Pipeline A | Baseline | SMOTE |
| XGBoost | Pipeline A | G3 | None |
| XGBRF | Pipeline A | Baseline | SMOTE |
| LightGBM scratch | Pipeline A | G3 | None |
| HistGradientBoosting | Pipeline A | G3 | Estimator `class_weight='balanced'` |
| Linear Regression | Pipeline B: scale-only | Baseline | Weights `1:1.5:2` |
| Ridge Regression | Pipeline B: scale-only | Baseline | None |
| Lasso Regression | Pipeline B: OHE-only | Baseline | None |
| Huber Regression | Pipeline B: scale-only | Baseline | Weights `1:1.5:2` |
| SVM | Pipeline B: scale + skew | Baseline | Weights `1:2:3` |

HistGB được ghi `imbalance='none'` ở lớp Trainer vì balanced class weight nằm
trong estimator. Đây vẫn là cấu hình balanced đã được ablation chọn; không áp
thêm Trainer sample weights để tránh nhân đôi weighting.

### 6.3 Các so sánh quyết định

**Random Forest:** baseline+SMOTE đạt 0.8647, cao hơn G3+SMOTE 0.8589,
G3+none 0.8555 và baseline+none 0.8476. SMOTE được giữ lại dù train/CV gap
lớn, vì đây là lựa chọn có F2 tốt nhất trong shortlist.

**SVM:** scale+skew+weights 1:2:3 đạt 0.7186, cao hơn scale+weights 0.7016,
scale+skew+none 0.6942 và scale+none 0.6378. Kết quả xác nhận preprocessing và
risk weighting đều quan trọng.

**HistGB:** G3+balanced đạt 0.9056, cao hơn G3+none 0.8953,
baseline+balanced 0.8968 và baseline+none 0.8883. Đây là winner tốt nhất toàn
bộ Ablation 4 theo F2 và High recall.

**XGBRF:** SMOTE tạo cải thiện lớn: baseline+SMOTE 0.8969 so với
baseline+none 0.8467. G3 không cải thiện thêm.

**Decision Tree và LightGBM:** cả hai chọn G3 không SMOTE. SMOTE làm giảm
Decision Tree rõ rệt và không cải thiện LightGBM so với G3+none.

**Linear/Huber:** scale-only với moderate weights `1:1.5:2` tốt hơn các biến
thể skew. **Ridge:** scale-only không balancing tốt nhất. **Lasso:** OHE-only
không balancing thắng trong confirmation dù explicit weights từng tốt hơn ở
ablation imbalance cô lập; điều này cho thấy interaction giữa preprocessing và
balancing phải được xác nhận bằng cấu hình kết hợp.

## 7. Kết luận từ ablation

1. Model-specific setup là cần thiết; không có pipeline chung tối ưu.
2. G3 là engineered feature có giá trị nhất cho tree models, nhưng RF và XGBRF
   vẫn ưu tiên baseline khi kết hợp SMOTE.
3. Scaling là điều kiện gần như bắt buộc cho Ridge, Linear, Huber và SVM.
4. Skew correction đặc biệt hữu ích cho SVM nhưng không giúp linear regression.
5. OHE phù hợp nhất với Lasso trong tập cấu hình được thử.
6. SMOTE chỉ thắng rõ cho RF và XGBRF; nhiều model khác tốt hơn khi không dùng.
7. Các tree models có train/CV gap lớn, do đó final tuning cần regularization.
8. Ablation winner chỉ khóa preprocessing/feature/imbalance setup; best
   hyperparameters cuối cùng vẫn được tìm ở final tuning stage.

Nguồn artifact:

- `results/ablation/ablation1_imbalance.png`
- `results/ablation/ablation2_features.png`
- `results/ablation/ablation3_preprocessing.png`
- `results/ablation/ablation4_final_configs.csv`
- `results/ablation/ablation4_remaining_fast.csv`
- `results/ablation/ablation4_remaining_slow.csv`
- `results/ablation/ablation4_all_winners.csv`
