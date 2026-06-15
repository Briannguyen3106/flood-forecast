# Kết Luận Và Hàm Ý

## 1. Kết luận chính

Corrected experiment xác nhận rằng **HistGradientBoosting (HistGB)** là lựa
chọn chính thức của project. Quyết định này không dựa trên test ranking mà dựa
trên train-only tuning CV F2-macro cao nhất, `0.9338`. Sau khi quyết định đã
khóa, HistGB đồng thời đạt Test F2-macro cao nhất, `0.9388`.

Kết quả quan trọng của HistGB:

- CV F2-macro: `0.9338`.
- Test F2-macro: `0.9388`.
- Test F1-weighted: `0.9392`.
- High recall: `0.9747`.
- High F1: `0.9006`.
- Medium recall: `0.9652`.
- Train-test F2 gap: `0.0251`.
- Dự đoán đúng 77/79 High rows.
- Không dự đoán Medium hoặc High row nào thành Low trên test split.

Các số liệu này phù hợp với mục tiêu vận hành của bài toán: ưu tiên recall và
hạn chế bỏ sót các vùng nguy hiểm.

## 2. Vì sao HistGB được chọn

HistGB có bốn ưu điểm đồng thời:

1. **Selection validity:** đứng đầu train-only tuning CV, nên việc chọn model
   không phụ thuộc vào test set.
2. **Recall profile:** High recall và Medium recall đều cao nhất hoặc gần cao
   nhất trong toàn bộ model set.
3. **Generalization:** khoảng cách train-test nhỏ hơn LightGBM, XGBoost và
   XGBRF.
4. **Regularized capacity:** best setup dùng learning rate 0.01,
   `min_samples_leaf=50`, `max_leaf_nodes=31`, depth 6 và L2=1. Cấu hình này đủ
   nonlinear nhưng hạn chế model complexity.

HistGB sử dụng Pipeline A với G3 và class weighting balanced trong estimator.
G3 kết hợp soil group với rainfall intensity, phản ánh tương tác vật lý hợp lý
giữa khả năng thấm và cường độ mưa.

## 3. Các model thay thế

### 3.1 LightGBM

LightGBM là alternative mạnh nhất khi ưu tiên precision/F1 tổng thể:

- Test F2-macro `0.9386`, gần như bằng HistGB.
- Test F1-weighted `0.9483`, cao nhất toàn bộ experiment.
- High precision `0.9036` và High F1 `0.9259`, đều cao hơn HistGB.

Không đổi lựa chọn sang LightGBM sau khi xem test vì primary model đã được chọn
bằng train-only CV. Tuy nhiên, LightGBM phù hợp làm benchmark hoặc challenger
model trong đánh giá tương lai.

### 3.2 XGBoost

XGBoost tạo cân bằng tốt giữa Low, Medium và High, với Test F2 `0.9294`, High
precision `0.9012` và High recall `0.9241`. Đây là alternative ổn định khi cần
một boosting implementation phổ biến và dễ triển khai hơn custom LightGBM.

### 3.3 Decision Tree

Decision Tree đạt Test F2 `0.9271`, cao hơn Random Forest trong corrected run.
Ưu điểm là dễ giải thích hơn ensemble models. Nhược điểm là kết quả có thể nhạy
với split và ablation từng cho thấy train/CV gap lớn. Nó phù hợp làm interpretable
baseline hơn là model production mặc định.

## 4. Kết luận về model families

### 4.1 Tree models phù hợp hơn

Tree family chiếm toàn bộ sáu vị trí đầu. Điều này cho thấy flood risk không
được mô tả tốt bằng quan hệ tuyến tính đơn giản. Các quan hệ có thể bao gồm:

- Threshold theo elevation hoặc drain proximity.
- Interaction giữa rainfall intensity và return period.
- Interaction giữa soil properties và rainfall.
- Conditional effects khác nhau theo land use và drainage type.

Tree boosting biểu diễn các quan hệ này tự nhiên hơn ordinal linear regression.

### 4.2 Linear models vẫn có giá trị học thuật

Mặc dù hiệu năng thấp hơn, linear family vẫn minh họa rõ:

- Tác động của target ordinal encoding.
- Vai trò của RobustScaler và Yeo-Johnson.
- L1/L2 regularization.
- Robust Huber loss.
- Risk-weighted optimization.
- Giới hạn của linear decision surface trong dữ liệu có nonlinear interactions.

SVM với polynomial kernel thu hẹp khoảng cách nhưng vẫn kém tree models khoảng
0.10-0.14 F2.

## 5. Bài học từ ablation

1. Không có một preprocessing pipeline chung tối ưu cho tất cả models.
2. Không nên mặc định SMOTE luôn cải thiện minority recall.
3. G3 là engineered feature hữu ích nhất, nhưng không phải model nào cũng cần.
4. SVM phụ thuộc mạnh vào scaling và skew correction.
5. Lasso cần OHE trong các cấu hình đã thử, còn Ridge ưu tiên scale-only.
6. Explicit class weights phù hợp SVM/Linear/Huber; RF và XGBRF ưu tiên SMOTE;
   HistGB dùng estimator-level balanced weights.
7. Isolated ablation result cần được xác nhận bằng combined configuration.

## 6. Ý nghĩa vận hành

HistGB có xu hướng ưu tiên recall cho Medium và High. Đổi lại, High precision
0.837 cho thấy có một số Low rows bị cảnh báo High. Trong flood warning, tradeoff
này có thể chấp nhận được nếu chi phí bỏ sót nguy hiểm cao hơn chi phí kiểm tra
cảnh báo dư.

Tuy nhiên, threshold vận hành cuối cùng phải phụ thuộc use case thực tế:

- Nếu nguồn lực kiểm tra hiện trường hạn chế, LightGBM có thể hấp dẫn nhờ High
  precision cao hơn.
- Nếu mục tiêu là screening và không bỏ sót, HistGB phù hợp hơn nhờ High recall.
- Nếu cần giải thích quyết định đơn giản, Decision Tree có thể được dùng làm
  surrogate/interpretable reference, không nhất thiết thay thế HistGB.

## 7. Các giới hạn của nghiên cứu

### 7.1 Một holdout split duy nhất

Final results dựa trên một test split 593 rows. Chênh lệch rất nhỏ giữa HistGB
và LightGBM không đủ để khẳng định model nào luôn tốt hơn trên mọi thành phố,
thời gian hoặc điều kiện khí hậu.

### 7.2 Test set đã được quan sát

Corrected final run đã mở test set. Mọi cải tiến tiếp theo dựa trên test result
sẽ làm test trở thành validation data gián tiếp. Một vòng nghiên cứu mới cần
holdout mới, temporal split, geographic split hoặc external dataset.

### 7.3 Ordinal threshold optimization

Các regression models tối ưu hai class thresholds trên chính fit rows. Cách
này có thể tạo optimistic bias. Out-of-fold threshold optimization hoặc nested
CV sẽ sạch hơn.

### 7.4 Final tuning chỉ có một CV repeat

Ablation dùng 5x3 repeated CV, nhưng final hyperparameter tuning dùng 5 folds
và một repeat. Điều này giảm computational cost nhưng làm estimate của từng
hyperparameter candidate nhạy hơn với fold assignment.

### 7.5 Generalization ngoài phân phối

Chưa có kiểm tra temporal drift, geographic transfer, calibration hoặc dữ liệu
từ thành phố khác. F2 cao trên random stratified split không tự động đảm bảo
khả năng triển khai ngoài distribution hiện tại.

### 7.6 Probability calibration và cost analysis

Experiment đánh giá hard class predictions. Chưa đánh giá calibration, expected
cost, warning capacity hoặc decision curve. Các yếu tố này cần thiết trước khi
triển khai hệ thống cảnh báo thực tế.

## 8. Hướng phát triển tiếp theo

1. Dùng temporal/geographic holdout hoặc external validation set.
2. Thực hiện nested CV cho cả model selection và hyperparameter tuning.
3. Tối ưu ordinal thresholds bằng out-of-fold predictions.
4. Báo cáo confidence intervals bằng bootstrap trên test hoặc repeated outer CV.
5. Đánh giá probability calibration: Brier score, reliability curve và ECE.
6. Phân tích feature importance bằng permutation importance hoặc SHAP trên
   fitted tree models.
7. Kiểm tra fairness/stability theo city, ward, land use và data source.
8. Xây dựng cost-sensitive decision policy dựa trên chi phí false negative của
   lớp High.
9. So sánh HistGB-LightGBM bằng statistical paired tests trên nhiều outer folds
   thay vì chỉ dựa trên chênh lệch `0.00025` Test F2 của một split.
10. Version hóa dataset, code contract và artifact schema cho từng experiment.

## 9. Đoạn kết luận có thể dùng trong báo cáo

> Nghiên cứu đã xây dựng và đánh giá 11 mô hình dự đoán rủi ro ngập đô thị theo
> quy trình chống rò rỉ dữ liệu, trong đó preprocessing và xử lý mất cân bằng
> được fit độc lập trong từng fold. Kết quả ablation cho thấy không tồn tại một
> cấu hình chung tối ưu cho mọi mô hình; tree models phù hợp với Pipeline A,
> trong khi các mô hình tuyến tính phụ thuộc mạnh vào scaling, encoding và
> sample weighting. HistGradientBoosting được lựa chọn bằng train-only tuning
> CV với F2-macro 0.9338 và đạt F2-macro 0.9388 trên test set. Mô hình nhận diện
> đúng 77/79 mẫu High-risk, tương ứng recall 0.9747. LightGBM đạt F1-weighted
> cao nhất 0.9483 và là mô hình thay thế mạnh. Kết quả xác nhận ưu thế của các
> mô hình tree boosting đối với quan hệ phi tuyến giữa địa hình, hạ tầng thoát
> nước, lượng mưa và đặc tính đất. Tuy nhiên, do đánh giá cuối dựa trên một
> holdout split duy nhất, nghiên cứu tiếp theo cần external, temporal hoặc
> geographic validation trước khi xem xét triển khai thực tế.

