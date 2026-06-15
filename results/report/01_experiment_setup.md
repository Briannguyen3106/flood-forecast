# Thiết Lập Thực Nghiệm

## 1. Mục tiêu thực nghiệm

Dự án xây dựng mô hình phân loại đa lớp để dự đoán mức độ rủi ro ngập đô thị
cho từng đoạn không gian. Ba lớp mục tiêu có thứ tự tăng dần về mức độ nguy
hiểm:

```text
Low = 0, Medium = 1, High = 2
```

Mục tiêu chính không chỉ là tối đa hóa độ chính xác tổng thể, mà còn phải hạn
chế bỏ sót các khu vực có rủi ro cao. Vì vậy, **F2-macro** được dùng làm metric
chính để lựa chọn cấu hình và hyperparameter; **F1-weighted** là metric phụ;
recall của lớp `High` là chỉ số vận hành bắt buộc phải kiểm tra.

Tài liệu này mô tả corrected post-fix experiment, không phải các kết quả lịch
sử trước khi sửa thứ tự target và leakage trong cross-validation.

## 2. Dữ liệu và cách chia tập

Dataset có tổng cộng 2.963 dòng và được chia stratified theo tỷ lệ 80/20 với
`random_state=42`:

| Tập dữ liệu | Số dòng | Low | Medium | High |
|---|---:|---:|---:|---:|
| Train | 2.370 | 1.595 | 459 | 316 |
| Test | 593 | 399 | 115 | 79 |
| Tổng | 2.963 | 1.994 | 574 | 395 |

Tỷ lệ lớp trong train xấp xỉ `67.3% Low`, `19.4% Medium`, `13.3% High`. Đây là
bài toán mất cân bằng lớp vừa phải nhưng có tính bất đối xứng về chi phí: dự
đoán sai một dòng `High` thành `Low/Medium` có thể nghiêm trọng hơn một cảnh
báo dư.

Quy tắc sử dụng dữ liệu:

- `data/splits/train.csv` được dùng cho ablation, lựa chọn preprocessing,
  imbalance strategy và hyperparameter tuning.
- `data/splits/test.csv` chỉ được dùng sau khi toàn bộ cấu hình đã khóa.
- Test không tham gia fit imputer, clipping threshold, feature threshold,
  encoder, scaler, SMOTE, class weight, hyperparameter hoặc ordinal threshold.
- Sau corrected final run, test set đã được quan sát. Không được dựa vào test
  score để sửa model rồi tiếp tục gọi kết quả mới là đánh giá trên unseen test.

## 3. Xây dựng target

Trường nguồn `risk_labels` chứa nhiều nhãn được phân tách bằng ký tự `|`. Một
class duy nhất được tạo bằng thứ tự ưu tiên:

1. `High` nếu có `ponding_hotspot` hoặc `extreme_rain_history`.
2. `Medium` nếu không thuộc High nhưng có `low_lying` hoặc `sparse_drainage`.
3. `Low` cho các trường hợp còn lại, chuỗi rỗng hoặc missing.

Mapping phải được gán tường minh là `Low=0, Medium=1, High=2`. Không sử dụng
`LabelEncoder` theo thứ tự alphabet vì điều đó phá vỡ ý nghĩa thứ tự của các mô
hình ordinal regression.

## 4. Shared preprocessing setup

Mọi model đều bắt đầu từ raw train rows. Trong mỗi fold, preprocessor mới được
khởi tạo và fit chỉ trên fold train. Fold validation chỉ gọi `transform()`.

Các bước chung gồm:

1. Tạo `risk_class` và `risk_class_encoded`.
2. Thay sentinel elevation `-3.0` bằng missing value.
3. Tính median cho năm biến số trên fold train và dùng median đó để impute.
4. Điền categorical missing bằng `Unknown`.
5. Học ngưỡng percentile 95% của `storm_drain_proximity_m` trên fold train và
   clipping theo ngưỡng đó.
6. Nếu dùng G4, học percentile 10% của elevation trên fold train.
7. Fit categorical encoders, Yeo-Johnson lambdas và scaler trên fold train.
8. Transform fold validation/test bằng đúng các tham số đã học.

Năm numeric features gốc:

- `elevation_m`
- `drainage_density_km_per_km2`
- `storm_drain_proximity_m`
- `historical_rainfall_intensity_mm_hr`
- `return_period_years`

Năm categorical features gốc:

- `land_use`
- `soil_group`
- `storm_drain_type`
- `rainfall_source`
- `dem_source`

Các cột định danh và metadata như `segment_id`, `city_name`, `admin_ward`,
`catchment_id`, `risk_labels`, `latitude`, `longitude` không được dùng làm
model features.

## 5. Engineered feature groups

| Nhóm | Feature | Ý nghĩa |
|---|---|---|
| G1 | `storm_drain_proximity / drainage_density` | Mức dễ tổn thương hạ tầng thoát nước |
| G2 | `rainfall_intensity * return_period` | Tương tác giữa cường độ mưa và chu kỳ lặp |
| G3 | `soil_group_encoded * rainfall_intensity` | Tương tác giữa khả năng thấm đất và lượng mưa |
| G4 | `elevation < train-fold p10` | Chỉ báo độ cao rất thấp |

Các nhóm này không được đưa vào mặc định cho mọi model. Ablation cho thấy G3
hữu ích với nhiều tree models, trong khi baseline không engineered feature phù
hợp hơn với phần lớn linear models.

## 6. Hai preprocessing pipeline

### 6.1 Pipeline A cho tree models

Pipeline A giữ biểu diễn gọn và không scale:

- Median/Unknown imputation và clipping như phần shared setup.
- Categorical variables được mã hóa bằng `OrdinalEncoder`.
- Category chưa thấy được gán `-1`.
- Baseline có 10 features; thêm G3 tạo 11 features.

Pipeline A được dùng cho Decision Tree, Random Forest, XGBoost, XGBRF,
LightGBM scratch và HistGradientBoosting.

### 6.2 Pipeline B cho linear và distance-based models

Pipeline B có thể bật/tắt độc lập:

- Yeo-Johnson transform khi `abs(skew) >= 0.5`.
- `RobustScaler`, phù hợp hơn StandardScaler khi dữ liệu có outlier hợp lệ.
- Ordinal encoding cho `soil_group` theo `Unknown < A < B < C < D`.
- One-hot encoding hoặc ordinal encoding cho các categorical features còn lại.

Pipeline B được dùng cho Linear, Ridge, Lasso, Huber và custom SVM. Lasso dùng
OHE-only nên có 30 features; các cấu hình scale-only/scale+skew dùng 10
features.

## 7. Candidate space trước ablation

Tại thời điểm thiết kế experiment, chưa có model-specific winner. Nghiên cứu
chỉ định nghĩa trước các lựa chọn ứng viên để ablation đánh giá:

- Preprocessing family: Pipeline A hoặc các biến thể Pipeline B.
- Feature setup: baseline, G1, G2, G3, G4 hoặc tất cả groups.
- Imbalance strategy: none, SMOTE, balanced sample weights và explicit risk
  weights `1:1.5:2`, `1:2:3`, `1:2:4`.
- Pipeline B options: scale, skew correction, OHE và các tổ hợp.

Không có lựa chọn nào trong danh sách trên được xem là cấu hình cuối cùng ở
giai đoạn setup. Các model-specific setups chỉ được xác định sau Ablation 1-4
và được trình bày trong `02_ablation_studies.md`.

SMOTE, nếu được thử, chỉ được fit trên fold train. Validation và test không bao
giờ được oversample.

## 8. Metrics

### 8.1 F2-score và F2-macro

Với từng class:

```text
F2 = 5 * Precision * Recall / (4 * Precision + Recall)
```

F2 đặt trọng số recall cao hơn precision. F2-macro là trung bình không trọng số
của F2 trên ba class, do đó lớp Low đông hơn không thể lấn át Medium và High.

### 8.2 F1-weighted

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

F1-weighted lấy trung bình theo support của từng class. Metric này phản ánh
hiệu năng tổng thể nhưng chịu ảnh hưởng lớn hơn từ lớp Low.

### 8.3 Per-class metrics

Mỗi model được báo cáo precision, recall và F1 cho Low, Medium, High. High recall
được ưu tiên kiểm tra vì false negative của High có chi phí thực tế cao.

### 8.4 Generalization gaps

```text
F2 gap = Train F2-macro - Test F2-macro
F1 gap = Train F1-weighted - Test F1-weighted
```

Gap dương lớn là tín hiệu overfitting. Gap âm nhỏ có thể xảy ra do biến thiên
sampling, regularization hoặc test split tình cờ dễ hơn train distribution.

## 9. Reproducibility và artifact control

Mỗi PKL lưu fitted preprocessor, estimator, feature names, target mapping,
frozen setup, imbalance strategy, best parameters, CV settings và training code
revision. Notebook chỉ reuse artifact khi model-contract metadata khớp. Git
revision khác chỉ tạo cảnh báo để các model đã lưu không bị train lại do thay
đổi tài liệu hoặc logging.

Nguồn số liệu chính:

- `results/final/final_metrics.csv`
- `results/final/per_class_metrics.csv`
- `results/final/detailed_metrics.json`
