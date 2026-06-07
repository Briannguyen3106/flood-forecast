# 📌 Highlights — Urban Flood Risk Dataset

> Tài liệu ghi lại các quyết định phân tích, tiền xử lý và xây dựng mô hình.
> Mục đích: làm tư liệu viết báo cáo project.

---

## 1. Tổng Quan Dataset

**Dataset:** Urban Flood Risk Data — Global City Analysis 2025 (Kaggle)
**Kích thước:** 2,963 segments × 17 cột
**Bài toán:** Phân loại mức độ rủi ro lũ lụt đô thị (multiclass classification)

Dataset mô tả các phân đoạn không gian (segments) tại các thành phố trên toàn cầu, mỗi segment được đặc trưng bởi các yếu tố địa hình, hạ tầng thoát nước và lượng mưa lịch sử. Mục tiêu là xây dựng mô hình ML dự đoán mức độ rủi ro lũ lụt của từng segment.

---

## 2. Xử Lý Target

### 2.1 Vấn Đề Với risk_labels Gốc

Cột `risk_labels` trong dataset gốc là **multi-label pipe-delimited**, ví dụ:

```
"ponding_hotspot|low_lying|event_2025-05-02"
"sparse_drainage"
"monitor"
""
```

Có 5 loại label chính:
- `ponding_hotspot`: có bằng chứng thực tế về tích nước
- `extreme_rain_history`: lịch sử mưa cực đoan
- `low_lying`: địa hình trũng thấp so với xung quanh
- `sparse_drainage`: hệ thống thoát nước thưa/kém
- `event_YYYY-MM-DD`: ngày xảy ra sự kiện cụ thể
- `monitor`: theo dõi, chưa phân loại rõ

Dạng multi-label này không phù hợp trực tiếp cho bài toán classification thông thường vì một segment có thể thuộc nhiều nhóm cùng lúc.

### 2.2 Quyết Định: Chuyển Sang 3-Class Theo Priority Rules

**Lý do chọn 3-class thay vì multi-label:**
- Multi-label classification yêu cầu metrics phức tạp hơn (Hamming loss, subset accuracy) và không phải model nào cũng hỗ trợ trực tiếp
- Với scope project, 3-class classification thể hiện được đầy đủ các kỹ thuật ML cần thiết
- 3 mức độ rủi ro (High/Medium/Low) có ý nghĩa thực tiễn rõ ràng trong bài toán quản lý rủi ro lũ lụt

**Priority rules được xây dựng dựa trên mức độ nguy hiểm:**

```
High   → ponding_hotspot HOẶC extreme_rain_history có trong labels
         (bằng chứng thực tế hoặc lịch sử nguy hiểm)

Medium → low_lying HOẶC sparse_drainage (không có High label)
         (yếu tố địa hình/hạ tầng tạo nguy cơ tiềm ẩn)

Low    → monitor, event date, hoặc rỗng
         (chưa có bằng chứng rủi ro rõ ràng)
```

Logic: nếu một segment vừa có `ponding_hotspot` vừa có `low_lying`, nó được xếp vào **High** vì `ponding_hotspot` là bằng chứng thực tế quan trọng hơn.

### 2.3 Phân Bố Sau Khi Chuyển Đổi

| Class | Số mẫu | Tỷ lệ |
|---|---|---|
| Low | 1,994 | 67.3% |
| Medium | 574 | 19.4% |
| High | 395 | 13.3% |

**Imbalance ratio: 5.05x** — số mẫu Low gấp 5 lần High, đây là mức imbalance đáng kể cần xử lý trong quá trình huấn luyện.

---

## 3. Phân Tích Features

### 3.1 Numeric Features

Dataset có 5 numeric features chính:

| Feature | Ý nghĩa | Skewness | Ghi chú |
|---|---|---|---|
| `elevation_m` | Độ cao địa hình (m) | 1.534 🔴 | Feature quan trọng nhất |
| `drainage_density_km_per_km2` | Mật độ hệ thống thoát nước | 0.115 ✅ | Phân bố tốt |
| `storm_drain_proximity_m` | Khoảng cách đến cống thoát gần nhất (m) | 1.610 🔴 | Có outlier |
| `historical_rainfall_intensity_mm_hr` | Cường độ mưa lịch sử (mm/hr) | 1.457 🔴 | Signal High risk |
| `return_period_years` | Chu kỳ lặp lại của sự kiện mưa (năm) | 1.986 🔴 | Discrete values |

**Tương quan với target:**

| Feature | Pearson | Spearman | Nhận xét |
|---|---|---|---|
| `elevation_m` | -0.42 | **-0.589** | Feature dự đoán tốt nhất |
| `historical_rainfall_intensity_mm_hr` | +0.54 | — | Feature quan trọng thứ 2 |
| `return_period_years` | +0.14 | — | Tương quan yếu |
| `storm_drain_proximity_m` | -0.12 | — | Tương quan yếu |
| `drainage_density_km_per_km2` | -0.02 | — | Gần như không tương quan |

**Lý do Spearman của `elevation_m` cao hơn Pearson:**
Pearson đo tương quan tuyến tính, trong khi Spearman đo tương quan đơn điệu. Vì `elevation_m` có phân bố skewed (right-skewed, skew=1.534), mối quan hệ giữa elevation và risk class không hoàn toàn tuyến tính. Spearman (-0.589) phản ánh chính xác hơn rằng "elevation càng thấp thì risk càng cao" bất kể dạng phân bố.

### 3.2 Categorical Features

| Feature | Unique values | Missing | Predictive Power |
|---|---|---|---|
| `land_use` | 9 | 0% | Thấp (phân bố class gần đều) |
| `soil_group` | 4 (A/B/C/D) | 12.2% | Trung bình (D có High risk cao hơn A) |
| `storm_drain_type` | 4 | 6.0% | Thấp |
| `rainfall_source` | 4 | 10.6% | Thấp |
| `dem_source` | 5 | 0% | Thấp |

**Nhận xét về `soil_group`:**
soil_group theo phân loại thủy văn có thứ tự tự nhiên A < B < C < D, trong đó:
- A: khả năng thấm nước cao nhất → ít rủi ro tích nước
- D: khả năng thấm nước thấp nhất → nhiều rủi ro tích nước

Đây là feature duy nhất trong nhóm categorical có thứ tự có ý nghĩa, cần xử lý khác so với các categorical không có thứ tự.

---

## 4. Phân Tích Outlier

### 4.1 Sentinel Value vs Outlier Thực

Đây là điểm quan trọng cần phân biệt rõ:

**Sentinel value** là giá trị đặc biệt được dùng để đánh dấu missing hoặc lỗi dữ liệu, không phải giá trị thực:
- `elevation_m = -3.0`: theo dataset description, đây là fill value cho các ô DEM không có dữ liệu hoặc bị lỗi → cần replace bằng NaN trước khi impute

**Outlier thực** là giá trị hợp lệ về mặt địa lý nhưng nằm ngoài phân bố thông thường:
- `elevation_m = 266.7`: đây là vùng địa hình cao thực sự → hoàn toàn hợp lệ
- `elevation_m = -2.19`: vùng dưới mực nước biển → hoàn toàn hợp lệ

Lưu ý: không phải tất cả giá trị âm đều là sentinel, chỉ đúng `-3.0` mới được xác định là fill value.

### 4.2 Quyết Định Xử Lý Từng Feature

**`elevation_m` (outlier cao 200-266m):**
Phân tích cho thấy 100% outlier này thuộc class **Low risk** — hoàn toàn hợp lý vì vùng địa hình cao ít có nguy cơ tích nước. Việc clip bỏ những giá trị này sẽ làm mất thông tin quan trọng về mối quan hệ giữa elevation cao và Low risk.
→ **Quyết định: Giữ nguyên**

**`historical_rainfall_intensity_mm_hr` (outlier cao 120-150 mm/hr):**
Phân tích cho thấy 100% outlier này thuộc class **High risk** — đây chính là signal quan trọng nhất phân biệt High risk với các class khác. Nếu clip bỏ những giá trị này, model sẽ mất đi khả năng nhận diện các trường hợp rủi ro cao nhất.
→ **Quyết định: Tuyệt đối giữ nguyên, không clip**

**`storm_drain_proximity_m` (outlier 400-750m):**
Outlier phân bố đều trong tất cả các class, không mang signal rõ ràng cho bất kỳ class nào. Những giá trị này có thể là lỗi đo lường hoặc vùng không có hạ tầng thoát nước gần.
→ **Quyết định: Clip tại 95th percentile**

**`return_period_years` (giá trị 50, 100):**
Đây là discrete values hợp lệ trong thiết kế thủy văn (chu kỳ lặp lại tiêu chuẩn: 2, 5, 10, 25, 50, 100 năm). Không phải outlier mà là giá trị thiết kế chuẩn.
→ **Quyết định: Giữ nguyên**

---

## 5. Missing Values

### 5.1 Pattern Phân Tích

Tỷ lệ missing theo từng class:

| Feature | Missing tổng | Low | Medium | High | Pattern |
|---|---|---|---|---|---|
| `soil_group` | 12.2% | ~11% | ~13% | ~14% | Đều giữa các class |
| `rainfall_source` | 10.6% | ~9% | ~11% | ~11% | Đều giữa các class |
| `drainage_density` | 9.6% | ~9% | ~10% | ~10% | Đều giữa các class |
| `storm_drain_proximity_m` | 8.1% | ~8% | ~9% | ~10% | Đều giữa các class |
| `storm_drain_type` | 6.0% | ~4% | ~7% | ~7% | Gần đều |
| `elevation_m` | 5.4% | ~1% | ~7% | ~7% | Hơi lệch |

**Kết luận:** Missing có pattern MCAR (Missing Completely At Random) — tỷ lệ missing không phụ thuộc vào class. Nguyên nhân chính là do thiếu dữ liệu theo thành phố (data collection issue), không liên quan đến mức độ rủi ro.

### 5.2 Chiến Lược Impute

**Numeric features → Median impute:**
Median được chọn thay vì mean vì các numeric features đều có skewness cao (1.4-2.0), khiến mean bị ảnh hưởng mạnh bởi outlier và không đại diện cho giá trị trung tâm thực sự. KNNImputer không được chọn vì dataset nhỏ (2,963 mẫu) — KNN với n_neighbors=5 không ổn định khi số mẫu ít.

**Categorical features → Category 'Unknown':**
Mode impute không được chọn vì missing theo city có thể mang thông tin địa lý — ví dụ, nếu tất cả segments ở một thành phố đều thiếu `soil_group`, việc điền mode của toàn dataset sẽ xóa đi thông tin về thành phố đó. Thêm category 'Unknown' giúp model học được pattern "thiếu thông tin soil_group" có thể liên quan đến đặc điểm địa lý của thành phố đó.

---

## 6. Feature Engineering

### 6.1 is_very_low_elev

```python
is_very_low_elev = (elevation_m < 5).astype(int)
```

**Lý do tạo feature này:**
Phân tích phân bố elevation theo class cho thấy:
- Class Low risk: 0% mẫu có elevation < 5m
- Class High risk: 50.7% mẫu có elevation < 5m
- Class Medium risk: 55.3% mẫu có elevation < 5m

Ngưỡng 5m là ranh giới rõ ràng phân biệt Low risk với High/Medium risk. Bằng cách tạo binary feature này, model có thể nắm bắt được ngưỡng phi tuyến này một cách tường minh, đặc biệt hữu ích cho Linear models vốn không tự học được ngưỡng phi tuyến.

### 6.2 rain_x_return

```python
rain_x_return = historical_rainfall_intensity_mm_hr × return_period_years
```

**Lý do tạo feature này:**
Phân tích cho thấy cả High và Medium risk đều tập trung ở vùng elevation thấp. Điểm khác biệt chính giữa 2 class này là:

| Class | rain_x_return (mean) |
|---|---|
| High | **3,040** |
| Medium | 656 |
| Low | 745 |

High risk = vùng thấp + mưa lớn + chu kỳ lặp dài. Interaction feature `rain_x_return` capture được sự kết hợp này mà 2 features riêng lẻ không thể hiện được đầy đủ. Spearman correlation của `rain_x_return` với target đạt 0.214 (p < 0.0001), cao hơn `return_period_years` đơn lẻ (0.14).

---

## 7. Tiền Xử Lý Dữ Liệu

### 7.1 Hai Pipeline Tiền Xử Lý

Dự án xây dựng 2 pipeline tiền xử lý riêng biệt tương ứng với 2 nhóm thuật toán có yêu cầu khác nhau về dữ liệu đầu vào.

**TreePreprocessor** (dành cho Decision Tree, Random Forest, XGBoost, LightGBM):

Tree-based models học bằng cách tìm **ngưỡng chia** tại mỗi node (ví dụ: `elevation_m < 15`). Vì chỉ so sánh thứ tự, không tính khoảng cách hay đạo hàm, nên:
- Scale không ảnh hưởng: `elevation_m = 15` hay `elevation_m = 1500` cho cùng vị trí chia
- Skewness không ảnh hưởng: phân bố lệch vẫn tìm được ngưỡng tốt
- Multicollinearity không ảnh hưởng: nếu 2 features tương quan cao, tree chỉ dùng 1

Vì vậy TreePreprocessor chỉ cần: sentinel handling → feature engineering → median impute → clip outlier → OrdinalEncoder cho categorical.

**LinearPreprocessor** (dành cho Logistic Regression, SVM, KNN):

Linear/Distance-based models có yêu cầu chặt chẽ hơn:

*Logistic Regression* tìm đường phân chia tuyến tính: `w₁×elevation + w₂×rainfall + ... = 0`. Nếu `elevation` có range [-3, 267] còn `rainfall` có range [5, 150], gradient của elevation sẽ áp đảo, gây ra hội tụ chậm và bias.

*KNN* tính khoảng cách Euclidean: `d = √((e₁-e₂)² + (r₁-r₂)² + ...)`. Feature có range lớn hơn sẽ chiếm ưu thế hoàn toàn trong tính khoảng cách.

*SVM* tìm hyperplane tối đa hóa margin, margin phụ thuộc khoảng cách Euclidean → cùng vấn đề với KNN.

Vì vậy LinearPreprocessor cần thêm: Yeo-Johnson transform → OrdinalEncoder/OneHotEncoder → RobustScaler.

### 7.2 Lý Do Chọn Yeo-Johnson Thay Vì Log Transform

| Transform | Xử lý giá trị âm | Fix right-skew | Fix left-skew |
|---|---|---|---|
| log1p | ❌ | ✅ | ❌ |
| sqrt | ❌ | ✅ nhẹ | ❌ |
| **Yeo-Johnson** | ✅ | ✅ | ✅ |

`elevation_m` có giá trị âm hợp lệ (vùng dưới mực nước biển, ví dụ -2.19m). Log transform sẽ gặp lỗi với giá trị âm, trong khi Yeo-Johnson xử lý được mọi range giá trị. Ngoài ra Yeo-Johnson tự động tìm lambda tối ưu để minimize skewness thông qua MLE, không cần điều chỉnh thủ công.

### 7.3 Encoding Categorical Features

**`soil_group` → OrdinalEncoder (cả 2 pipeline):**
Soil group A/B/C/D có thứ tự tự nhiên về khả năng thấm nước (A tốt nhất, D kém nhất), tương ứng với mức độ rủi ro tăng dần. OneHotEncoder sẽ phá vỡ thứ tự có ý nghĩa này, trong khi OrdinalEncoder giữ được thông tin thứ tự.

**Các categorical còn lại → OrdinalEncoder (Tree) / OneHotEncoder (Linear):**
`land_use`, `storm_drain_type`, `rainfall_source`, `dem_source` không có thứ tự tự nhiên. Tree models không bị ảnh hưởng bởi giá trị số tùy ý của OrdinalEncoder. Linear/Distance models cần OneHotEncoder để tránh mô hình giả định có thứ tự giữa các category.

### 7.4 Lý Do Chọn RobustScaler Thay Vì StandardScaler

RobustScaler sử dụng median và IQR thay vì mean và std:
```
X_scaled = (X - median) / IQR
```

Vì dataset có outlier hợp lệ (đặc biệt `historical_rainfall_intensity_mm_hr` với outlier là signal của High risk, không được clip), StandardScaler sẽ bị ảnh hưởng bởi outlier này làm lệch mean và phóng đại std. RobustScaler ít bị ảnh hưởng hơn vì median và IQR robust với outlier.

---

## 8. Chiến Lược Split Dữ Liệu

### 8.1 Quyết Định Bỏ Validation Set

Ban đầu dự án được thiết kế với 3 tập train/val/test (70/15/15). Tuy nhiên với dataset chỉ có 2,963 mẫu, validation set (~444 mẫu) quá nhỏ để đánh giá đáng tin cậy:

```
Val set với 444 mẫu:
  High  : ~59 mẫu  ← quá ít để ước lượng F2-macro ổn định
  Medium: ~86 mẫu
  Low   : ~299 mẫu
```

Với chỉ 59 mẫu High risk trong val set, một vài dự đoán sai có thể thay đổi F2-macro đáng kể → kết quả không ổn định, không đáng tin cậy để so sánh model.

**Giải pháp:** Thay val set bằng **Stratified K-Fold CV=10** trong train set. Mỗi fold val có ~237 mẫu, average 10 lần cho kết quả ổn định hơn nhiều. Tỷ lệ split được điều chỉnh thành 80/20 để tối đa data cho training.

### 8.2 Stratified Split

Cả train/test split lẫn CV đều dùng stratified sampling để đảm bảo tỷ lệ 3 class (Low/Medium/High) được giữ nguyên trong tất cả các tập. Điều này đặc biệt quan trọng với dataset imbalanced 5x — nếu không stratify, có thể val/test set không có đủ mẫu High risk.

---

## 9. Xử Lý Class Imbalance

### 9.1 Lựa Chọn SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) tạo synthetic samples cho minority classes bằng cách nội suy giữa các mẫu gần nhau trong feature space. So sánh với các phương pháp khác:

| Phương pháp | Ưu điểm | Nhược điểm |
|---|---|---|
| `class_weight='balanced'` | Đơn giản, không tạo data mới | Không support KNN, hiệu quả thấp hơn |
| Random oversampling | Đơn giản | Dễ overfit vì copy nguyên mẫu |
| **SMOTE** | Tạo diverse synthetic samples | Cần cẩn thận tránh data leak |
| ADASYN | Tập trung vùng khó phân loại | Phức tạp hơn SMOTE |

SMOTE được chọn vì cân bằng tốt giữa hiệu quả và độ phức tạp, đồng thời support tất cả model trong dự án.

### 9.2 ImbPipeline — Tránh Data Leak Khi Dùng SMOTE Với CV

Đây là vấn đề kỹ thuật quan trọng. Nếu apply SMOTE trên toàn bộ train set trước khi chạy CV:

```
SMOTE(toàn bộ train) → Synthetic samples được tạo
                        từ toàn bộ train, bao gồm
                        data sẽ vào val fold
CV fold 1:
  train fold → bao gồm synthetic samples từ val fold data ❌
  val fold   → model đã "thấy" thông tin từ val fold qua SMOTE
```

Điều này dẫn đến CV score lạc quan hơn thực tế — một dạng data leak tinh tế.

**Giải pháp: ImbPipeline từ imbalanced-learn:**

```
CV fold 1:
  train fold → SMOTE.fit_transform() → Model.fit()
               (SMOTE chỉ dùng train fold, không biết val fold)
  val fold   → Model.predict()  (phân bố thực tế, không có SMOTE) ✅
```

ImbPipeline đảm bảo SMOTE được fit và apply chỉ trong train fold của mỗi CV iteration, val fold hoàn toàn không bị ảnh hưởng → CV score khách quan và đáng tin cậy.

---

## 10. Lựa Chọn Metric Đánh Giá

### 10.1 F2-macro Là Metric Chính

**F-beta score:**
```
Fβ = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

Khi β=2, Recall được coi trọng gấp 2 lần Precision. Trong bài toán flood risk:
- **False Negative** (bỏ sót vùng nguy hiểm thực sự) → hậu quả nghiêm trọng: không có biện pháp phòng ngừa, thiệt hại về người và tài sản
- **False Positive** (cảnh báo nhầm vùng an toàn) → tốn chi phí phòng ngừa không cần thiết nhưng chấp nhận được

Vì vậy F2 (ưu tiên Recall) phù hợp hơn F1 (cân bằng Precision-Recall) cho bài toán này.

**Macro averaging:**
Tính F2 riêng cho từng class rồi lấy trung bình không trọng số. Điều này đảm bảo class High risk (chỉ 13.3%) có ảnh hưởng ngang bằng class Low risk (67.3%) trong metric tổng hợp — phù hợp khi quan tâm đến hiệu năng đồng đều trên tất cả các class.

### 10.2 Weighted F1 Là Metric Phụ

F1-weighted tính F1 cho từng class rồi lấy trung bình có trọng số theo số mẫu. Metric này phản ánh hiệu năng tổng thể của model trên toàn bộ dataset, hữu ích để so sánh bổ sung bên cạnh F2-macro.

---

## 11. Thiết Kế Hệ Thống Tune Hyperparameter

### 11.1 Hai Tuner: RandomSearch và Optuna

Dự án cung cấp 2 tuner theo strategy pattern, mỗi thành viên tự chọn phù hợp với model của mình:

**RandomizedSearchCV:**
Sample ngẫu nhiên n_iter bộ params từ search space định nghĩa sẵn. Phù hợp cho model ít hyperparameter hoặc khi search space nhỏ. Dễ implement và không cần cài thêm thư viện ngoài sklearn.

**Optuna (TPE Sampler):**
Dùng Tree-structured Parzen Estimator (TPE) — một dạng Bayesian Optimization. Sau mỗi trial, Optuna xây dựng probabilistic model của search space để ước tính vùng nào có khả năng cho kết quả tốt, từ đó sample params tiếp theo thông minh hơn. Phù hợp cho model nhiều hyperparameter như XGBoost, LightGBM nơi search space lớn và random search kém hiệu quả.

### 11.2 Phát Hiện Overfitting

Trainer log cả train score và CV score, tính overfit gap:
```
gap = train_F2 - CV_F2
gap > 0.1 → cảnh báo overfit → cần tăng regularization
```

Đây là thông tin quan trọng giúp thành viên điều chỉnh model kịp thời thay vì chỉ nhìn vào CV score.

---

## 12. Historical Pre-Fix Conclusion From Train and Test Performance

The results in `experiments/Train_Test.ipynb` show that **tree-based models are substantially more suitable for this flood-risk classification problem than linear models**. The five strongest models on the test set are Random Forest, XGBoost, XGBRF, HistGradientBoosting (`HitsGB`), and Decision Tree, all of which achieve a test F2-macro above 0.90. In comparison, the best linear-pipeline model, SVM, reaches only 0.7149, while the regression-based models remain around 0.54-0.56.

| Model | Train F2-macro | Test F2-macro | F2 gap | Test F1-weighted |
|---|---:|---:|---:|---:|
| **Random Forest** | **0.9989** | **0.9315** | 0.0674 | **0.9340** |
| XGBoost | 0.9510 | 0.9255 | **0.0255** | 0.9308 |
| XGBRF | 0.9776 | 0.9230 | 0.0546 | 0.9291 |
| HistGradientBoosting | 0.9478 | 0.9223 | **0.0255** | 0.9320 |
| Decision Tree | 0.9345 | 0.9016 | 0.0329 | 0.9085 |

**Random Forest is the recommended final model** because it obtains the highest test F2-macro (0.9315) and test F1-weighted (0.9340). It also provides the highest recall for the operationally important `High` class (0.975), meaning that it misses very few high-risk areas. Its recall is also strong for `Medium` (0.939), while the `Low` class has an F1-score of 0.955.

However, Random Forest's near-perfect training performance and F2 gap of 0.0674 indicate **moderate overfitting**. Its test performance remains the best, so this does not invalidate the selection, but the gap should be monitored on new or out-of-distribution data. XGBoost and HistGradientBoosting are strong alternatives when model stability is prioritized: both retain test F2-macro above 0.92 with much smaller train-test gaps of about 0.0255.

LightGBM has the smallest gap among the tree models, but its lower test F2-macro (0.8493) and especially low `High`-class recall (0.633) make it unsuitable as the primary flood-warning model. Similarly, the linear and regression models underfit the nonlinear relationships in the dataset and should not be selected for deployment.

### 12.1 Analysis of Linear-Based Models

| Model | Train F2-macro | Test F2-macro | F2 gap | Test F1-weighted | High recall | Medium recall |
|---|---:|---:|---:|---:|---:|---:|
| **SVM** | **0.8156** | **0.7149** | 0.1007 | **0.7401** | **0.785** | **0.739** |
| **Huber** | 0.5901 | **0.5641** | 0.0260 | **0.5855** | 0.582 | **0.678** |
| Linear Regression | 0.5766 | 0.5623 | 0.0143 | 0.5776 | 0.608 | 0.652 |
| Lasso | 0.5645 | 0.5569 | 0.0076 | 0.5483 | 0.671 | 0.678 |
| Ridge | 0.5674 | 0.5397 | 0.0277 | 0.5430 | 0.633 | 0.635 |

Among the **regression-based models**, Huber is the best overall: it obtains the highest test F2-macro (0.5641) and test F1-weighted (0.5855). Huber reduces the influence of samples with large residuals, so its advantage over ordinary Linear Regression suggests that the dataset may contain noisy observations, outliers, or cases that do not follow the dominant linear pattern. The improvement is small, however: Huber exceeds Linear Regression by only 0.0018 in test F2-macro and 0.0079 in test F1-weighted. Therefore, the evidence supports a modest robustness benefit rather than a major performance improvement.

Ridge and Lasso do not improve generalization over unregularized Linear Regression. Ridge reduces test F2-macro from 0.5623 to 0.5397, while Lasso reduces it to 0.5569. Their small train-test gaps show that this is mainly **underfitting rather than overfitting**. The base linear model already has limited capacity, and coefficient shrinkage adds bias without solving the more important problem: flood-risk classes have nonlinear relationships and interactions that a linear decision function cannot adequately represent. Lasso may also remove weak features that are individually small but useful when combined with other variables.

The class-level results add an important qualification. Linear Regression has slightly better `High`-class recall than Huber (0.608 versus 0.582), while Huber improves `Medium` recall and overall weighted F1. Lasso achieves the best `High` recall among the four regressors (0.671), but its poor precision and performance on other classes keep its aggregate scores low. Thus, Huber is the most balanced regression model, but it is not the best on every individual class.

SVM should be evaluated separately from the four regressors. It is the **best linear-pipeline model overall**, with a test F2-macro of 0.7149, because its margin-based classification objective is better aligned with class separation than predicting encoded class values through regression. Nevertheless, its large F2 gap (0.1007) indicates more overfitting, and it remains well behind every major tree ensemble. Consequently, Huber is the preferred regression baseline, SVM is the preferred linear classifier, and neither should replace the selected tree-based model.

Overall, the experiment supports deploying **Random Forest as the primary model**, with XGBoost or HistGradientBoosting retained as benchmark alternatives. Future evaluation should use new temporal or geographic data to confirm that Random Forest's strong test results generalize beyond the current split.

---

## 13. Corrected Post-Fix Experiment Status (2026-06-07)

Section 12 reports historical pre-regeneration test results. It must not be
treated as the final corrected conclusion. Since that run, the project fixed
the ordinal target mapping and moved learned preprocessing inside CV folds.

### 13.1 Completed corrected workflow

- Canonical mapping is `Low=0, Medium=1, High=2`.
- Data preparation artifacts were regenerated.
- Ablation uses raw train rows and fold-local preprocessing.
- SMOTE is applied only to CV training folds.
- Ablation 4 and 4b confirmed model-specific setups for all 11 models.

| Model | Selected setup | CV F2-macro | High recall | Train/CV gap |
|---|---|---:|---:|---:|
| Random Forest | Pipeline A baseline + SMOTE | 0.8647 | 0.7808 | 0.1353 |
| SVM | scale + skew + weights `1:2:3` | 0.7186 | 0.6931 | 0.1090 |
| Ridge | scale-only + no balancing | 0.5887 | 0.6772 | 0.0083 |
| Lasso | OHE-only + no balancing | 0.5743 | 0.6500 | 0.0080 |

These results demonstrate that one shared preprocessing and SMOTE strategy is
not appropriate for all model families. The completed winner table is stored
in `results/ablation/ablation4_all_winners.csv`.

### 13.2 Corrected final run

`Train_Test.ipynb` completed all 12 cells without an error. It tuned all 11
models from raw train rows with fold-local preprocessing, required compatible
metadata for every checkpoint, and opened the final test cells only after the
11/11 artifact gate passed. Final tuning used stratified five-fold CV with one
repeat, `N_ITER=100`, and `random_state=42`.

| Rank | Model | CV F2-macro | Test F2-macro | Test F1-weighted | High recall |
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

HistGB is the corrected final selection because it ranked first by train-only
tuning CV F2-macro. It also ranked first on test F2-macro and correctly
identified 77 of 79 `High` rows. LightGBM ranked first on test F1-weighted,
which remains the secondary metric.

### 13.3 LightGBM runtime work

The custom LightGBM implementation was optimized before its Ablation 4b run.
Multiclass trees now share binning per boosting iteration, histogram counts use
vectorized cumulative operations, and prediction traversal partitions rows by
node. A project-sized local benchmark estimated roughly 32 seconds for one
100-estimator fit before CV overhead; Kaggle runtime will vary.

### 13.4 Final interpretation and artifact status

Section 12 remains a pre-fix historical baseline. Its Random Forest deployment
conclusion is superseded by the corrected final selection of HistGB. Current
reports are under `results/final/`, and all 11 serialized checkpoints are under
`saved_models/`.

The artifacts retain revision `86a6493` for provenance. A different current
revision now produces a warning instead of forcing retraining when the model
contract metadata still matches. Changes that alter model behavior should bump
the artifact schema or require explicit deletion of affected PKLs. The current
test split has now been observed and must not be used to revise configurations
for another nominally final run.
