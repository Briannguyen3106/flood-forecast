# Huong Dan Tao Slide: Introduction Va Data Analytics

## 1. Muc dich tai lieu

Tai lieu nay la script huong dan de xay dung va thuyet trinh hai phan:

1. **Introduction**: Motivation, Objective, Scope, Data.
2. **Data Analytics**: anh minh hoa tu code, nhan xet du lieu, data process va conclusion.

Noi dung phai dua tren corrected post-fix experiment. Su dung mapping muc tieu:

```text
Low = 0, Medium = 1, High = 2
```

Khong dung ket qua pre-fix trong Section 12 cua `Highlights.md` lam ket luan hien tai.

## 2. Nguyen tac trinh bay

- Moi slide chi nen truyen dat mot thong diep chinh.
- Tren slide chi de 3-5 bullet ngan; phan giai thich chi tiet nam trong loi noi.
- Moi bieu do phai co ten, nhan truc, chu thich va nguon: `Source: EDA.ipynb` hoac notebook tuong ung.
- Anh minh hoa phai la output tao tu code trong repository, khong dung anh bieu do lay tren Internet.
- Dung mau nhat quan cho ba lop: Low mau xanh, Medium mau cam, High mau do.
- Khi noi ve "best model", phai lam ro model duoc chon bang train-only CV, khong phai chon lai sau khi xem test.

## 3. Cau truc de xuat

Nen dung **8 slide**, thoi luong khoang **7-9 phut**:

| Slide | Noi dung | Thoi luong |
|---:|---|---:|
| 1 | Problem and Motivation | 50-60 giay |
| 2 | Objectives | 40-50 giay |
| 3 | Project Scope | 40-50 giay |
| 4 | Dataset Overview | 50-60 giay |
| 5 | Target and Class Distribution | 60-70 giay |
| 6 | Data Quality and Exploratory Analysis | 70-90 giay |
| 7 | Data Processing Pipeline | 70-90 giay |
| 8 | Data Analytics Conclusions | 50-60 giay |

---

## Slide 1 - Problem and Motivation

### Tieu de tren slide

**Urban Flood-Risk Prediction: Motivation**

### Noi dung tren slide

- Urban flooding threatens people, infrastructure and transportation.
- Risk depends on terrain, rainfall, soil and drainage conditions.
- Manual assessment is costly and difficult to scale.
- Machine learning can support early screening of high-risk areas.

### Minh hoa

Su dung mot anh nho ve ngap do thi de tao boi canh. Neu slide bat buoc chi dung anh tu code, thay anh bang so do don gian:

```text
Terrain + Rainfall + Soil + Drainage -> Flood-risk class
```

### Loi thuyet trinh

> Ngap do thi gay anh huong truc tiep den con nguoi, giao thong va ha tang. Muc do rui ro khong phu thuoc vao mot yeu to don le ma la su ket hop giua do cao dia hinh, cuong do mua, dac tinh dat va kha nang thoat nuoc. Danh gia thu cong tung khu vuc ton nhieu thoi gian va kho mo rong. Vi vay, du an nghien cuu kha nang dung machine learning de ho tro sang loc va nhan dien cac khu vuc co rui ro ngap, dac biet la nhom High-risk can duoc uu tien canh bao.

### Thong diep chinh

Machine learning duoc dung de **ho tro uu tien canh bao**, khong thay the hoan toan danh gia cua chuyen gia.

---

## Slide 2 - Objectives

### Tieu de tren slide

**Project Objectives**

### Noi dung tren slide

- Predict three ordered risk levels: Low, Medium and High.
- Compare tree, boosting, ordinal regression and kernel models.
- Maximize F2-macro and monitor High-risk recall.
- Build a leakage-safe and reproducible experiment workflow.

### Loi thuyet trinh

> Muc tieu dau tien la chuyen bai toan thanh phan loai ba muc rui ro co thu tu Low, Medium va High. Du an so sanh 11 mo hinh thuoc nhieu nhom, tu Decision Tree, Random Forest va boosting den ordinal regression va SVM. Do bo sot mot khu vuc nguy hiem nghiem trong hon mot canh bao du, F2-macro duoc chon lam metric chinh vi metric nay uu tien recall va danh trong so ngang nhau cho ba lop. Ngoai hieu nang, du an con dat muc tieu xay dung quy trinh tranh data leakage va co the tai lap voi random state bang 42.

### Ghi chu

Khong viet Accuracy la metric chinh. Dataset mat can bang nen Accuracy co the bi chi phoi boi lop Low.

---

## Slide 3 - Project Scope

### Tieu de tren slide

**Scope of the Study**

### Noi dung tren slide

**Included**

- Multiclass prediction for urban spatial segments.
- Data cleaning, EDA, feature engineering and model comparison.
- Train-only cross-validation and final holdout evaluation.

**Not included**

- Real-time rainfall forecasting.
- Flood-depth or inundation-map simulation.
- Production deployment and live warning decisions.

### Loi thuyet trinh

> Pham vi cua du an la du doan muc rui ro cho tung spatial segment dua tren du lieu tabular hien co. Quy trinh bao gom phan tich du lieu, tien xu ly, feature engineering, ablation, tuning va danh gia cuoi. Du an khong du bao luong mua theo thoi gian thuc, khong mo phong do sau ngap va khong phai he thong canh bao production. Gioi han pham vi nay giup ket qua duoc dien giai dung: day la mo hinh sang loc rui ro tren dataset nghien cuu, chua phai cong cu ra quyet dinh van hanh doc lap.

---

## Slide 4 - Dataset Overview

### Tieu de tren slide

**Dataset Overview**

### Noi dung tren slide

| Thuoc tinh | Gia tri |
|---|---|
| So dong | 2,963 spatial segments |
| Bai toan | 3-class classification |
| Raw features | 5 numeric + 5 categorical |
| Train/Test | 2,370 / 593, stratified 80/20 |
| Random seed | 42 |

Nhom thong tin chinh:

- Terrain: `elevation_m`.
- Drainage: density, proximity and drain type.
- Rainfall: historical intensity and return period.
- Context: soil, land use and data sources.

### Minh hoa

Tao mot so do feature groups bang PowerPoint shapes, khong can chen toan bo ten cot. Co the dat bon khoi xung quanh khoi trung tam `Flood Risk`.

### Loi thuyet trinh

> Dataset gom 2,963 spatial segments. Sau stratified split, train co 2,370 dong va test co 593 dong. Moi dong mo ta mot khu vuc bang nam bien so va nam bien phan loai. Cac feature dai dien cho bon nhom yeu to: dia hinh, ha tang thoat nuoc, mua va boi canh dat-su dung dat. Test set duoc giu lai cho danh gia cuoi, khong duoc dung de chon preprocessing, feature, class weight hay hyperparameter.

### Nguon

`data/raw/urban_pluvial_flood_risk_dataset.xlsx` va `experiments/EDA.ipynb`.

---

## Slide 5 - Target and Class Distribution

### Tieu de tren slide

**Target Construction and Class Imbalance**

### Noi dung tren slide

Priority rules:

```text
High   : ponding_hotspot or extreme_rain_history
Medium : low_lying or sparse_drainage, without a High label
Low    : monitoring, event date, empty or no stronger indicator
```

Class counts:

- Low: 1,994 rows, 67.3%.
- Medium: 574 rows, 19.4%.
- High: 395 rows, 13.3%.

### Anh bat buoc tu code

Chen:

```text
experiments/figures/class_distribution.png
```

Anh nay duoc tao tu `experiments/EDA.ipynb`.

### Loi thuyet trinh

> Cot risk_labels ban dau chua nhieu label trong cung mot chuoi, vi vay du an chuyen no thanh mot target duy nhat theo priority rules. Neu co bang chung ponding hotspot hoac lich su mua cuc doan thi segment duoc xep High. Neu khong thuoc High nhung co dia hinh thap hoac thoat nuoc thua thi xep Medium. Cac truong hop con lai la Low. Sau chuyen doi, Low chiem 67.3 phan tram, trong khi High chi chiem 13.3 phan tram. Su mat can bang nay la ly do khong dung Accuracy lam metric chinh va phai danh gia recall rieng cho lop High.

### Nhan xet tren slide

Dat mot callout canh bieu do:

> Low is about five times larger than High; class-aware evaluation is required.

---

## Slide 6 - Data Quality and Exploratory Analysis

Nen tach thanh hai cot va chi dung toi da hai bieu do de slide khong bi roi.

### Tieu de tren slide

**Data Quality and Key Patterns**

### Anh tu code

Lua chon 2 trong 4 anh sau:

1. `experiments/figures/missing_values.png`
2. `experiments/figures/numeric_distribution.png`
3. `experiments/figures/boxplot_by_class.png`
4. `experiments/figures/correlation_matrix.png`

Lua chon khuyen nghi:

- Ben trai: `missing_values.png`.
- Ben phai: `boxplot_by_class.png`.

Neu co them mot slide EDA, dung `numeric_distribution.png` va `correlation_matrix.png` tren slide rieng.

### Nhan xet du lieu can hien thi

- Missing values occur in several numeric and categorical columns.
- Numeric features are skewed and contain valid extreme values.
- Only `elevation_m = -3.0` is treated as a missing sentinel.
- Elevation and historical rainfall show strong class separation.
- Extreme rainfall values are useful High-risk signals and must not be clipped.

### Loi thuyet trinh

> EDA cho thay du lieu co missing o ca bien so va bien phan loai. Rieng elevation co gia tri sentinel bang am 3, day la ma danh dau du lieu loi chu khong phai do cao that. Cac gia tri am khac van co the hop le va khong duoc xoa may moc. Nhieu bien so co phan bo lech phai, dac biet la elevation, rainfall intensity va return period. Boxplot theo lop cho thay elevation thap va rainfall intensity cao co kha nang phan biet rui ro ro rang. Cac gia tri rainfall cuc lon tap trung o lop High, vi vay day la signal quan trong, khong phai outlier can loai bo. Chi storm-drain proximity duoc clip vi cac gia tri cuc lon cua bien nay khong cho thay signal phan lop ro rang.

### Nhan xet chi tiet de tra loi cau hoi

- `elevation_m` co quan he nghich voi risk: do cao thap thuong gan voi Medium/High.
- `historical_rainfall_intensity_mm_hr` co quan he duong voi risk va la signal manh cho High.
- `return_period_years` co cac muc 2, 5, 10, 25, 50 va 100 nam; day la gia tri thiet ke hop le.
- Missing chu yeu lien quan den qua trinh thu thap du lieu, nhung tai lieu khong nen khang dinh chac chan MCAR neu chua co statistical test.
- Categorical features nhin chung co predictive power thap hon numeric features; `soil_group` co thu tu thuy van co y nghia.

---

## Slide 7 - Data Processing Pipeline

### Tieu de tren slide

**Leakage-Safe Data Processing**

### So do tren slide

```text
Raw data
   |
   v
Target construction: Low=0, Medium=1, High=2
   |
   v
Stratified train/test split (80/20)
   |
   v
Fold-local preprocessing during CV
   |
   v
Shared steps for both pipelines
sentinel handling -> optional features -> impute -> clip proximity
   |
   +--> Pipeline A: tree models
   |    compact ordinal encoding; no scaling
   |
   +--> Pipeline B: linear/SVM models
        optional Yeo-Johnson -> encode -> optional RobustScaler
   |
   v
Model-specific imbalance handling and training
```

### Nhan xet tren slide

- Imputer, clipping threshold, encoders and scalers are fitted on each training fold.
- SMOTE is applied only to fold-training rows.
- Validation and test distributions remain unchanged.
- Tree and linear models use different preprocessing configurations.

### Loi thuyet trinh

> Data process bat dau bang viec tao target theo dung thu tu Low bang 0, Medium bang 1 va High bang 2. Sau do du lieu duoc stratified split 80/20. Diem quan trong nhat la moi preprocessor duoc fit lai ben trong tung training fold. Ca Pipeline A va Pipeline B deu ke thua cac buoc chung gom xu ly sentinel, feature engineering tuy chon, imputation va clipping storm-drain proximity. Sau do Pipeline A dung bieu dien categorical gon cho tree models, con Pipeline B co them cac lua chon Yeo-Johnson, encoding va RobustScaler cho linear models va SVM. Median, clipping threshold, feature threshold, category vocabulary, Yeo-Johnson parameters va scaler khong duoc hoc tu validation hoac test. Neu mot mo hinh dung SMOTE, SMOTE chi duoc ap dung tren training rows cua fold. Cach thiet ke nay ngan validation data anh huong den qua trinh hoc va giup CV score dang tin cay hon.

### Goi y minh hoa tu code

Co the chen mot anh so sanh raw va processed data tu:

```text
experiments/figures/compare_data/
```

Nguon tao anh: `experiments/Compare_preprocessed_data.ipynb`.

Neu dung anh so sanh, chi chon mot anh the hien ro tac dong cua scaling hoac transform; khong dua tat ca anh vao cung slide.

---

## Slide 8 - Data Analytics Conclusions

### Tieu de tren slide

**Conclusions from Data Analytics**

### Noi dung tren slide

1. The target is imbalanced, so macro and class-level metrics are necessary.
2. Elevation and rainfall provide the clearest predictive signals.
3. Valid extremes should be preserved; only justified anomalies are corrected.
4. Preprocessing must be model-specific and fitted inside each CV fold.
5. The data suggests nonlinear thresholds and interactions, favoring tree models.

### Loi thuyet trinh

> Tu phan tich du lieu, nhom rut ra nam ket luan. Thu nhat, target mat can bang nen ket qua phai duoc danh gia bang macro metric va recall theo lop. Thu hai, elevation va rainfall la hai nhom signal ro nhat. Thu ba, khong phai moi gia tri cuc doan deu la du lieu xau; rainfall rat cao va elevation cao co the mang thong tin phan lop quan trong. Thu tu, preprocessing phai khac nhau theo model va phai fit ben trong moi CV fold de tranh leakage. Cuoi cung, cac threshold va interaction quan sat duoc trong EDA cho thay bai toan co tinh phi tuyen, tao co so de ky vong tree-based models phu hop hon linear models. Day la ket luan tu data analytics; ket qua model cu the se duoc trinh bay o phan experiment.

### Cau chuyen tiep phan

> Based on these findings, the next section evaluates which preprocessing, imbalance strategy and model family best capture these nonlinear patterns.

---

## 4. Anh minh hoa va nguon code

| Muc dich | Anh de xuat | Notebook/code nguon |
|---|---|---|
| Phan bo target | `experiments/figures/class_distribution.png` | `experiments/EDA.ipynb` |
| Tan suat raw labels | `experiments/figures/label_frequency.png` | `experiments/EDA.ipynb` |
| Missing values | `experiments/figures/missing_values.png` | `experiments/EDA.ipynb` |
| Phan bo numeric | `experiments/figures/numeric_distribution.png` | `experiments/EDA.ipynb` |
| Numeric theo class | `experiments/figures/boxplot_by_class.png` | `experiments/EDA.ipynb` |
| Correlation | `experiments/figures/correlation_matrix.png` | `experiments/EDA.ipynb` |
| Categorical theo class | `experiments/figures/categorical_vs_class.png` | `experiments/EDA.ipynb` |
| Raw va processed | `experiments/figures/compare_data/` | `experiments/Compare_preprocessed_data.ipynb` |

Khong dung `results/final/model_comparison.png` trong phan Data Analytics vi day la ket qua model, phu hop hon voi phan Experiment Results.

## 5. Code mau de xuat anh cho slide

Neu can tao lai bieu do voi font lon va do phan giai phu hop slide, them mot cell vao `experiments/EDA.ipynb` sau khi dataframe da co `risk_class`:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

SLIDE_DIR = Path("figures/slide")
SLIDE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["Low", "Medium", "High"]
CLASS_COLORS = {
    "Low": "#2E8B57",
    "Medium": "#F4A261",
    "High": "#D62828",
}

sns.set_theme(style="whitegrid", context="talk")

counts = df["risk_class"].value_counts().reindex(CLASS_ORDER)
ax = counts.plot(
    kind="bar",
    color=[CLASS_COLORS[label] for label in CLASS_ORDER],
    figsize=(9, 5),
)
ax.set_title("Flood-Risk Class Distribution")
ax.set_xlabel("Risk class")
ax.set_ylabel("Number of spatial segments")
ax.tick_params(axis="x", rotation=0)

for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.tight_layout()
plt.savefig(SLIDE_DIR / "class_distribution_slide.png", dpi=220)
plt.show()
```

Code mau cho missing-value chart:

```python
missing_pct = df.isna().mean().mul(100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

plt.figure(figsize=(10, 5))
ax = sns.barplot(
    x=missing_pct.values,
    y=missing_pct.index,
    color="#457B9D",
)
ax.set_title("Missing Values by Feature")
ax.set_xlabel("Missing values (%)")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(SLIDE_DIR / "missing_values_slide.png", dpi=220)
plt.show()
```

Luu y: bieu do phai duoc tao tu raw data hoac dataframe EDA da tao target, khong duoc fit preprocessing tren toan bo data roi dung ket qua do de bao cao CV performance.

## 6. Checklist truoc khi nop slide

- [ ] Dung mapping `Low=0, Medium=1, High=2`.
- [ ] Dung tong so 2,963 dong; train 2,370; test 593.
- [ ] Ghi ro test set chi dung cho final evaluation.
- [ ] Co it nhat 2 anh EDA la output tu code.
- [ ] Moi anh co caption va notebook nguon.
- [ ] Khong goi final tuning 5-fold, 1 repeat la repeated CV.
- [ ] Khong dua Random Forest pre-fix lam model ket luan hien tai.
- [ ] Khong dua model result vao phan Data Analytics neu nhom co phan Results rieng.
- [ ] Slide it chu, loi noi giai thich chi tiet hon noi dung hien thi.
- [ ] Ket luan Data Analytics dan tu nhien sang phan model/experiment.

## 7. Phien ban rut gon neu chi duoc trinh bay 5 slide

Neu bi gioi han so slide, gop noi dung nhu sau:

1. Motivation and Objectives.
2. Scope and Dataset Overview.
3. Target Construction and Class Distribution.
4. Data Quality and Key EDA Findings.
5. Data Processing and Analytics Conclusions.

Van phai giu hai anh quan trong nhat:

- `class_distribution.png` de trinh bay imbalance.
- `boxplot_by_class.png` hoac `missing_values.png` de trinh bay insight tu data.
