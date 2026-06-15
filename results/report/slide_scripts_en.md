# English Presentation Scripts: Introduction and Data Analytics

Estimated speaking time: 3-5 minutes.

## Slide 1 - Urban Flood-Risk Prediction: Motivation

Urban flooding threatens people, transportation, and infrastructure. Its risk depends on interacting factors such as terrain, rainfall, soil, and drainage. Since manual assessment is difficult to scale, this project uses machine learning to support the early screening and prioritization of flood-prone areas, especially High-risk locations.

## Slide 2 - Project Objectives

Our objective is to classify urban segments into three ordered risk levels: Low, Medium, and High. We compare eleven models from tree, boosting, ordinal regression, and kernel-based families. Because missing a hazardous area is more costly than issuing an extra warning, macro F2-score is our primary metric, with particular attention to High-risk recall. We also aim to make the workflow reproducible and free from data leakage.

## Slide 3 - Scope of the Study

The project covers exploratory analysis, data cleaning, feature engineering, model comparison, tuning, and final evaluation using tabular data. It does not forecast rainfall in real time, simulate flood depth, or provide a production warning system. Therefore, the model should be viewed as a risk-screening tool rather than a replacement for expert decisions.

## Slide 4 - Dataset Overview

The dataset contains 2,963 urban spatial segments, with five numerical and five categorical features. These features describe terrain, drainage, rainfall, soil, and land-use conditions. We use a stratified 80/20 split, giving 2,370 training rows and 593 test rows. The test set is reserved exclusively for final evaluation and is not used for preprocessing or model selection.

## Slide 5 - Target Construction and Class Imbalance

The original labels are converted into one target using priority rules. Ponding hotspots or extreme rainfall history indicate High risk. Low-lying terrain or sparse drainage indicates Medium risk when no High-risk condition is present. The remaining cases are Low risk. We use the ordinal mapping Low equals zero, Medium equals one, and High equals two. The classes are imbalanced: Low represents 67.3 percent, while High represents only 13.3 percent. This makes class-aware evaluation essential.

## Slide 6 - Data Quality and Key Patterns

The data contains missing values in both numerical and categorical features. An elevation value of negative three is treated as a missing-data sentinel, while other negative elevations may be valid. The analysis shows that lower elevation and higher historical rainfall are strong risk indicators. Extreme rainfall values are useful High-risk signals, so they should not be removed automatically as outliers.

## Slide 7 - Leakage-Safe Data Processing

After target construction, we split the data before selecting any preprocessing strategy. During cross-validation, imputation, clipping, feature engineering, encoding, transformation, and scaling are fitted only on each training fold. Tree models use Pipeline A without scaling, while linear models and SVMs use Pipeline B with optional transformation and robust scaling. If SMOTE is used, it is applied only to fold-training rows, never to validation or test data.

## Slide 8 - Conclusions from Data Analytics

In summary, the target is imbalanced, and elevation and rainfall provide the clearest predictive signals. Valid extreme values must be preserved, and preprocessing must be model-specific and leakage-safe. The patterns also suggest nonlinear thresholds and feature interactions, providing a strong reason to evaluate tree-based models. The next section examines which model and strategy best capture these patterns.
