# COM747 – Data Science and Machine Learning  
## Group Coursework 2: Heart Disease Prediction (BRFSS 2020)

**Module:** COM747 Data Science and Machine Learning  
**Institution:** Ulster University  
**Academic Year:** 2025–2026  

---

## Project Overview

This project applies a supervised machine learning pipeline to predict
self-reported heart disease using the cleaned 2020 Behavioral Risk Factor
Surveillance System (BRFSS) dataset. Three classification models are trained
and compared: Logistic Regression, C5.0 Decision Tree, and Random Forest.
Class imbalance is addressed through both ROSE and SMOTE oversampling.
All models are evaluated on a shared held-out test set using a consistent
set of metrics including ROC-AUC, Balanced Accuracy, Sensitivity, Specificity,
F1-score, Precision, and Accuracy.

**Dataset source:** https://www.kaggle.com/datasets/youssefislamelrefaie/heart-2020-cleaned

---

## Team and File Ownership

| File | Owner | Description |
|---|---|---|
| `00_config.R` | All (shared) | Central settings: file paths, seed, CV parameters |
| `01_preprocessing.R` | Salah | Data cleaning, encoding, scaling, Boruta feature selection, ROSE/SMOTE, splits |
| `02_eda.R` | Aleemna | Exploratory data analysis, descriptive statistics, visualisations |
| `03_logistic_regression.R` | Jonnie | LR training (original/ROSE/SMOTE variants), CV, evaluation, plots |
| `04_decision_tree.R` | Yasar | C5.0 Decision Tree training, evaluation, plots |
| `05_random_forest.R` | Aleemna | Random Forest training, feature importances, evaluation, plots |
| `06_evaluation.R` | Jonnie (merge) | Combines all model metrics into comparison table and ROC overlay plot |
| `main.R` | Jonnie (merge) | Sources all scripts in order; runs the complete pipeline end to end |

---

## Folder Structure

```
project/
│
├── data/                        # All datasets (do not modify)
│   ├── heart_train.csv          # 70% training split (original)
│   ├── heart_train_rose.csv     # Training set balanced with ROSE
│   ├── heart_train_smote.csv    # Training set balanced with SMOTE
│   ├── heart_val.csv            # 15% validation split
│   ├── heart_test.csv           # 15% held-out test split (never used in training)
│   └── selected_features.csv    # Features selected by Boruta
│
├── outputs/                     # All generated plots and saved model objects
│   ├── lr_roc_curve.png
│   ├── lr_confusion_matrix.png
│   ├── lr_training.rds
│   ├── lr_results.rds
│   └── ...
│── scripts/  
|   ├──01_preprocessing.R
|   ├──02_eda.R
|   ├──03_logistic_regression.R
|   ├──04_decision_tree.R
|   ├──05_random_forest.R
|   ├──06_evaluation.R
|
|   config.R
|   main.R
└── README.md
```

---

## How to Run

### Full pipeline 
Open R or RStudio, set your working directory to the project root then run:

```r
source("main.R")
```

This will execute every script in the correct order and save all outputs
to the `outputs/` folder.

### Running a single script independently
Each modelling script (e.g. `03_logistic_regression.R`) can be run on its own for development and testing:

```r
source("3_logistic_regression.R")
```

The standalone execution guard at the bottom of each script detects whether it's being sourced by `main.R` or run directly, and behaves accordingly.

---

## Required R Packages

Install all dependencies in one go:

```r
install.packages(c(
  "tidyverse",
  "caret",
  "pROC",
  "ROSE",
  "smotefamily",
  "Boruta",
  "C50",
  "randomForest",
  "gridExtra"
))
```

---

## Data Pipeline Summary

```
Raw BRFSS CSV
    │
    ▼  1_preprocessing.R (Salah)
Outlier capping (1st–99th percentile)
Z-score standardisation
One-hot encoding (Sex, Race, Diabetic)
Ordinal encoding (AgeCategory, GenHealth)
Boruta feature selection (run on 1,000-row sample)
70 / 15 / 15 stratified split  →  train / val / test CSVs
ROSE and SMOTE applied to training set only
    │
    ▼  2_eda.R (Aleemna)
Descriptive statistics, correlation analysis, class balance plots
    │
    ├──▶  3_logistic_regression.R (Jonnie)
    │     Trains 3 LR variants (original/ROSE/SMOTE)
    │     Selects best on validation AUC
    │     Evaluates on test set
    │
    ├──▶  4_decision_tree.R (Yasar)
    │     C5.0 with cross-validation
    │     Evaluates on test set
    │
    └──▶  5_random_forest.R (Aleemna)
          Random Forest with feature importances
          Evaluates on test set
    │
    ▼  6_evaluation.R (Jonnie)
Comparison table of all models
ROC curve overlay plot
```

---

## Reproducibility

All random operations use `seed = 42` as set in `config.R`.  
This value was established by the preprocessing script and is used consistently
across all modelling scripts to ensure identical results on every run.

---

## Notes on Class Imbalance

The BRFSS dataset contains approximately 9% positive cases (HeartDisease = Yes).
Key decisions made to handle this:

- Models are optimised on **ROC-AUC**, not Accuracy, during cross-validation.
  A classifier predicting "No" for every case would achieve ~91% accuracy
  while being clinically useless.
- **Balanced Accuracy** and **Sensitivity** are reported alongside Accuracy
  so that performance on the minority class is never hidden.
- ROSE and SMOTE variants are trained and compared; the best on the
  validation set is taken forward to test set evaluation.
- ROSE/SMOTE are applied to the **training set only**. The validation and
  test sets retain the original class distribution to give a realistic
  picture of real-world performance.

---

## Known Limitations

- Boruta feature selection was run on a 1,000-row sample of the training
  set (computational constraint) with `maxRuns = 11`. Results may not
  fully reflect feature importance across the full dataset.
- The dataset is cross-sectional and self-reported, which introduces
  recall bias and prevents causal inference.
- Models are not calibrated; predicted probabilities should not be
  interpreted as clinical risk scores without further validation.
