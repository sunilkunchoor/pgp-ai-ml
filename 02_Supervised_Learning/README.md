# 🤖 Supervised Learning

> **Module Completed:** July 2021

---

## 🎯 Module Overview

Supervised learning is the backbone of most applied ML — the model learns a mapping from input features to an output label using labelled training data. This module covers the core classification and regression algorithms, model evaluation metrics, and the end-to-end workflow of training, testing and tuning a model.

---

## 📖 Topics Covered

### Introduction to Machine Learning
Overview of the ML workflow: problem framing, data collection, feature engineering, model selection, training, evaluation and deployment. Distinction between supervised, unsupervised and reinforcement learning.

### Linear Regression
Fitting a line (or hyperplane) to predict a continuous output. Covers the OLS (Ordinary Least Squares) cost function, gradient descent, assumptions (linearity, independence, homoscedasticity, normality of errors), R², adjusted R², residual analysis, and multicollinearity detection via VIF.

### Logistic Regression
Extension of linear regression for binary (and multi-class) classification. Applies the sigmoid function to output probabilities, uses log-loss (cross-entropy) as the cost function. Covers decision boundary, odds ratio, regularisation (L1/L2), solver options (`liblinear`, `lbfgs`, `saga`), threshold tuning via `predict_proba`, and the confusion matrix.

### K-Nearest Neighbours (KNN)
Non-parametric, instance-based learner — classifies a point by majority vote among its K nearest neighbours. Covers:
- Distance metrics: Euclidean, Manhattan, Minkowski
- Choosing optimal K (error rate vs K plot, cross-validation)
- Effect of scaling — KNN is distance-sensitive so normalisation is mandatory
- Algorithms: BallTree, KDTree, Brute-force
- `leaf_size` and `p` (Minkowski power) parameters
- GridSearchCV for full hyperparameter tuning

### Naive Bayes
Probabilistic classifier based on Bayes theorem with a "naive" conditional independence assumption between features. Covers:
- Gaussian Naive Bayes — for continuous features (assumes Normal distribution per class)
- Multinomial Naive Bayes — for count/frequency data (e.g., text)
- Bernoulli Naive Bayes — for binary features
- Laplace smoothing to handle zero-probability problem
- Works well on text and high-dimensional data despite simplicity

### Support Vector Machine (SVM)
Finds the maximum-margin hyperplane separating classes. Covers:
- Hard vs soft margin (C parameter — controls regularisation/penalty for misclassification)
- Kernel trick: Linear, Polynomial, RBF (Radial Basis Function), Sigmoid
- Gamma (γ) — controls influence radius of a single training example; high γ = tight fit
- C vs γ trade-off — key to tuning SVM
- Support vectors and the dual optimisation problem
- SVM for multi-class: One-vs-One (OvO) and One-vs-Rest (OvR)

### Model Evaluation Metrics
- **Confusion Matrix** — TP, TN, FP, FN
- **Accuracy** — overall correct predictions (misleading for imbalanced data)
- **Precision** — of all predicted positives, how many are actually positive
- **Recall (Sensitivity)** — of all actual positives, how many were caught
- **F1-Score** — harmonic mean of precision and recall; preferred when classes are imbalanced
- **AUC-ROC** — area under the ROC curve; model's ability to discriminate between classes
- **Balanced Accuracy** — average recall per class; robust to class imbalance
- Classification report interpretation (macro avg, weighted avg, support)

### Handling Class Imbalance
- **Random Under-Sampling** — removes majority class samples (risk: losing information)
- **Random Over-Sampling** — duplicates minority class samples
- **SMOTE** (Synthetic Minority Oversampling Technique) — generates synthetic minority samples by interpolating between existing ones
- Using `balanced_accuracy_score` instead of `accuracy_score` for imbalanced targets

### Train-Test Split & Cross-Validation
Stratified train/test split to preserve class proportions. K-fold cross-validation to get reliable performance estimates. `GridSearchCV` combines cross-validation with hyperparameter search.

---

## 🏆 Project — Supervised Learning

**Submitted:** 04 July 2021

---

### Part 1 — Orthopaedic Patient Classification (KNN)

**Business Problem:** A medical research group wants to classify patients into one of three orthopaedic conditions — `Normal`, `Type_S` (Spondylolisthesis), `Type_H` (Hernia) — based on biomechanical measurements.

**Datasets:** Three CSVs merged — `Part1 - Normal.csv`, `Part1 - Type_H.csv`, `Part1 - Type_S.csv`

**Features:** 6 numerical biomechanical measurements — `P_incidence`, `P_tilt`, `L_angle`, `S_slope`, `P_radius`, `S_degree`

**Target:** `Class` — 3 categories (Normal, Type_H, Type_S)

**Approach:**
1. **EDA** — `P_tilt` and `P_radius` symmetrically distributed; `S_degree` right-skewed; all columns contain outliers (retained — KNN is robust to outliers via distance weighting)
2. **Correlation** — moderate correlations between `P_incidence`/`S_degree` (0.64) and `S_slope`/`L_angle` (0.60) — no strong multicollinearity requiring column removal
3. **Preprocessing** — MinMaxScaler normalisation (mandatory for KNN); Label Encoding of target; identified class imbalance (`Type_S` dominant) — used `balanced_accuracy_score`
4. **Model Tuning** — tested multiple K values via error rate plot; different algorithms (BallTree, KDTree); Minkowski `p` values; GridSearchCV → best K = 9, best balanced accuracy = 0.756

**Outcome:** Final model — KNN with K=19, BallTree/KDTree algorithm, default Minkowski metric

**Key challenge:** Small dataset size and class imbalance limited model performance; SMOTE not applied in Part 1

---

### Part 2 — Bank Customer Loan Conversion Prediction (Logistic Regression + Naive Bayes)

**Business Problem:** A bank wants to identify potential customers who are likely to take a loan on their credit card — targeted marketing to increase conversion rate.

**Dataset:** `bank_x.xlsx` + `Part2 - Data1.csv`, `Part2 - Data2.csv` — merged customer data

**Features:** `Age`, `CustomerSince`, `Mortgage`, `MonthlyAverageSpend`, `HighestSpend`, `InternetBanking`, `CreditCard`, `LoanOnCard`

**Target:** `LoanOnCard` (binary — customer has loan on credit card or not)

**Approach:**
1. **EDA** — avg customer age 45; outliers in `Mortgage` and `MonthlyAverageSpend`; high correlation between `Age` and `CustomerSince` → dropped one
2. **Target Engineering** — `LoanOnCard` chosen as the conversion target; `CreditCard` column dropped (337 records showed LoanOnCard=True with CreditCard=False — data inconsistency)
3. **Class Imbalance** — SMOTE applied to balance the training data
4. **Scaling** — StandardScaler on numerical columns
5. **Models trained:** Logistic Regression (solvers: `liblinear`, `lbfgs`, `saga`) and Gaussian Naive Bayes
6. **Tuning** — tested different solvers; threshold tuning via `predict_proba` — no improvement found

**Outcome:** **Logistic Regression** selected as final model — marginally better accuracy, recall and F1-score than Naive Bayes

| Metric | Logistic Regression | Naive Bayes |
|--------|--------------------|----|
| Accuracy | Higher | Lower |
| Recall | Better | Lower |
| F1-Score | Better | Lower |

**Key challenge:** Target column was not explicitly labelled — required domain reasoning to identify; data inconsistencies in credit card column required careful handling

---

## 🔑 Key Learnings

1. **KNN requires scaling** — it is purely distance-based; unscaled features dominate distance calculations
2. **Imbalanced data ≠ use accuracy** — always check class distribution before choosing evaluation metric; prefer balanced accuracy, F1, or AUC-ROC
3. **SMOTE > random oversampling** — generates diverse synthetic samples rather than duplicates; applied only to training data, never to test data
4. **Naive Bayes is fast but fragile** — independence assumption often violated in real data, but still performs surprisingly well for text/high-dimensional data
5. **Logistic Regression is interpretable** — coefficients directly show feature influence on log-odds; useful for explaining decisions to stakeholders
6. **GridSearchCV is expensive** — for small datasets, exhaustive search is feasible; for large datasets, use RandomizedSearchCV or Bayesian optimisation
7. **Data quality issues matter** — unclear target columns, unit-less binary values, and logical inconsistencies (loan without credit card) are common in real data
8. **Correlation ≠ drop** — only drop a feature if correlation > 0.8–0.9 AND the features measure the same concept

---

## 🛠️ Tools & Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
```
