# 🌲 Ensemble Techniques

> **Module Completed:** July 2021

---

## 🎯 Module Overview

Ensemble techniques combine multiple weak learners to build a stronger, more robust model. Rather than relying on a single algorithm, ensembles reduce variance (bagging) or bias (boosting) by aggregating predictions across many models. This module covers Decision Trees as the base learner, followed by all major ensemble strategies.

---

## 📖 Topics Covered

### Decision Trees (CART)
The foundational building block of ensemble methods. A tree splits data recursively using a feature threshold that maximises information gain (entropy-based) or minimises Gini impurity. Covers:
- Splitting criteria: Gini impurity vs Information Gain (entropy)
- Tree depth, min samples split, min samples leaf — controls overfitting
- Pruning: pre-pruning (max depth) and post-pruning (cost-complexity)
- Decision Trees for both classification and regression (CART)
- Visualising a decision tree with `sklearn.tree.plot_tree`

### Bias-Variance Trade-off
The core tension in ML: a high-bias model underfits (too simple), a high-variance model overfits (too complex). Ensemble methods address this:
- **Bagging** reduces variance without increasing bias
- **Boosting** reduces bias (and some variance) sequentially
- The ideal model minimises total error = Bias² + Variance + Irreducible noise

### Bagging (Bootstrap Aggregation)
Trains multiple independent models on different random subsets of the training data (sampled with replacement). Predictions are aggregated by majority vote (classification) or averaging (regression). Key idea: decorrelating models reduces overall variance.

### Random Forest
Extension of bagging — each tree is also trained on a random subset of features at every split (feature randomness). This further decorrelates trees and produces a more robust ensemble. Covers:
- `n_estimators` — number of trees (more = more stable, diminishing returns)
- `max_features` — number of features considered per split (`sqrt(n)` for classification)
- `max_depth`, `min_samples_split`, `min_samples_leaf` — tree complexity control
- Out-of-Bag (OOB) error as a built-in validation estimate
- Feature importance via mean decrease in impurity

### AdaBoost (Adaptive Boosting)
Sequential ensemble: each new model focuses on the misclassified samples from the previous model by increasing their weights. Final prediction is a weighted vote of all weak learners. Covers:
- `n_estimators` and `learning_rate` (shrinkage)
- Sensitive to noisy data and outliers (up-weights misclassified points)
- Weak learner is typically a Decision Stump (1-level tree)

### Gradient Boosting Classifier (GBC)
Generalisation of boosting — fits each new tree to the **residuals (pseudo-gradients)** of the previous model using gradient descent in function space. Covers:
- `n_estimators`, `learning_rate`, `max_depth`, `subsample`
- Subsample < 1.0 introduces stochastic gradient boosting (reduces overfitting)
- Typically slower to train than Random Forest but often more accurate
- `sklearn.ensemble.GradientBoostingClassifier`

### XGBoost (Extreme Gradient Boosting)
Highly optimised, regularised implementation of gradient boosting. Covers:
- Built-in L1 (`reg_alpha`) and L2 (`reg_lambda`) regularisation
- `scale_pos_weight` — handles class imbalance by weighting minority class
- Native handling of missing values (NaN) — no imputation needed
- Tree pruning with `max_depth` and `min_child_weight`
- `early_stopping_rounds` to avoid overfitting
- K-fold cross-validation using `cv_results_` from `GridSearchCV`

### CatBoost
Gradient boosting library by Yandex designed to handle categorical features natively without explicit encoding. Covers:
- Symmetric trees (oblivious trees) — faster inference
- Target encoding built-in for categorical columns
- `depth`, `learning_rate`, `iterations` as key parameters

### Model Comparison & Selection
After training all models, compare using:
- Train vs Test balanced accuracy (to detect overfitting)
- ROC AUC score
- Overfitting check: large gap between train and test accuracy = overfitting

**Observed ranking (this project):**
- XGBoost — best training accuracy but **overfit** (large train–test gap)
- CatBoost — underfit on training but decent test score
- Gradient Boosting — best balance; chosen as final model
- AdaBoost — decent, slight overfitting
- Random Forest — lowest balanced accuracy among the ensemble methods

### Handling Class Imbalance (Advanced)
Beyond SMOTE, this module covers:
- **ADASYN** (Adaptive Synthetic Sampling) — generates more synthetic samples near the decision boundary (harder-to-classify minority points get more neighbours)
- **SMOTETomek** — combines SMOTE oversampling + Tomek link cleaning (removes ambiguous borderline majority samples)
- **SMOTEENN** — SMOTE + Edited Nearest Neighbours cleaning

### Pickling & Deployment
Saving the trained model to disk using `pickle` (`.pkl`) for future inference without retraining. Integration with a web app (Flask) for a GUI-based prediction interface.

---

## 🏆 Project — Ensemble Techniques (Customer Churn Prediction)

**Submitted:** 25 July 2021
**Business Problem:** A telecom company wants to predict which customers are likely to churn (leave) so they can launch focused retention programmes.

---

### Dataset
**Source:** Two CSV files merged — `TelcomCustomer-Churn_1.csv` + `TelcomCustomer-Churn_2.csv`

**Features (21 columns):**
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account:** Tenure, Contract, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Target:** `Churn` (Yes / No)

---

### Data Cleaning
- `TotalCharges` had NaN values — dropped rows (mean ≠ median, so imputation would skew distribution)
- `SeniorCitizen` stored as 0/1 integer — kept numeric for modelling, converted to object for EDA only
- `CustomerID` dropped (no predictive value)
- All categorical columns manually encoded to numeric (binary: 0/1; multi-class: ordinal mapping)
- Automated preprocessing pipeline function built for reuse with the XGBoost NaN-native run

### EDA Key Findings
- Target imbalance: ~73% No-churn, ~27% Churn
- Higher monthly charges → higher churn rate
- Short tenure → more likely to churn (avg tenure: churned=15.5, retained=35.5)
- Month-to-month contract customers churn significantly more than annual/2-year contract customers
- No outliers in numerical columns (tenure, monthly charges, total charges)
- `TotalCharges` and `Tenure` are highly correlated (retained as ensemble handles collinearity)

### Preprocessing
- **Class imbalance fix:** ADASYN oversampling applied to training data only
- **Train-test split:** 70:30 stratified
- No scaling required — tree-based methods are scale-invariant

### Models Trained

| Model | Overfitting? | Notes |
|-------|-------------|-------|
| Random Forest | Moderate | Lower balanced accuracy than boosting |
| AdaBoost | Slight | Sensitive to noise |
| Gradient Boosting | Minimal | Best balance of train/test performance |
| CatBoost | Underfitting on train | Better generalisation |
| XGBoost | Yes (overfit) | Best train accuracy; poor generalisation |

**Hyperparameter tuning:** GridSearchCV applied to all 5 models; `cv_results_` used for XGBoost analysis

### Final Model: Gradient Boosting Classifier
Selected because:
- Minimal overfitting (train ≈ test accuracy)
- Best ROC AUC score among non-overfit models
- Stable across cross-validation folds
- Model serialised using `pickle` → `gbc_model.pkl`

### GUI Deployment
- Flask web application built for interactive churn prediction
- Deployed on Heroku: `https://ensemble-gui.herokuapp.com/`
- Source code: `https://github.com/sunilkunchoor/project-ensemble`

---

## 🔑 Key Learnings

1. **Bagging reduces variance; boosting reduces bias** — choose based on whether your base model overfits or underfits
2. **Random Forest ≠ always best** — boosting methods (GBM, XGBoost) often outperform Random Forest on structured tabular data
3. **XGBoost overfits easily** — always use `early_stopping_rounds` or regularisation (`reg_alpha`, `reg_lambda`) when tuning
4. **ADASYN > SMOTE for imbalanced data near decision boundaries** — it focuses synthetic generation where the classifier struggles most
5. **Tree-based methods don't need scaling** — skip StandardScaler/MinMaxScaler; they're irrelevant for split-based algorithms
6. **High correlation between features is OK for ensembles** — trees find the best split regardless; collinearity only matters for linear models
7. **Pickle your final model** — always serialise the fitted model for deployment; retraining in production is expensive
8. **Train accuracy >> Test accuracy = overfitting** — a large gap is a warning sign; reduce model complexity or add regularisation

---

## 🛠️ Tools & Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE, SMOTETomek
```
