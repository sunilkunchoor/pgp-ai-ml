# 🎓 PGP in Artificial Intelligence & Machine Learning

> **Program:** Post Graduate Program in AI & ML
> **Institution:** Great Learning (in collaboration with The University of Texas at Austin — McCombs School of Business)
> **Duration:** ~12 months (2021 – 2022)
> **Portfolio:** [eportfolio.mygreatlearning.com/sunil-kunchoor-basavaraju](https://eportfolio.mygreatlearning.com/sunil-kunchoor-basavaraju)

---

## 📖 About the Program

The **Post Graduate Program in AI & ML** is an intensive, industry-focused program delivered by Great Learning in collaboration with **The University of Texas at Austin**. It covers the full spectrum of modern Artificial Intelligence and Machine Learning — from statistical foundations through to cutting-edge deep learning, computer vision, and natural language processing.

The program combines:
- **Live online sessions** with industry experts and faculty
- **Hands-on projects** using real-world datasets
- **Mentorship** from AI/ML practitioners
- A **Capstone Project** that integrates learnings across all modules

---

## 🗺️ Program Structure

| # | Module | Key Focus |
|---|--------|-----------|
| 01 | [Applied Statistics](#01-applied-statistics) | Probability, Hypothesis Testing, Statistical Inference |
| 02 | [Supervised Learning](#02-supervised-learning) | Regression, Classification, Model Evaluation |
| 03 | [Ensemble Techniques](#03-ensemble-techniques) | Bagging, Boosting, Random Forests, XGBoost |
| 04 | [Unsupervised Learning](#04-unsupervised-learning) | Clustering, Dimensionality Reduction |
| 05 | [Feature Engineering & Model Tuning](#05-feature-engineering--model-tuning) | Feature Selection, Hyperparameter Tuning |
| 06 | [Recommendation Systems](#06-recommendation-systems) | Collaborative Filtering, Content-Based, Hybrid |
| 07 | [Neural Networks & Deep Learning](#07-neural-networks--deep-learning) | ANN, CNN basics, RNN basics |
| 08 | [CNN Architecture & Transfer Learning](#08-cnn-architecture--transfer-learning) | VGG, ResNet, InceptionV3, Fine-Tuning |
| 09 | [Advanced Computer Vision](#09-advanced-computer-vision---object-detection) | YOLO, R-CNN, Segmentation |
| 10 | [Statistical NLP](#10-statistical-natural-language-processing) | BoW, TF-IDF, Word2Vec, LDA |
| 11 | [Sequential NLP](#11-sequential-natural-language-processing) | LSTM, Transformers, BERT |
| 12 | [Capstone Project](#12-capstone-project) | End-to-end AI/ML solution |

---

### 01. Applied Statistics
📁 [`01_Applied_Statistics/`](./01_Applied_Statistics/)
📅 **Completed:** May 2021

**Topics Covered:**
- **Descriptive Statistics** — summarising datasets using mean, median, mode, variance, SD, IQR, skewness and kurtosis; visualised via histograms and box plots
- **Probability Theory** — joint, marginal and conditional probability; Bayes theorem
- **Probability Distributions** — Binomial (discrete, fixed trials), Poisson (event counts in intervals), Normal (continuous, symmetric); PMF/PDF/CDF using `scipy.stats`
- **Confidence Intervals** — margin of error, significance level, effect of sample size
- **Hypothesis Testing** — H₀ / H₁ definition, p-value interpretation, Type I & II errors
- **One & Two Sample Tests** — one-sample z/t-test, independent two-sample t-test, paired t-test
- **Test of Proportions & Variance** — z-test for proportions, F-test for variance equality
- **ANOVA** — one-way ANOVA for comparing 3+ group means simultaneously
- **Chi-Square Test** — independence test for categorical variables via contingency tables
- **Statistical Power** — relationship between sample size, effect size, α and power (1 − β)
- **Correlation & Covariance** — Pearson r, covariance matrix; correlation ≠ causation
- **Central Limit Theorem** — sampling distribution of the mean converges to Normal as n → ∞

**Project — Applied Statistics:**

*Part 1 — Probability & Distributions:* Solved 6 real-world problems using Binomial, Poisson and Normal distributions. Designed an e-learning question difficulty classifier (Easy 90%, Moderate 60%, Hard 30%) using `norm.cdf`.

*Part 2 — Basketball Team Investment Analysis:* EDA and feature engineering (win %, avg baskets per game/tournament) on 61-team dataset. Identified confounding variable (team age) via correlation analysis. Shortlisted 6 investable teams (Team 21, 11, 39, 27, 47, 59) formed 1941–1998.

*Part 3 — EU Startup Funding Analysis:* Hypothesis testing on Disrupt event funding data. H1: Operating vs Closed funds — t-test, p=0.1932 (fail to reject). H2: Winner vs Contestant operating rate — z-test, p=0.8767 (fail to reject). H3: Funds across NY/SF/EU — one-way ANOVA.

**Tools:** Python, Pandas, NumPy, SciPy, Matplotlib, Seaborn

---
### 02. Supervised Learning
📁 [`02_Supervised_Learning/`](./02_Supervised_Learning/)
📅 **Completed:** July 2021

**Topics Covered:**
- **Linear Regression** — OLS cost function, gradient descent, R², residual analysis, multicollinearity (VIF)
- **Logistic Regression** — sigmoid function, log-loss, decision boundary, L1/L2 regularisation, threshold tuning via `predict_proba`
- **K-Nearest Neighbours (KNN)** — distance metrics (Euclidean, Manhattan, Minkowski), choosing optimal K, BallTree/KDTree algorithms, mandatory feature scaling
- **Naive Bayes** — Bayes theorem, Gaussian/Multinomial/Bernoulli variants, Laplace smoothing, independence assumption
- **Support Vector Machines (SVM)** — maximum-margin hyperplane, C (regularisation) and γ (gamma) parameters, kernel trick (Linear, RBF, Polynomial), multi-class strategies
- **Model Evaluation** — confusion matrix, accuracy, precision, recall, F1-score, AUC-ROC, balanced accuracy; when NOT to use accuracy (imbalanced classes)
- **Class Imbalance** — under-sampling, over-sampling, SMOTE (Synthetic Minority Oversampling Technique)
- **Hyperparameter Tuning** — GridSearchCV with cross-validation, stratified train-test split

**Project — Supervised Learning:**

*Part 1 — Orthopaedic Patient Classification (KNN):* Classified patients into Normal / Hernia / Spondylolisthesis using 6 biomechanical features. Applied MinMaxScaler, balanced accuracy scoring (class imbalance), and GridSearchCV. Final model: KNN with K=19, BallTree algorithm, balanced accuracy = 0.756.

*Part 2 — Bank Customer Loan Conversion (Logistic Regression + Naive Bayes):* Predicted whether a customer would take a loan on their credit card (targeted marketing). Applied SMOTE for class imbalance, StandardScaler, engineered target column from ambiguous data. Final model: Logistic Regression — better recall and F1 than Naive Bayes.

**Tools:** Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn

---
### 03. Ensemble Techniques
📁 [`03_Ensemble_Techniques/`](./03_Ensemble_Techniques/)
📅 **Completed:** July 2021

**Topics Covered:**
- **Decision Trees (CART)** — Gini impurity vs Information Gain splitting, pre/post-pruning, depth and leaf controls
- **Bias-Variance Trade-off** — bagging reduces variance, boosting reduces bias; total error decomposition
- **Bagging** — bootstrap sampling of training data, aggregate predictions; decorrelates models to reduce variance
- **Random Forest** — bagging + random feature subsets per split; OOB error, feature importance via mean decrease in impurity
- **AdaBoost** — sequential boosting by up-weighting misclassified samples; `learning_rate`, decision stumps as weak learners
- **Gradient Boosting (GBC)** — fits each new tree to residuals via gradient descent; `subsample` for stochastic boosting
- **XGBoost** — regularised gradient boosting with L1/L2 penalties, native NaN handling, `scale_pos_weight` for class imbalance
- **CatBoost** — gradient boosting with native categorical feature handling (no encoding needed), symmetric trees
- **Model Comparison** — train vs test gap to detect overfitting; ROC AUC as comparison metric
- **Class Imbalance (Advanced)** — ADASYN, SMOTETomek, SMOTEENN (beyond basic SMOTE)
- **Model Serialisation** — pickling trained models (`.pkl`) for deployment; Flask GUI integration

**Project — Telecom Customer Churn Prediction:**

Predicted customer churn (Yes/No) for a telecom company to drive retention programmes. Dataset: ~7,000 customers × 21 features (services, account info, demographics). Applied ADASYN for class imbalance (~27% churn), manual categorical encoding, 70:30 train-test split. Trained and tuned 5 ensemble models (Random Forest, AdaBoost, GBC, CatBoost, XGBoost) using GridSearchCV. XGBoost overfit; CatBoost underfit; **Gradient Boosting Classifier** selected as final model (best train/test balance, highest ROC AUC). Model pickled and deployed as a Flask web app on Heroku.

**Tools:** Python, Pandas, Scikit-learn, XGBoost, CatBoost, Imbalanced-learn, Pickle, Flask

---
### 04. Unsupervised Learning
📁 `04_Unsupervised_Learning/`

**Core Topics:**
- K-Means Clustering (Elbow method, Silhouette score)
- Hierarchical / Agglomerative Clustering
- DBSCAN
- Principal Component Analysis (PCA)
- t-SNE for high-dimensional visualisation
- Anomaly Detection

**Tools:** Python, Scikit-learn, Matplotlib, Seaborn

---

### 05. Feature Engineering & Model Tuning
📁 `05_Feature_Engineering_Model_Tuning/`

**Core Topics:**
- Handling missing values and outliers
- Encoding: One-Hot, Label, Target Encoding
- Feature scaling: StandardScaler, MinMaxScaler, RobustScaler
- Feature selection: RFE, SelectKBest
- Hyperparameter tuning: GridSearchCV, RandomizedSearchCV, Bayesian Optimisation
- Building ML Pipelines with Scikit-learn

**Tools:** Python, Scikit-learn, Optuna, Pandas

---

### 06. Recommendation Systems
📁 `06_Recommendation_Systems/`

**Core Topics:**
- Collaborative Filtering (User-based and Item-based)
- Content-Based Filtering
- Matrix Factorisation (SVD, ALS)
- Hybrid Recommendation approaches
- Evaluation metrics: RMSE, MAE, Precision@K, Recall@K
- Cold Start problem

**Tools:** Python, Surprise, Pandas, NumPy

---

### 07. Neural Networks & Deep Learning
📁 `07_Neural_Networks_Deep_Learning/`

**Core Topics:**
- Biological vs Artificial Neurons
- Multi-Layer Perceptrons (MLP)
- Activation functions: ReLU, Sigmoid, Tanh, Softmax
- Backpropagation and Gradient Descent
- Regularisation: Dropout, Batch Normalisation, L1/L2
- Convolutional Neural Networks (CNN) — basics
- Recurrent Neural Networks (RNN) — basics
- Optimisers: Adam, SGD, RMSProp

**Tools:** Python, TensorFlow, Keras, NumPy

---

### 08. CNN Architecture & Transfer Learning
📁 `08_Computer_Vision_CNN_Transfer_Learning/`

**Core Topics:**
- Deep dive into CNN layers: Conv, Pooling, Fully Connected
- Architectures: VGG16, VGG19, ResNet50, InceptionV3, MobileNet
- Transfer Learning: feature extraction vs fine-tuning
- Image augmentation techniques
- Multi-class image classification
- Visualising CNN feature maps

**Tools:** Python, TensorFlow, Keras, OpenCV, PIL

---

### 09. Advanced Computer Vision - Object Detection
📁 `09_Computer_Vision_Object_Detection/`

**Core Topics:**
- Region-based CNNs: R-CNN, Fast R-CNN, Faster R-CNN
- YOLO (You Only Look Once) family
- Single Shot MultiBox Detector (SSD)
- Anchor boxes, IoU, Non-Maximum Suppression (NMS)
- Semantic Segmentation: FCN, U-Net
- Instance Segmentation: Mask R-CNN
- Object Tracking basics

**Tools:** Python, TensorFlow, Keras, OpenCV, YOLO

---

### 10. Statistical Natural Language Processing
📁 `10_Statistical_NLP/`

**Core Topics:**
- Text preprocessing: Tokenization, Stemming, Lemmatization, Stop-word removal
- Bag of Words (BoW) and N-grams
- TF-IDF
- Word Embeddings: Word2Vec (CBOW, Skip-gram), GloVe, FastText
- Sentiment Analysis
- Topic Modelling: LDA (Latent Dirichlet Allocation)
- Named Entity Recognition (NER)

**Tools:** Python, NLTK, spaCy, Gensim, Scikit-learn

---

### 11. Sequential Natural Language Processing
📁 `11_Sequential_NLP/`

**Core Topics:**
- Sequence-to-Sequence models
- RNN for text
- Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)
- Attention Mechanism
- Transformer architecture (self-attention, positional encoding)
- BERT (Bidirectional Encoder Representations from Transformers)
- Text generation and language modelling

**Tools:** Python, TensorFlow, Keras, HuggingFace Transformers

---

### 12. Capstone Project
📁 `12_Capstone_Project_AIML/`

An end-to-end AI/ML project integrating skills from all prior modules:
- Problem definition and business case framing
- Data collection, exploration, and preprocessing
- Feature engineering and model selection
- Model training, evaluation, and optimisation
- Presentation of insights and recommendations

---

## 🛠️ Technologies & Tools

### Data Science & ML Libraries

| Category | Libraries |
|----------|-----------|
| Data Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Deep Learning | TensorFlow, Keras |
| NLP | NLTK, spaCy, Gensim, HuggingFace Transformers |
| Computer Vision | OpenCV, PIL/Pillow |
| Recommendation Systems | Surprise |

### Development Tools

| Tool | Purpose |
|------|---------|
| Jupyter Notebook | Exploratory analysis and prototyping |
| Google Colab | GPU-accelerated training |
| Git & GitHub | Version control |
| Python | Primary language |

---

## 🏆 Projects Summary

| Project | Module | Description |
|---------|--------|-------------|
| Statistical Analysis | Applied Statistics | Hypothesis testing and EDA on real datasets |
| Predictive Modelling | Supervised Learning | Classification and regression models |
| Customer Segmentation | Unsupervised Learning | K-Means & PCA for market segmentation |
| Movie Recommendation | Recommendation Systems | Collaborative filtering engine |
| Image Classifier | CNN & Transfer Learning | Fine-tuned ResNet50 for image classification |
| Object Detection | Advanced CV | YOLO-based object detection pipeline |
| Sentiment Analysis | Statistical NLP | NLP pipeline for product review sentiment |
| Text Generation | Sequential NLP | LSTM-based language model |
| Capstone | Capstone | End-to-end business AI/ML solution |

---

## 📂 Repository Structure

```
pgp-ai-ml/
├── README.md                                   ← You are here
├── 01_Applied_Statistics/
│   └── README.md                               ← Module notes & projects
├── 02_Supervised_Learning/
│   └── README.md
├── 03_Ensemble_Techniques/
│   └── README.md
├── 04_Unsupervised_Learning/
│   └── README.md
├── 05_Feature_Engineering_Model_Tuning/
│   └── README.md
├── 06_Recommendation_Systems/
│   └── README.md
├── 07_Neural_Networks_Deep_Learning/
│   └── README.md
├── 08_Computer_Vision_CNN_Transfer_Learning/
│   └── README.md
├── 09_Computer_Vision_Object_Detection/
│   └── README.md
├── 10_Statistical_NLP/
│   └── README.md
├── 11_Sequential_NLP/
│   └── README.md
└── 12_Capstone_Project_AIML/
    └── README.md
```

---

## 🚀 How to Use This Repository

1. **Browse by module** — Navigate into any numbered folder to find notes, notebooks, and project files.
2. **Load into NotebookLM** — Upload this README and module READMEs into [NotebookLM](https://notebooklm.google.com/) to generate summaries, Q&A, and study guides.
3. **Add your materials** — Drop slides, notebooks (`.ipynb`), PDFs, or notes into each module folder and update the module README.

---

## 📜 Certification

- **Post Graduate Program in AI & ML** — Great Learning / UT Austin McCombs School of Business
- **Portfolio:** https://eportfolio.mygreatlearning.com/sunil-kunchoor-basavaraju

---

*Last updated: June 2026*



