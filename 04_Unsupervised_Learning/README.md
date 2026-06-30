# 🔍 Unsupervised Learning

> **Module Completed:** July 2021

---

## 🎯 Module Overview

Unsupervised learning finds hidden structure in unlabelled data. Without a target variable, the algorithms discover natural groupings (clustering) or compact representations (dimensionality reduction) from the data itself. This module covers the core clustering algorithms, evaluation methods, and dimensionality reduction techniques including PCA.

---

## 📖 Topics Covered

### Introduction to Clustering
Clustering partitions data into groups where points within a group are more similar to each other than to points in other groups. Key concepts:
- Distance metrics: Euclidean, Manhattan, Cosine similarity
- Intra-cluster cohesion vs inter-cluster separation
- Hard vs soft clustering (definite vs probabilistic assignment)
- When clustering is useful: customer segmentation, anomaly detection, data compression, imputing missing values

### K-Means Clustering
Iterative algorithm that assigns each point to the nearest of K centroids, then recomputes centroids, until convergence. Covers:
- Initialisation: random vs K-Means++ (smarter seeding to avoid poor local optima)
- **Elbow Method** — plot WCSS (Within-Cluster Sum of Squares / inertia) vs K; choose the K at the "elbow" where improvement plateaus
- **Silhouette Score** — measures how similar a point is to its own cluster vs the nearest other cluster; ranges −1 to +1; higher = better-defined clusters
- Limitations: assumes spherical clusters, sensitive to outliers and feature scale — always normalise before K-Means
- `sklearn.cluster.KMeans` with `init='k-means++'`, `n_init=10`

### Hierarchical / Agglomerative Clustering
Builds a tree (dendrogram) of nested clusters by iteratively merging the two closest clusters (bottom-up agglomerative). Covers:
- **Linkage methods:** Single (min distance), Complete (max distance), Average, Ward (minimises total within-cluster variance — most commonly used)
- **Cophenetic Correlation Coefficient** — measures how faithfully the dendrogram preserves pairwise distances; closer to 1 = better
- Reading a dendrogram: horizontal cut determines number of clusters; longer vertical lines = better-separated clusters
- Advantages over K-Means: no need to pre-specify K; reveals hierarchical structure; deterministic (no random initialisation)
- `scipy.cluster.hierarchy.linkage`, `dendrogram`, `fcluster`

### K-Means vs Hierarchical — Key Differences

| Aspect | K-Means | Hierarchical |
|--------|---------|--------------|
| K required upfront | Yes | No (read from dendrogram) |
| Scalability | Fast — O(nKt) | Slow — O(n² log n) |
| Result | Hard partition | Tree of clusters |
| Reproducibility | Non-deterministic | Deterministic |
| Best for | Large datasets | Small datasets, tree structure needed |

### Using Clustering for Missing Value Imputation
Rather than mean/median imputation, cluster the complete records first, then assign missing-value rows to their nearest cluster and impute with that cluster's statistics. Produces more contextually accurate imputations than global averages.

### Synthetic Data Generation with Kernel Density Estimation
`sklearn.neighbors.KernelDensity` estimates the probability density of the training data, from which new synthetic samples can be drawn. Combined with K-Means to assign cluster labels to generated points.

### Principal Component Analysis (PCA)
Linear dimensionality reduction that projects data onto the directions of maximum variance (principal components). Covers:
- **Covariance matrix** — measures how features vary together
- **Eigenvectors and eigenvalues** — eigenvectors define component directions; eigenvalues define the variance explained by each component
- **Explained Variance Ratio** — proportion of total variance captured by each PC; cumulative plot used to choose number of components
- **Scree plot** — visualises eigenvalues to find the "elbow"
- Rule of thumb: retain components that explain ≥ 95% of variance
- PCA for speed: reduce features to fewer PCs, train model on PCs — near-same accuracy with much less compute
- PCA for images: each pixel is a feature; PCA compresses image while retaining visual information
- `sklearn.decomposition.PCA(n_components=k)`

### Dimensionality Reduction — Broader Taxonomy
Beyond PCA, the module surveys:

**Linear Algebra Methods:**
- **PCA** — directions of maximum variance
- **SVD** (Singular Value Decomposition) — matrix factorisation; used in recommendation systems
- **NMF** (Non-Negative Matrix Factorisation) — parts-based decomposition; useful for topic modelling

**Manifold Learning Methods (non-linear):**
- **t-SNE** — preserves local neighbourhood structure; ideal for visualising high-dimensional clusters in 2D/3D; not suitable for compression/inference
- **Isomap** — geodesic distances on the data manifold
- **LLE** (Locally Linear Embedding) — local linear reconstructions

**Feature Selection Methods (dimensionality reduction without transformation):**
- Missing Value Ratio, Low Variance Filter, High Correlation Filter
- Recursive Feature Elimination (RFE) with Random Forest
- Forward/Backward Feature Selection

### PCA on Multimedia Data
PCA is applicable beyond tabular data:
- **Images** — each pixel is a feature; PCA retains the most visually informative components (reconstructed image looks similar with far fewer features)
- **Text** — via TF-IDF matrix, LSA (Latent Semantic Analysis) uses SVD to find latent topics
- **Video** — dimensionality reduction frame-by-frame or across temporal features

---

## 🏆 Project — Unsupervised Learning

**Submitted:** 22 July 2021

The project had 5 independent parts, each targeting a different unsupervised technique.

---

### Part 1 — Car Clustering & Regression (K-Means + Hierarchical)

**Business Problem:** Group cars by their characteristics and fit separate linear regression models per cluster to understand cluster-specific relationships.

**Dataset:** Merged from `Part1 - Car name.csv` + `Part1 - Car-Attributes.json` → `Part1 - Final Dataset.csv`

**Features:** `mpg`, `cylinders`, `displacement` (disp), `horsepower` (hp), `weight`, `acceleration`, `model_year`, `origin`

**Data Cleaning:**
- `hp` column stored as `object` — used regex to find 6 rows with `"?"` values; dropped those rows
- `origin` and `model_year` one-hot encoded (categorical, not ordinal)
- All numerical columns normalised before clustering

**EDA Findings:**
- More cylinders → lower mpg; newer model year → higher mpg; higher weight → lower mpg
- Visible multimodal distributions suggest multiple natural groups
- No outliers present

**Clustering Results:**
- **K-Means:** Elbow method identified k=3 as optimal (clear inflection in WCSS plot)
- **Hierarchical:** Average linkage gave highest Cophenetic Correlation; dendrogram shows 3 clean clusters with 2 sub-clusters detectable in the middle group
- **Consensus:** k=3 clusters chosen for final model

**Regression per Cluster:** Linear regression trained independently per cluster; cluster-specific coefficients reveal that the relationship between features and mpg differs meaningfully across vehicle groups

---

### Part 2 — Wine Quality Imputation (K-Means)

**Dataset:** `Part2 - Company.xlsx` — chemical composition + quality rating of wine samples; 18 missing values in `Quality` column

**Approach:**
1. Removed NaN rows temporarily to build a clean K-Means model (bimodal distribution → 2 clusters)
2. Applied the cluster model to the rows with missing `Quality` — each row assigned to a cluster
3. Imputed `Quality` with the cluster centroid's value

**Bonus — Synthetic Data Generation:** Used `KernelDensity` to model the training distribution, sampled new points, then applied K-Means to assign quality labels. Successfully generated synthetic wine samples that matched the original distribution.

---

### Part 3 — Vehicle Shape Classification + PCA (SVM)

**Dataset:** `Part3 - vehicle.csv` — 18 shape features extracted from silhouettes of 4 vehicle types (car, bus, van, opel)

**Challenge:** 18 features + class imbalance + high correlation in `elongatedness`

**Approach:**
- Dropped `elongatedness` (high correlation); StandardScaler applied
- **Baseline SVM:** trained on all 18 features → high balanced accuracy
- **PCA:** cumulative explained variance plot showed 6 components capture ~94% of variance
- **PCA + SVM:** trained on 6 PCA components — accuracy dropped only ~8% while using 1/3 of the features

**Key Insight:** Dimensionality reduction trades a small accuracy loss for a large reduction in computation cost — critical at scale.

---

### Part 4 — IPL Batsman Ranking (PCA + K-Means)

**Dataset:** `Part4 - batting_bowling_ipl_bat.csv` — IPL batting statistics; 90 null rows dropped

**Features:** Runs, Strike Rate, Fours, Sixes, Half-Centuries, Average score per match

**Approach:**
1. **PCA for ranking:** Computed covariance matrix → eigenvalues/eigenvectors → sorted by eigenvalue magnitude → top 4 components used to generate a composite performance rank
2. **K-Means for grouping:** Elbow + Silhouette methods both identified k=4 as optimal

**Player Tier Results:**
| Rank | Cluster | Profile |
|------|---------|---------|
| 1 | High performers | Mean score/match: 40.86, avg strike rate, 54 mean fours, 5 mean sixes |
| 2 | Above average | Moderate across all metrics |
| 3 | Average | Below Rank 2 |
| 4 | Low performers | Weakest across all batting metrics |

---

### Part 5 — Dimensionality Reduction Survey

Catalogued all major dimensionality reduction techniques available in Python:
- Linear: PCA, SVD, NMF
- Manifold: t-SNE, Isomap, LLE, MDS, Spectral Embedding
- Feature selection: VIF filter, correlation filter, RFE, forward/backward selection
- Demonstrated PCA on image data (bee and panda images) — showed visual quality at different numbers of PCA components

---

## 🔑 Key Learnings

1. **Always normalise before K-Means** — distance-based; unscaled features dominate cluster assignments
2. **Elbow + Silhouette together** — elbow gives inertia view, silhouette gives separation quality; use both for robust K selection
3. **Cophenetic coefficient validates hierarchical clustering** — value close to 1 = dendrogram faithfully represents data distances
4. **K-Means for imputation is better than mean/median** — imputes based on similar records, not global average
5. **PCA components ≠ original features** — they are linear combinations; interpretability is lost but variance is preserved efficiently
6. **94% explained variance with 1/3 the features** — strong case for PCA before training expensive models on large datasets
7. **t-SNE is for visualisation only** — do not use for compression or model training; not deterministic, not invertible
8. **Eigenvalues rank information content** — larger eigenvalue = more variance explained = higher priority component

---

## 🛠️ Tools & Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

from imblearn.over_sampling import SMOTE
```
