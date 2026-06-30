# 📊 Applied Statistics

> **Module Completed:** May 2021

---

## 🎯 Module Overview

This module builds the statistical foundation needed for all subsequent AI/ML work. It covers the complete workflow from descriptive statistics through to inferential statistics and hypothesis testing — the toolkit used to understand data, test assumptions, and draw evidence-based conclusions.

---

## 📖 Topics Covered

### Descriptive Statistics
Summarising and describing datasets using measures of central tendency (mean, median, mode) and spread (variance, standard deviation, IQR). Also covers skewness, kurtosis, and visualisation using box plots and histograms.

### Probability Theory
Fundamentals of probability: joint probability (P(A ∩ B)), marginal probability, and conditional probability (P(A|B)). Includes Bayes theorem and its real-world applications.

### Probability Distributions
Understanding when and how to apply:
- **Binomial** — fixed number of trials, binary outcomes (e.g., pass/fail, default/no default)
- **Poisson** — count of events in a fixed interval (e.g., customer arrivals per hour)
- **Normal** — continuous symmetric distribution; used for CDF lookups and z-score calculations
- Covers PMF, PDF, and CDF concepts with `scipy.stats`

### Confidence Intervals
Constructing intervals around a sample statistic (e.g., mean) that capture the population parameter with a specified level of confidence (e.g., 95%). Covers margin of error and the effect of sample size.

### Hypothesis Testing Framework
Step-by-step process: define H₀ and H₁ → choose significance level (α) → select test → compute test statistic and p-value → make decision. Covers Type I error (false positive) and Type II error (false negative).

### One-Sample & Two-Sample Tests
- **One-sample t-test / z-test** — compare a sample mean to a known population value
- **Two-sample independent t-test** — compare means of two independent groups
- **Paired t-test** — compare means of two related/matched groups

### Test of Proportions & Variance
Z-test for proportions to compare percentages between groups. F-test and Levene's test for comparing variances across groups.

### ANOVA (Analysis of Variance)
One-way ANOVA for comparing means across **3 or more** groups simultaneously. Determines if at least one group mean is significantly different without inflating Type I error from multiple t-tests.

### Chi-Square Test
Tests whether two categorical variables are independent. Used when data is in frequency/count form (e.g., contingency tables).

### Statistical Power & Type II Error
Power = 1 − P(Type II Error). Covers how sample size, effect size, and significance level affect the ability to detect a true effect.

### Correlation & Covariance
Pearson correlation coefficient (r) to measure linear relationship strength and direction. Covariance matrix for multi-variable relationships. Key reminder: correlation does not imply causation.

### Central Limit Theorem (CLT)
As sample size increases, the sampling distribution of the mean approaches normal regardless of the population distribution. Foundation for most inferential statistics.

---

## 🏆 Project — Applied Statistics

**Submitted:** 30 May 2021

### Part 1 — Probability & Distributions

Applied probability theory and distributions to 6 real-world problems:

| Problem | Distribution Used | Key Insight |
|---------|------------------|-------------|
| Joint & conditional probability | Probability rules | P(A∩B) vs P(A\|B) — different denominators |
| Bank loan defaults | Binomial | `binom.pmf(k, n, p)` |
| Customer arrival rate | Poisson | `poisson.pmf(k, mu)` |
| Student exam scoring | Binomial | Discrete outcomes over fixed trials |
| Exam distinction threshold | Normal | Reverse lookup via `norm.ppf` / `norm.cdf` |
| E-learning question difficulty | Normal | Easy (90% solve), Moderate (60%), Hard (30%) — mapped to z-scores |

---

### Part 2 — Basketball Team Investment Analysis

**Business Problem:** Company X wants to identify best basketball teams to sign contracts with for future tournaments.

**Dataset:** 61 teams × 13 features — tournaments played, score, wins, losses, baskets scored/given, year of establishment.

**Approach:**
1. **EDA** — raw rankings by score, games played, baskets scored
2. **Data Cleaning** — handled `"-"` strings in 9 columns (not null, but undefined); dropped `TournamentChampion` & `Runner-up` (>78% missing); removed `Team 61` (sparse)
3. **Feature Engineering** — created `Games_Won_Percentage`, `Avg_Basket_Per_Game`, `Avg_Basket_Per_Tournament` to enable like-for-like comparison
4. **Correlation Analysis** — strong positive correlation across all columns, but this was a confounding effect (older teams simply played more games)
5. **Filtering** — excluded teams formed before 1940 (already under competitor contracts)

**Outcome:** 6 shortlisted teams — **Team 21, 11, 39, 27, 47, 59** (all formed 1941–1998, comparable performance to older teams)

---

### Part 3 — EU Startup Funding Analysis

**Business Problem:** Analyse EU startup data from Disrupt events to understand funding patterns.

**Dataset:** Startup name, funds raised (in millions), operating status, event location (NY / SF / EU), result (winner / contestant), year.

| Hypothesis | Test | p-value | Decision |
|------------|------|---------|----------|
| H1: No difference in funds — Operating vs Closed | Two-sample t-test | 0.1932 | Fail to reject H₀ |
| H2: No difference in operating rate — Winners vs Contestants | Z-test for proportions | 0.8767 | Fail to reject H₀ |
| H3: No difference in funds — across NY, SF, EU (Disrupt, 2013+) | One-way ANOVA | — | Multi-group test |

**Preprocessing:** Converted funding strings ("5M") to float; removed outliers above box plot upper fence; filtered for `Disrupt` events from 2013 onwards.

---

## 🔑 Key Learnings

1. Always check data types — numeric columns stored as strings are a common data quality issue
2. **Correlation ≠ Causation** — identify confounding variables before drawing conclusions
3. Feature engineering reveals insights that raw features hide (e.g., win % vs raw win count)
4. "Fail to reject H₀" ≠ "Accept H₀" — absence of evidence is not evidence of absence
5. Choose the right test: t-test (2 group means) → ANOVA (3+ group means) → Z-test (proportions) → Chi-square (categorical independence)
6. Outliers don't always need removal — understand *why* they exist before deciding
7. A `"-"` in data means "not applicable" or "unknown" — never impute it as zero without domain validation

---

## 🛠️ Tools & Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom, poisson, norm, ttest_ind, f_oneway, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
```
