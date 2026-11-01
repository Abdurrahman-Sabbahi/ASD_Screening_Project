# ASD Risk Screening: Comparative Evaluation of Statistical Learning Methods

**ISyE 7406 – Data Mining & Statistical Learning | Georgia Tech OMSA**  
**Author:** Abdurrahman Sabbahi (`asabbahi3@gatech.edu`) · Group 20  
**Course Term:** Summer 2026

---

## Overview

This project compares statistical learning classifiers for predicting the **AQ-10 questionnaire-derived screening outcome** for Autism Spectrum Disorder (ASD) across three developmental age groups: children, adolescents, and adults.

> **Important framing note:** The outcome variable is a *questionnaire-derived screening label* — a binary flag generated when an individual's AQ-10 total score exceeds the clinical threshold (> 6). This is **not** an independent clinical diagnosis. All models learn to reproduce the AQ-10 scoring algorithm, not predict ASD from first principles. Near-perfect predictive accuracy reflects fidelity to the scoring rule, not clinical validity.

---

## Repository Contents

| File | Description |
|------|-------------|
| [`Group_20_ISyE 7406 Project Proposal.pdf`](./Group_20_ISyE%207406%20Project%20Proposal.pdf) | Original project proposal outlining motivation, dataset, research questions, and planned methodology |
| [`Group_20_Midterm_report-1.pdf`](./Group_20_Midterm_report-1.pdf) | Midterm report covering EDA, Phase 1 modeling (5 classifiers), cross-validation results, feature importance, and preliminary conclusions |
| [`Group20_ASD_Final_report_submission-1.pdf`](./Group20_ASD_Final_report_submission-1.pdf) | Final report extending Phase 1 with GBM, structural analysis of near-perfect LR performance, clinical cutoff comparison, and complete source code |

---

## Motivation

ASD affects roughly 1 in 36 children in the United States, yet the path from parental concern to a confirmed specialist referral can take years. This project grew from a personal motivation: observing ASD-related behaviors in my son during early childhood sparked my interest in understanding how statistical learning methods can be applied to behavioral screening data and which factors contribute most to early ASD risk screening.

---

## Dataset

- **Source:** Fadi Fayez Thabtah via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets?search=Autism) / [Kaggle](https://www.kaggle.com/datasets/raiyangani/autistic-spectrum-disorder-screening-data)
- **Three datasets** representing different developmental stages:

| Group | Age Range | N | Screen+ (%) |
|-------|-----------|---|-------------|
| Child | 4–11 | 292 | 48.3% |
| Adolescent | 12–16 | 104 | 60.6% |
| Adult | 17+ | 704 | 26.8% |
| **Combined** | – | **1,100** | **35.7%** |

Each row corresponds to one individual who completed the AQ-10 screener. Features include:
- **AQ-10 behavioral items** (A1–A10): 10 binary screening responses
- **Demographics:** age, gender, jaundice history, family ASD history, ethnicity, country of residence, relation

---

## Research Questions

1. Which statistical learning classifier provides the best predictive performance for ASD screening?
2. Does classifier performance vary meaningfully across age groups?
3. Which behavioral items and demographic characteristics are the strongest predictors?
4. Are the ten AQ items alone sufficient, or do demographic variables add predictive value?

---

## Methodology

### Preprocessing
- Raw AQ total score (`result`) dropped immediately (encodes the label directly)
- Age missingness (n = 3) imputed from training-set medians only (leakage-free)
- Binary variables (gender, jaundice, family history, prior app use) recoded to 0/1
- Nominal factors (ethnicity, relation) dummy-encoded from training split only
- Country of residence excluded (50+ sparse levels, near-zero variance)
- Feature scaling applied inside each CV fold

### Evaluation Pipeline
1. Stratified **80/20 train/test split** (done once before any modeling)
2. **5-fold cross-validation** on training data for model selection and hyperparameter tuning
3. Final models refit on the full training set with tuned hyperparameters
4. Evaluated **exactly once** on the held-out test set

**Primary metric:** AUC (threshold-independent, robust to class imbalance)  
**Secondary metric:** Recall (important in a screening context to minimize missed cases)

### Models (Phase 1)

| Model | Rationale |
|-------|-----------|
| Logistic Regression | Interpretable linear baseline; directly models log-odds of ASD |
| LDA | Generative linear model; useful comparator to LR under Gaussian assumptions |
| Naive Bayes | Tests how far the conditional independence assumption departs from reality |
| KNN | Non-parametric, distance-based; no linearity assumption, sensitive to scale |
| Random Forest | Flexible ensemble; captures interactions without manual feature engineering |

### Phase 2 Extension
- **GBM (Gradient Boosted Machine)** added as a competitive non-linear benchmark and independent feature importance source
- Structural analysis of *why* Logistic Regression achieves near-perfect AUC
- Comparison of all ML models against the AQ-10 clinical cutoff rule (score > 6)

---

## Key Results

### Cross-Validation Performance (Combined Dataset)

| Model | AUC | Acc | Recall | F1 |
|-------|-----|-----|--------|-----|
| Logistic Reg. | 0.998 | 0.998 | 1.000 | 0.997 |
| LDA | 0.998 | 0.975 | 0.987 | 0.966 |
| Random Forest | 0.995 | 0.960 | 0.917 | 0.943 |
| GBM | 1.000 | 1.000 | 1.000 | 1.000 |
| Naive Bayes | 0.960 | 0.902 | 0.851 | 0.862 |
| KNN (k=5) | 0.960 | 0.890 | 0.908 | 0.855 |

### Hold-Out Test Set Performance (Combined Dataset)

| Model | AUC | Acc | Recall | F1 |
|-------|-----|-----|--------|-----|
| Logistic Reg. | 1.000 | 1.000 | 1.000 | 1.000 |
| LDA | 1.000 | 0.977 | 0.987 | 0.969 |
| GBM | 1.000 | 0.991 | 0.974 | 0.987 |
| Random Forest | 0.995 | 0.968 | 0.923 | 0.954 |
| KNN (tuned) | 0.950 | 0.886 | 0.846 | 0.841 |

---

## Key Findings

### 1. Feature Importance
- **AQ items dominate** — all ten AQ-10 behavioral items consistently outrank every demographic variable across both RF Mean Decrease in Accuracy and GBM Relative Influence
- Top predictors: **A5** (social interaction), **A9** (social ease), **A4** (task switching), **A6** (pragmatic language)
- The **AQ-only feature set matched or exceeded the full feature set** on all models and age groups

### 2. Why Logistic Regression Achieves Near-Perfect Performance
The ASD screening label is generated deterministically:

```
Y = 1(A1 + A2 + ... + A10 > 6)
```

The outcome is a **linear threshold of the same item scores used as predictors**. Logistic regression is structurally matched to this problem — it learns a weighted linear sum, and when the true boundary is exactly a linear sum threshold, near-perfect performance is the expected result. This is structural alignment, not label leakage (the raw AQ total was dropped before modeling).

### 3. Ensemble Methods Add No Value
GBM and RF achieve competitive but not superior AUC. When the true decision boundary is a linear sum threshold, tree-based ensembles approximate a simple linear function using piecewise-constant representations — computationally expensive with no accuracy benefit.

### 4. ML vs. Clinical Rule
The AQ-10 cutoff rule (score > 6) achieves near-perfect performance across all groups because it was designed to be sensitive and the released labels are derived directly from it. All trained ML models match this baseline. The practical advantage of ML lies in **calibrated probability estimates** and **flexible decision thresholds**, not in improved label agreement.

---

## Limitations

- Outcome is derived directly from AQ-10 responses — models learn the scoring algorithm, not clinical ASD
- Single 80/20 split introduces sampling variability; adolescent group (n = 104) is most affected
- Selection bias: only individuals who completed the AQ-10 appear in the data; models cannot generalize to unscreened populations
- No confidence intervals computed; all conclusions rest on point estimates

---

## Software

**R (v4.3+):** `tidyverse`, `caret`, `randomForest`, `e1071`, `MASS`, `glmnet`, `pROC`, `corrplot`, `gridExtra`, `knitr`, `kableExtra`, `gbm`

---

## References

- Thabtah, F. (2017). *ASD Screening Datasets*. UCI Machine Learning Repository.
- Baron-Cohen, S. et al. (2001). The Autism-Spectrum Quotient (AQ). *Journal of Autism and Developmental Disorders*, 31(1), 5–17.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Kuhn, M. (2008). Building Predictive Models in R Using the caret Package. *Journal of Statistical Software*, 28(5).
