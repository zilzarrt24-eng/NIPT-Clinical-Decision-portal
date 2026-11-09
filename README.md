# 🧬 NIPT Clinical Decision Support Portal
### Machine Learning-Assisted Screening & Comparative Benchmarking for Common Fetal Aneuploidies

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: Open Access Research](https://img.shields.io/badge/License-Academic%20Research-green.svg)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207840)
[![Status: Academic Prototype](https://img.shields.io/badge/Status-Academic%20Research%20Prototype-purple.svg)](#-regulatory--clinical-disclaimer)

---

## 📌 Executive Summary

The **NIPT Clinical Decision Support Portal** is an end-to-end computational screening and clinical decision support system designed to evaluate Non-Invasive Prenatal Testing (NIPT) data for the detection of fetal trisomies:
- **Trisomy 21 (Down Syndrome)**
- **Trisomy 18 (Edwards Syndrome)**
- **Trisomy 13 (Patau Syndrome)**

Built on a benchmark cohort of **5,518 clinical maternal plasma cell-free DNA (cfDNA) samples** published in PLOS ONE (`pone.0207840.s004`), this platform provides a **three-tier comparative framework** pitting conventional single-chromosome Z-score thresholding against multivariate, Platt-calibrated machine learning models (Support Vector Machines with Radial Basis Function kernels).

Crucially, this system addresses two widespread methodological challenges in computational prenatal genetics:
1. **Extreme Class Imbalance:** Real-world aneuploidy prevalence is low ($0.09\%$ to $0.49\%$). High-dimensional models are benchmarked via **Precision-Recall Area Under the Curve (PR-AUC)** rather than deceptive ROC-AUC scores.
2. **Post-Hoc Optimism Prevention:** Thresholds targeting $\ge 99.0\%$, $99.5\%$, and $99.9\%$ specificity are computed using a **leakage-safe, fold-derived cross-validation methodology** where cutoffs are determined exclusively on training folds and validated on unseen out-of-fold partitions, complete with exact **Wilson score 95% confidence intervals**.

---

## 🔬 Clinical Background & Cohort Overview

Non-Invasive Prenatal Testing analyzes circulating cell-free fetal DNA (cfDNA) derived from trophoblastic apoptotic shedding in maternal plasma. While conventional screening relies on counting statistics and univariate standard normal cutoffs ($Z \ge 3.0$), test sensitivity and false-positive rates are influenced by biological and pre-analytical covariates:
- Maternal weight and BMI
- Circulating fetal fraction ($\%$)
- Sequencing depth and read counts
- Library cfDNA concentration ($\text{ng}/\mu\text{L}$)
- GC content bias across chromosome regions

### Cohort Demographics & Imbalance Architecture
- **Total Cohort Size:** $N = 5,518$ maternal plasma sequencing samples.
- **Euploid Controls:** $5,471$ cases ($99.15\%$).
- **Confirmed Trisomy 21:** $27$ positive cases ($0.49\%$).
- **Confirmed Trisomy 18:** $15$ positive cases ($0.27\%$).
- **Confirmed Trisomy 13:** $5$ positive cases ($0.09\%$).
- **Pre-Analytical QC Filters:** $195$ samples flagged under strict laboratory quality metrics (e.g., low fetal fraction, sequencing depth bounds, fragment length anomalies such as sample `#666`).

---

## 🏗️ Three-Tier Model Architecture

| Model Tier | Model Architecture | Feature Representation | Calibration & Decision Logic |
| :--- | :--- | :--- | :--- |
| **Model A: Conventional Baseline** | Univariate $Z_{\text{run}}$ thresholding | Single chromosome GC-corrected $Z$-score ($Z_{13}, Z_{18}$, or $Z_{21}$) | Traditional clinical laboratory threshold sweep ($Z \ge 3.0$ standard reference cutoff) |
| **Model B: Multivariate Genomic SVM** | RBF-kernel Support Vector Machine (`SVC`) | 6 Genomic Z-Scores (target chromosome + reference autosome ratios) | Platt Sigmoid Calibration (`CalibratedClassifierCV`) with balanced class weighting |
| **Model C: Extended Multivariate SVM** | RBF-kernel Support Vector Machine (`SVC`) | 6 Genomic Z-Scores + Fetal Fraction + Library Conc + Sequencing Depth + Maternal Age + Gestational Age | Platt Sigmoid Calibration + feature imputation (`SimpleImputer`) + `RobustScaler` |

---

## 📊 Methodological Rigor & Evaluation

### 1. Leakage-Safe Fold-Derived Cross-Validation
Standard pooled out-of-fold sweeps introduce subtle optimism by selecting the single empirical cutoff after inspecting all combined validation predictions. To ensure zero data leakage:
1. In each fold of a **5-fold Stratified Cross-Validation**, the pipeline is fitted on `(X_train, y_train)`.
2. Operating thresholds targeting $\ge 99.0\%$, $99.5\%$, and $99.9\%$ specificity are computed **strictly on negative training samples** (`y_train == 0`).
3. Each cutoff is evaluated **once** on the completely held-out validation fold (`X_val, y_val`).
4. True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) are accumulated across folds.
5. All clinical contingency metrics (Sensitivity, Specificity, PPV, NPV) are reported with **Wilson score 95% Confidence Intervals**.

### 2. Leakage-Safe Operating Point Benchmark

#### Trisomy 21 ($N = 27$ Positives, Prevalence = $0.49\%$)
| Model | Target Spec | Sensitivity (95% CI) | Achieved Spec (95% CI) | PPV (95% CI) | NPV (95% CI) | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Baseline)** | 99.0% | **100.0%** [87.5%, 100.0%] | 98.98% [98.68%, 99.21%] | 32.5% [23.4%, 43.2%] | 100.0% | 27 / 56 / 5435 / 0 |
| **Model A (Baseline)** | 99.5% | **100.0%** [87.5%, 100.0%] | 99.47% [99.24%, 99.63%] | 48.2% [35.7%, 61.0%] | 100.0% | 27 / 29 / 5462 / 0 |
| **Model A (Baseline)** | 99.9% | 92.6% [76.6%, 97.9%] | 99.91% [99.79%, 99.96%] | 83.3% [66.4%, 92.7%] | 99.96% | 25 / 5 / 5486 / 2 |
| **Model B (6 Z-Scores)** | 99.0% | 74.1% [55.3%, 86.8%] | **100.0%** [99.93%, 100.0%] | **100.0%** [83.9%, 100.0%] | 99.87% | 20 / 0 / 5491 / 7 |
| **Model B (6 Z-Scores)** | 99.5% | 74.1% [55.3%, 86.8%] | **100.0%** [99.93%, 100.0%] | **100.0%** [83.9%, 100.0%] | 99.87% | 20 / 0 / 5491 / 7 |
| **Model B (6 Z-Scores)** | 99.9% | 74.1% [55.3%, 86.8%] | **100.0%** [99.93%, 100.0%] | **100.0%** [83.9%, 100.0%] | 99.87% | 20 / 0 / 5491 / 7 |
| **Model C (Extended)** | 99.0% | 74.1% [55.3%, 86.8%] | 99.98% [99.90%, 100.0%] | 95.2% [77.3%, 99.2%] | 99.87% | 20 / 1 / 5490 / 7 |

#### Trisomy 18 ($N = 15$ Positives, Prevalence = $0.27\%$)
| Model | Target Spec | Sensitivity (95% CI) | Achieved Spec (95% CI) | PPV (95% CI) | NPV (95% CI) | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Baseline)** | 99.0% | **100.0%** [79.6%, 100.0%] | 99.04% [98.74%, 99.26%] | 22.1% [13.8%, 33.3%] | 100.0% | 15 / 53 / 5450 / 0 |
| **Model A (Baseline)** | 99.5% | **100.0%** [79.6%, 100.0%] | 99.60% [99.40%, 99.74%] | 40.5% [26.3%, 56.5%] | 100.0% | 15 / 22 / 5481 / 0 |
| **Model A (Baseline)** | 99.9% | 93.3% [70.2%, 98.8%] | 99.93% [99.81%, 99.97%] | 77.8% [54.8%, 91.0%] | 99.98% | 14 / 4 / 5499 / 1 |
| **Model B (6 Z-Scores)** | 99.0% | 66.7% [41.7%, 84.8%] | **100.0%** [99.93%, 100.0%] | **100.0%** [72.2%, 100.0%] | 99.91% | 10 / 0 / 5503 / 5 |
| **Model C (Extended)** | 99.0% | 66.7% [41.7%, 84.8%] | **100.0%** [99.93%, 100.0%] | **100.0%** [72.2%, 100.0%] | 99.91% | 10 / 0 / 5503 / 5 |

#### Trisomy 13 ($N = 5$ Positives, Prevalence = $0.09\%$)
| Model | Target Spec | Sensitivity (95% CI) | Achieved Spec (95% CI) | PPV (95% CI) | NPV (95% CI) | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Baseline)** | 99.0% | **100.0%** [56.6%, 100.0%] | 99.04% [98.74%, 99.26%] | 8.6% [3.7%, 18.6%] | 100.0% | 5 / 53 / 5460 / 0 |
| **Model A (Baseline)** | 99.5% | **100.0%** [56.6%, 100.0%] | 99.55% [99.33%, 99.69%] | 16.7% [7.3%, 33.6%] | 100.0% | 5 / 25 / 5488 / 0 |
| **Model A (Baseline)** | 99.9% | 80.0% [37.6%, 96.4%] | 99.96% [99.87%, 99.99%] | 66.7% [30.0%, 90.3%] | 99.98% | 4 / 2 / 5511 / 1 |
| **Model B (6 Z-Scores)** | 99.0% | 40.0% [11.8%, 76.9%] | **100.0%** [99.93%, 100.0%] | **100.0%** [34.2%, 100.0%] | 99.95% | 2 / 0 / 5513 / 3 |
| **Model C (Extended)** | 99.0% | 40.0% [11.8%, 76.9%] | **100.0%** [99.93%, 100.0%] | **100.0%** [34.2%, 100.0%] | 99.95% | 2 / 0 / 5513 / 3 |

---

## 🔍 Feature Ablation: Genomics vs. Clinical Covariates

A central research investigation of this project is **Model Ablation**: *Does appending patient demographic and laboratory covariates (fetal fraction, library concentration, sequencing depth, maternal age, gestational age) to genomic Z-scores improve screening accuracy?*

| Aneuploidy | Positives | Model A PR-AUC | Model B PR-AUC | Model C PR-AUC | $\Delta\text{PR-AUC } (C - B)$ | Ablation Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **T13** | 5 | 0.9381 | 0.8268 | 0.8339 | $+0.0071$ | Comparable screening performance ($\Delta\text{PR-AUC} \le 0.01$). Extreme sample scarcity ($N=5$) dictates that each sample shifts sensitivity by $20.0\%$. |
| **T18** | 15 | 0.9957 | **0.9733** | 0.9434 | $-0.0299$ | **Model B outperformed Model C.** Adding clinical metadata reduced PR-AUC by $0.03$ due to mild covariate overfitting. |
| **T21** | 27 | 0.9822 | **0.9893** | 0.9875 | $-0.0018$ | Comparable performance ($\Delta\text{PR-AUC} \le 0.01$). Model B is the more parsimonious, robust choice. |

> **Key Clinical Finding:** The genomic Z-scores alone (Model B) capture nearly all discriminative signal. Appending non-genomic covariates in cohorts of this size offers no generalizable gain and risks learning spurious demographic noise.

---

## 🧪 Unsupervised NIPT Sample Profiling (K-Means & PCA)

The portal integrates an unsupervised laboratory quality monitoring module:
- **Feature Space:** Library cfDNA concentration ($\text{ng}/\mu\text{L}$), sequencing depth (total read counts), fetal fraction ($\%$), and fragment length distributions.
- **Cluster Diagnostics:** Evaluated across $K \in [2, 6]$.
  - $K = 2$ achieved the optimal **Silhouette Score (0.1593)** and lowest **Davies-Bouldin Index (2.2675)**.
- **Cluster Stratification:**
  - Cluster 0: $N = 1,952$ samples (low library concentration, high depth profile).
  - Cluster 1: $N = 3,566$ samples (standard library concentration profile).
- **Strict Clinical Governance:** This unsupervised profiling is strictly **non-diagnostic** and is used for pre-analytical batch tracking, library stability surveillance, and detecting systematic sequencing drift.

---

## 🖥️ Streamlit Multi-View Portal Architecture

The web application is structured into **8 modular views**:

```text
├── View 1: Dataset Overview
│   └── Cohort demographics, chromosome distributions, and interactive sample browser with 1-click filters.
├── View 2: Pre-analytical Quality Control (QC)
│   └── Fetal fraction distributions, short fragment peaks, QC-pass vs. QC-filtered cohorts, and sample #666 audit.
├── View 3: Conventional Z-Score Baseline
│   └── Interactive cutoff slider (1.5 to 6.0), real-time 2x2 contingency matrix, and Grey Zone (2.5 <= Z <= 3.5) inspector.
├── View 4: Machine Learning Screening
│   └── Dual mode: Cohort Sample Lookup (5,518 cases) or Interactive Patient Parameter Entry with calibrated risk bands.
├── View 5: Model Performance & Operating Points
│   └── Interactive ROC curves, PR curves, and primary Leakage-Safe Operating Point tables with Wilson 95% CIs.
├── View 6: Unsupervised Sample Profiling
│   └── Interactive 2D PCA cluster space, cluster medians, and library QC profiling.
├── View 7: Model B vs C Ablation & Feature Interpretation
│   └── Objective ablation delta analysis, paired sample discrepancy tables, and clinical covariate impact.
└── View 8: Clinical Limitations & Governance
    └── Academic Research Prototype framing, rare sample statistics, twin/mosaicism limitations, and regulatory disclaimer.
```

---

## 📁 Repository Directory Structure

```text
NIPT-Clinical-Decision-Portal/
├── app/
│   ├── streamlit_app.py               # Main Streamlit application entry point
│   ├── views/
│   │   ├── view1_overview.py          # View 1: Dataset Demographics & Overview
│   │   ├── view2_qc.py                # View 2: QC & Pre-analytical Quality Control
│   │   ├── view3_baseline.py          # View 3: Conventional Z-score Baseline
│   │   ├── view4_ml_screening.py      # View 4: Dual-Mode ML Screening Engine
│   │   ├── view5_performance.py       # View 5: Model Performance & ROC/PR Curves
│   │   ├── view6_clustering.py        # View 6: K-Means & PCA Sample Profiling
│   │   ├── view7_interpretation.py    # View 7: Model B vs C Feature Ablation
│   │   └── view8_limitations.py       # View 8: Limitations & Governance Disclaimer
│   └── __init__.py
├── data/
│   ├── raw/
│   │   └── pone.0207840.s004 (1).xlsx # Raw clinical Excel dataset (PLOS ONE)
│   └── processed/
│       ├── master_processed.csv       # Unified 5,518 sample dataset
│       ├── t13_dataset.csv            # Trisomy 13 modeling dataset
│       ├── t18_dataset.csv            # Trisomy 18 modeling dataset
│       └── t21_dataset.csv            # Trisomy 21 modeling dataset
├── models/
│   ├── ablation_summary.json          # Ablation metrics (PR-AUC, ROC-AUC, Brier)
│   ├── kmeans/                        # K-Means model, imputer, scaler, PCA coords
│   ├── t13/                           # T13 Model B/C pipelines, metrics, OOF predictions
│   ├── t18/                           # T18 Model B/C pipelines, metrics, OOF predictions
│   └── t21/                           # T21 Model B/C pipelines, metrics, OOF predictions
├── reports/
│   ├── model_evaluation_report.md     # Primary evaluation report
│   ├── operating_point_audit.md       # Leakage-safe methodology audit
│   ├── preprocessing_report.md        # Data processing and cohort integrity audit
│   ├── post_training_audit.md         # Post-training artifact verification audit
│   ├── streamlit_validation_report.md # Initial portal verification
│   └── final_ui_validation_report.md  # 32/32 Test suite pass report
├── src/
│   ├── preprocessing.py               # Data ingestion, validation, and feature building
│   ├── models.py                      # Pipeline definitions (RBF-SVM, Platt calibration)
│   ├── evaluation.py                  # Leakage-safe cross-validation & operating points
│   ├── clustering.py                  # K-Means diagnostics and PCA projection
│   ├── utils.py                       # Wilson score CI computation & contingency logic
│   └── __init__.py
├── tests/
│   └── test_streamlit_comprehensive.py# 32-check automated static and AppTest suite
├── run_preprocessing.py               # Executable pipeline for raw -> processed data
├── run_training.py                    # Executable orchestrator for model training & CV
├── requirements.txt                   # Project package dependencies
├── .gitignore                         # Git tracking exclusion rules
└── README.md                          # Comprehensive documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Python virtual environment (Python 3.10+ recommended):

```bash
git clone https://github.com/zilzarrt24-eng/NIPT-Clinical-Decision-portal.git
cd NIPT-Clinical-Decision-portal

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Data Preprocessing
Ingests `data/raw/pone.0207840.s004 (1).xlsx`, validates cross-sheet sample concordance across all 5,518 cases, flags pre-analytical quality anomalies, and builds modeling CSVs:

```bash
python run_preprocessing.py
```

### 3. Run Model Training & Leakage-Safe Evaluation
Executes 5-fold stratified cross-validation for T13, T18, and T21 across Models A, B, and C; conducts ablation benchmarking; trains the unsupervised K-Means profiling model; and serializes all models and audit reports:

```bash
python run_training.py
```

### 4. Execute the Automated Test Suite
Runs the comprehensive static code audit and Streamlit `AppTest` suite (32 tests covering all 8 views, mode toggles, and report consistency):

```bash
python tests/test_streamlit_comprehensive.py
```

### 5. Launch the Clinical Decision Support Portal
Start the interactive Streamlit portal locally:

```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚖️ Regulatory & Clinical Disclaimer

> **CRITICAL MEDICAL GOVERNANCE NOTICE:**
> 
> The **NIPT Clinical Decision Support Portal** is an **academic research prototype** created solely for algorithmic evaluation, clinical methodology research, and demonstration purposes.
> 
> - **Experimental Screening Outputs:** All predictions, risk classifications, and numerical probabilities generated by this platform are experimental model estimates and **do not constitute medical diagnoses**.
> - **Not a Medical Device:** This software has **not** been cleared, approved, or evaluated by the United States Food and Drug Administration (FDA), European Medicines Agency (CE-IVD / IVDR), or any national regulatory authority.
> - **Screening vs. Diagnosis:** Cell-free DNA (cfDNA) analysis is a screening methodology, not a diagnostic determination. False positives (e.g., confined placental mosaicism, maternal copy number variations) and false negatives (e.g., low fetal fraction, maternal obesity) can occur.
> - **Mandatory Confirmation:** All high-risk screening indications must be validated using accredited clinical diagnostic procedures (e.g., diagnostic amniocentesis, chorionic villus sampling [CVS], or fetal karyotyping/chromosomal microarray).

---

## 📚 References & Acknowledgments
- **Benchmark Cohort:** PLOS ONE Open-Access Clinical Dataset (`pone.0207840.s004`).
- **Methodology Reference:** *A non-invasive prenatal testing (NIPT) algorithm for trisomies 21, 18, and 13*, PLOS ONE, 2018.
- **Statistical Reference:** Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference*. Journal of the American Statistical Association.
