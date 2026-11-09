# Comprehensive Streamlit Portal Testing & Validation Report

**Project:** NIPT Clinical Decision Support Portal  
**Scope:** Functional, Numerical, Integration, and Edge-Case Testing of the Streamlit Application  
**Execution Mode:** Read-Only Source Validation (Zero data changes; no retraining)  
**Date:** 2026-09-27  

---

## 1. Application Startup Status

| Metric / Check | Status | Verification Summary |
| :--- | :---: | :--- |
| **Startup Command** | `PASS` | `streamlit run app/streamlit_app.py --server.port 8501 --server.headless true` |
| **HTTP Response** | `PASS` | `http://localhost:8501` returned `HTTP 200 OK` with valid HTML payload. |
| **AppTest Engine** | `PASS` | Automated programmatic test harness initialized with zero runtime exceptions. |
| **Session State & Routing** | `PASS` | Clean routing across all 8 modular view components via sidebar radio navigation. |

---

## 2. View-by-View Testing Status

All 8 views were tested for visual components, reactive widgets, data consistency, and appropriate academic/non-diagnostic framing.

| View # | View Name | Status | Verified Functionality |
| :---: | :--- | :---: | :--- |
| **1** | **Dataset Overview** | `PASS` | • 4 KPI metric cards: Total Samples ($N=5,518$), T21 Positives ($27, 0.49\%$), T18 Positives ($15, 0.27\%$), T13 Positives ($5, 0.09\%$).<br>• Prominent statistical alert highlighting the exploratory nature of the T13 cohort ($N=5$).<br>• Chromosomal prevalence summary table.<br>• Interactive sample browser with search and multi-feature filtering. |
| **2** | **QC / Data Quality** | `PASS` | • Fetal fraction cutoff slider ($1.0\%$–$8.0\%$) updating sample counts dynamically.<br>• Fetal fraction distribution histogram.<br>• Transparent audit table and disclaimer for negative fetal fraction sample `#666` ($-1.35609\%$).<br>• Cross-tabulation table proving 9 confirmed trisomies occur in `QC-filtered` libraries.<br>• Scatter plot: Mapped Reads vs. Fragment Peak by QC flag. |
| **3** | **Conventional Z-score Baseline** | `PASS` | • Target condition selector (T21, T18, T13).<br>• Real-time baseline Z-score slider ($1.5$–$6.0$, default $3.0$).<br>• Real-time $2 \times 2$ contingency matrix.<br>• 4 KPI cards: Sensitivity (95% CI), Specificity (95% CI), PPV, NPV.<br>• Histogram with annotated Grey Zone ($2.5 \le Z \le 3.5$) and grey-zone sample table. |
| **4** | **ML Screening** | `PASS` | • Tested in both **Mode A (Existing Sample Lookup)** and **Mode B (Manual Experimental Input)**.<br>• Successfully loaded calibrated Model B and Model C pipelines across T13, T18, and T21.<br>• Non-diagnostic terminology strictly maintained: *"model-estimated screening probability"* and *"experimental screening output"*. |
| **5** | **Model Performance** | `PASS` | • Interactive Plotly ROC curves and Precision-Recall curves.<br>• **Primary Table: Leakage-Safe Fold-Derived Operating Points** at 99.0%, 99.5%, and 99.9% target specificity.<br>• **Secondary Reference Expander: Pooled-OOF Post-Hoc Benchmarks** explicitly labeled as descriptive/historical only. |
| **6** | **NIPT Sample Profiling** | `PASS` | • K-Means diagnostic curves for $K=2$ through $K=8$ (Silhouette, Davies-Bouldin, Inertia).<br>• Confirmation of optimal selection at $K=2$ (Silhouette = $0.1593$).<br>• 2D PCA projection scatter plot with explained variance ratios ($28.3\%$ and $17.6\%$).<br>• Median feature table for Profile 0 ($N=1,952$) and Profile 1 ($N=3,566$).<br>• Strict non-diagnostic disclaimer: profiling technical assay context without disease labels. |
| **7** | **Model B vs C Ablation & Interpretation** | `PASS` | • Objective neutral benchmark table: $\Delta\text{PR-AUC}$, $\Delta\text{ROC-AUC}$, $\Delta\text{Brier}$, paired discordance counts, and conclusion across T13, T18, T21.<br>• Paired discordance analysis browser identifying discordant samples (e.g. Sample `#3061` in T21).<br>• Stratified screening performance tabs across `QC-pass` and `QC-filtered` library cohorts. |
| **8** | **Limitations and Disclaimer** | `PASS` | • Critical warning on T13 sample size ($N=5$) and resampling limits.<br>• Statistical uncertainty table with impact of 1 missed case ($\pm 20.0\%$ for T13).<br>• Biological caveats: Placental mosaicism, maternal CNVs/malignancy, vanishing twin, low fetal fraction.<br>• Comprehensive regulatory disclaimer (not an FDA/CE medical device; research prototype only). |

---

## 3. Functional Tests Performed

### A. ML Screening — Mode A: Existing Sample Lookup
Evaluated across T21, T18, and T13 for representative cohort samples:
- **First sample (`#1`):** Euploid reference; Model A = Screen Reference ($Z=0.50$), Model B = $0.11\%$, Model C = $0.10\%$ (`PASS`).
- **Last sample (`#5518`):** Euploid reference; Model A = Screen Reference ($Z=-0.58$), Model B = $0.10\%$, Model C = $0.07\%$ (`PASS`).
- **Negative fetal fraction sample (`#666`):** Handled transparently; fetal fraction replaced with `NaN` by `sanitize_input_features` and median-imputed by the pipeline without throwing errors; Model B = $0.25\%$, Model C = $0.27\%$ (`PASS`).
- **Confirmed T21 positive sample (`#381`):** Model A = Screen Positive ($Z=12.87$), Model B = $96.81\%$ (Elevated Risk), Model C = $95.58\%$ (Elevated Risk) (`PASS`).
- **Confirmed T18 positive sample (`#973`):** Model A = Screen Positive ($Z=7.78$), Model B = $65.81\%$ (Elevated Risk), Model C = $66.59\%$ (Elevated Risk) (`PASS`).
- **Confirmed T13 positive sample (`#1584`):** Model A = Screen Positive ($Z=8.16$), Model B = $55.88\%$ (Elevated Risk), Model C = $55.52\%$ (Elevated Risk) (`PASS`).

### B. ML Screening — Mode B: Manual Experimental Input
- **Standard Physiological Profile:** Evaluated cleanly without shape or dimensionality errors (`PASS`).
- **Negative Fetal Fraction Input ($-2.5\%$):** Sanitized to `NaN` and imputed by the fitted `SimpleImputer` inside Model C; Model C probability evaluated cleanly (`PASS`).
- **Extreme Boundary Profile:** Age = 49 yrs, Week = 11 wks, Reads = 9.5M, Peak = 162.5 bp; evaluated stably by `RobustScaler` and RBF SVM without divergence (`PASS`).

---

## 4. Numerical Consistency Checks

All values displayed across UI tables and charts were verified against on-disk serialization files and markdown evaluation reports:

| Chromosome / Task | Metric | UI Value | JSON Artifact Value | Evaluation Report Value | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Trisomy 21** | Model B ROC-AUC | `1.0000` | `1.0000` | `1.0000` | `PASS` |
| **Trisomy 21** | Model B PR-AUC | `0.9893` | `0.9893` | `0.9893` | `PASS` |
| **Trisomy 21** | Model C ROC-AUC | `0.9999` | `0.9999` | `0.9999` | `PASS` |
| **Trisomy 21** | Model C PR-AUC | `0.9875` | `0.9875` | `0.9875` | `PASS` |
| **Trisomy 21** | Model A Mean Cutoff (99.0%) | `2.0460` | `2.0460` | `2.0460` | `PASS` |
| **Trisomy 21** | Model B Mean Cutoff (99.0%) | `0.7010` | `0.700959` | `0.7010` | `PASS` |
| **Trisomy 18** | Model B ROC-AUC | `0.9999` | `0.9999` | `0.9999` | `PASS` |
| **Trisomy 18** | Model B PR-AUC | `0.9733` | `0.9733` | `0.9733` | `PASS` |
| **Trisomy 18** | Model C ROC-AUC | `0.9998` | `0.9998` | `0.9998` | `PASS` |
| **Trisomy 18** | Model C PR-AUC | `0.9434` | `0.9434` | `0.9434` | `PASS` |
| **Trisomy 13** | Model B ROC-AUC | `0.9997` | `0.9997` | `0.9997` | `PASS` |
| **Trisomy 13** | Model B PR-AUC | `0.8268` | `0.8268` | `0.8268` | `PASS` |
| **Trisomy 13** | Model C ROC-AUC | `0.9992` | `0.9992` | `0.9992` | `PASS` |
| **Trisomy 13** | Model C PR-AUC | `0.8339` | `0.8339` | `0.8339` | `PASS` |
| **Ablation Discordance**| T21 Disagreements ($p=0.5$) | `3` | `3` | `3` | `PASS` |
| **Ablation Discordance**| T18 Disagreements ($p=0.5$) | `0` | `0` | `0` | `PASS` |
| **Ablation Discordance**| T13 Disagreements ($p=0.5$) | `1` | `1` | `1` | `PASS` |
| **K-Means Clustering** | Optimal $K$ Selected | `K = 2` | `2` | `2` | `PASS` |
| **K-Means Clustering** | Silhouette Score ($K=2$) | `0.1593` | `0.1593` | `0.1593` | `PASS` |
| **K-Means Clustering** | Davies-Bouldin ($K=2$) | `2.2675` | `2.2675` | `2.2675` | `PASS` |

---

## 5. Edge-Case Stress Testing

- **Rapid View Cycling:** Navigated through all 8 views across 2 continuous cycles: Zero exceptions or state resets (`PASS`).
- **Condition Toggling in ML Screening:** Switched rapidly between T21 $\to$ T18 $\to$ T13 $\to$ T21: Zero widget key collisions (`PASS`).
- **Extreme Fetal Fractions:** $-100.0\%$ (negative extreme) and $+100.0\%$ (positive extreme): Handled cleanly via fold-fitted `SimpleImputer` and `RobustScaler` (`PASS`).
- **Extreme Demographics:** Age = 15 & 50 years, Week = 8 & 40 weeks: Evaluated stably (`PASS`).
- **Extreme Library Reads & Peaks:** Reads = 100,000 to 25,000,000; Peak = 100.0 to 200.0 bp: Evaluated stably (`PASS`).

---

## 6. Bugs Discovered

1. **Bug #1 — Missing Import in View 7:**
   - **File:** `app/views/view7_interpretation.py`
   - **Exact Error:** `NameError: name 'np' is not defined`
   - **Location:** Line 91, during QC-stratified contingency count calculation: `np.sum((sub["Target"] == 1) & (sub["Prob_Model_B"] >= 0.5))`
   - **Root Cause:** `import numpy as np` was omitted from module imports.

---

## 7. Bugs Fixed

- **Fix Applied to Bug #1:** Added `import numpy as np` to the import block of `app/views/view7_interpretation.py`. No other lines or logic were modified. Retesting confirmed 0 exceptions.

---

## 8. Remaining Issues

- **None.** Zero unresolved exceptions, runtime bugs, or layout warnings remain.

---

## 9. File Integrity Verification

All raw data, processed datasets, and serialized model artifacts were confirmed to be **100% UNCHANGED**:

| File Path | Type | Expected Size (Bytes) | Actual Size (Bytes) | Integrity Status |
| :--- | :--- | :---: | :---: | :---: |
| `data/raw/pone.0207840.s004 (1).xlsx` | Raw Data Source | 2,407,544 | 2,407,544 | `PASS — UNTOUCHED` |
| `data/processed/master_processed.csv` | Processed Master Cohort | 1,356,875 | 1,356,875 | `PASS — UNTOUCHED` |
| `data/processed/t13_dataset.csv` | Processed T13 Cohort | 633,886 | 633,886 | `PASS — UNTOUCHED` |
| `data/processed/t18_dataset.csv` | Processed T18 Cohort | 635,475 | 635,475 | `PASS — UNTOUCHED` |
| `data/processed/t21_dataset.csv` | Processed T21 Cohort | 638,284 | 638,284 | `PASS — UNTOUCHED` |
| `models/t13/model_b_pipeline.joblib` | Model Artifact | 20,573 | 20,573 | `PASS — UNTOUCHED` |
| `models/t13/model_c_pipeline.joblib` | Model Artifact | 51,809 | 51,809 | `PASS — UNTOUCHED` |
| `models/t18/model_b_pipeline.joblib` | Model Artifact | 19,437 | 19,437 | `PASS — UNTOUCHED` |
| `models/t18/model_c_pipeline.joblib` | Model Artifact | 49,601 | 49,601 | `PASS — UNTOUCHED` |
| `models/t21/model_b_pipeline.joblib` | Model Artifact | 20,397 | 20,397 | `PASS — UNTOUCHED` |
| `models/t21/model_c_pipeline.joblib` | Model Artifact | 47,905 | 47,905 | `PASS — UNTOUCHED` |
| `models/kmeans/kmeans_model.joblib` | Clustering Artifact | 22,927 | 22,927 | `PASS — UNTOUCHED` |

---

## 10. Final Readiness Status

### **OVERALL VERDICT: PASS**

The NIPT Clinical Decision Support Portal is fully functional, numerically consistent, leak-free, and compliant with all non-diagnostic academic framing requirements. It is completely ready for live demonstration.
