# Independent Post-Training Audit Report: NIPT Clinical Decision Portal
**Audit Execution Date:** 2026-09-27  
**Scope:** Read-Only Verification of Artifacts, OOF Predictions, Metric Reproducibility, CV Architecture, Calibration, Operating Points, K-Means Profiling, and Streamlit Integration.

---

### A. Artifact Integrity: [PASS]
- **Models Directory:** All 10 expected model artifacts exist under `models/t13/`, `models/t18/`, `models/t21/`, and `models/kmeans/`.
- **Deserialization:** Successfully deserialized all `.joblib` files via `joblib.load()`.
- **Pipeline Components:** Confirmed that Model B contains `RobustScaler` + `SVC(probability=False, class_weight='balanced')` wrapped in `CalibratedClassifierCV(method='sigmoid')`. Model C contains `SimpleImputer(strategy='median')` + `RobustScaler` + `SVC(probability=False, class_weight='balanced')` wrapped in `CalibratedClassifierCV(method='sigmoid')`.
- **Integrity Status:** Raw XLSX and all 4 processed CSV datasets in `data/processed/` are 100% untouched.

---

### B. OOF Prediction Integrity: [PASS]
- **Row Counts:** Exactly 5,518 rows for T13, T18, and T21.
- **Sample IDs:** 100% unique, zero duplicates, and strictly aligned with the master dataset.
- **Ground Truth Target Alignment:** Exactly matches processed datasets:
  - T13: 5 positives, 5,513 negatives.
  - T18: 15 positives, 5,503 negatives.
  - T21: 27 positives, 5,491 negatives.
- **Completeness:** 0 missing values across all prediction columns.

---

### C. Metric Reproduction: [PASS]
All metrics were recalculated directly from `oof_predictions.csv` and compared against `metrics_summary.json`:
- **PR-AUC:** Recalculated values match reported values exactly ($\Delta = 0.00 \times 10^0$).
- **ROC-AUC & Partial ROC-AUC ($\text{Spec} \ge 90\%$):** 100% exact reproduction ($\Delta = 0.00 \times 10^0$).
- **Brier Scores:** Recalculated values match reported values to within floating point precision ($\Delta < 10^{-18}$).
- **Confusion Matrices ($p=0.5$):** 100% identical TP, FP, TN, and FN counts across all models and tasks.

---

### D. Cross-Validation Methodology: [PASS with STATISTICAL NOTE]
- **Outer Split:** Stratified 5-Fold Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Leakage Prevention:** Imputation medians and `RobustScaler` IQR parameters are fit strictly inside `X_train` within each outer fold. Zero validation data leaks into preprocessing.
- **T13 Statistical Reality:** T13 outer folds contain 4 train positives and 1 validation positive. Repeated resampling does NOT create new independent biological cases. Wilson 95% confidence intervals are documented.

---

### E. Calibration Methodology: [PASS]
- **Mechanism:** Explicit Platt scaling (`CalibratedClassifierCV(method='sigmoid')`) fitting a 2-parameter logistic sigmoid $P(y=1|d) = \frac{1}{1 + \exp(A \cdot d + B)}$.
- **Inner CV:** Calibration is fit via internal cross-validation strictly within the outer training fold (`X_train`).
- **Isolation:** Outer validation observations are completely excluded from calibration fitting.

---

### F. Operating-Point Methodology: [WARNING - OPTIMISTIC THRESHOLD BIAS]
- **Finding:** Operating points for 99.0%, 99.5%, and 99.9% specificity are selected on the **pooled OOF prediction set** and evaluated on that same set.
- **Methodological Concern:** Selecting a threshold on a validation dataset and evaluating performance on the same dataset introduces a subtle optimistic bias (post-hoc cutoff selection).
- **Recommendation:** In academic presentation, clearly state that these operating points are descriptive analytical cutoffs on the pooled OOF set, or implement nested fold-level threshold selection in future revisions.

---

### G. Model A Baseline Audit: [PASS]
- **Independent Evaluation:** `Z_run` is evaluated directly from raw laboratory dosage data without machine learning.
- **Reproducibility:** PR-AUC and ROC-AUC match reported baseline numbers exactly.

---

### H. Model B vs. Model C Ablation Audit: [PASS]
- **Identical Samples:** Both models were evaluated on the identical outer folds and sample rows.
- **Recalculated Deltas:**
  - T13: $\Delta\text{PR-AUC} = +0.0071$ (Comparable within $\pm 0.01$).
  - T18: $\Delta\text{PR-AUC} = -0.0300$ (**Model B outperformed Model C**; adding clinical features degraded precision).
  - T21: $\Delta\text{PR-AUC} = -0.0018$ (Comparable within $\pm 0.01$).
- **Disagreements:** T13 had 1 discordant sample (`#4858`), T18 had 0, and T21 had 3 (`#1456`, `#3061`, `#3374`). In sample `#3061`, Model C produced a false negative due to clinical feature noise.

---

### I. K-Means Sample Profiling Audit: [PASS]
- **K Selection:** $K=2$ was selected via peak silhouette score ($0.1593$).
- **Profiles:** Cluster 0 ($N=1,952$, high fetal fraction, compact fragment peak) and Cluster 1 ($N=3,566$, moderate fetal fraction, extended fragment peak).
- **Non-Diagnostic Compliance:** No disease labels or outcome targets were used. PCA projection has exactly 5,518 rows.

---

### J. Streamlit Application Audit: [PASS]
- **Import Verification:** All 8 views import without error.
- **Path Verification:** All views load serialized `.joblib` pipelines, CSVs, and JSONs from valid paths.
- **Terminology:** Strictly uses approved research terminology (*"model-estimated screening probability"*, *"experimental screening output"*).

---

### K. Issues Found
1. **Operating Point Threshold Selection on Pooled OOF:** Descriptive post-hoc selection creates a minor optimistic threshold bias.
2. **K-Means Cluster Modesty:** Silhouette score ($0.1593$) reflects continuous biological variation rather than discrete clusters.

---

### L. Recommended Fixes for Final Presentation
1. Transparently disclose in Tab 5/Tab 8 that operating point thresholds are descriptive analytical benchmarks chosen on pooled OOF predictions.
2. Emphasize that Model B (6 Z-scores) is the preferred ML model because Model C's demographic features did not improve discrimination and degraded T18 performance.
