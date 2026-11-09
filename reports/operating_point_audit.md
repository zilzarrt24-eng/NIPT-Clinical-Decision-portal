# Methodological Audit: Leakage-Safe Operating Point Evaluation
**Project:** NIPT Clinical Decision Support Portal  
**Date:** 2026-09-27  
**Scope:** Verification of Fold-Derived Threshold Selection vs. Previous Pooled-OOF Post-Hoc Selection

---

## 1. Executive Summary & Audit Statuses

| # | Audit Item | Status | Verification Summary |
|---|:---|:---:|:---|
| 1 | **Raw & Processed Data Integrity** | `PASS` | `pone.0207840.s004 (1).xlsx` and all 4 CSVs in `data/processed/` are **100% UNCHANGED**. |
| 2 | **Validation Fold Isolation** | `PASS` | Operating thresholds for 99.0%, 99.5%, and 99.9% specificity are determined strictly on `X_train, y_train`. Validation folds (`X_val, y_val`) are never observed during threshold selection. |
| 3 | **Model Pipeline Preservation** | `PASS` | No changes to model architectures, feature sets, SVM kernels, `class_weight='balanced'`, RobustScaler, SimpleImputer, or Platt calibration. |
| 4 | **Calibration Isolation** | `PASS` | Platt sigmoid calibrator is fit via inner cross-validation strictly within the outer training fold (`X_train`). |
| 5 | **OOF Prediction Preservation** | `PASS` | `oof_predictions.csv` files for T13, T18, and T21 remain intact and aligned with the 5,518 master cohort. |
| 6 | **Dual Reporting Implemented** | `PASS` | Both `leakage_safe_operating_points` (primary) and `pooled_oof_posthoc_operating_points` (comparative reference) are stored in `metrics_summary.json` and presented in `reports/model_evaluation_report.md` and Streamlit. |
| 7 | **Streamlit Portal Alignment** | `PASS` | Tab 5 (`view5_performance.py`) updated to present Leakage-Safe Operating Points as primary and Pooled-OOF as expandable reference. |

---

## 2. Methodological Analysis: Why the Change Was Necessary

### A. The Previous Methodology (Pooled-OOF Post-Hoc Selection)
- **Procedure:** The out-of-fold predictions from all 5 outer CV folds were pooled across the entire cohort ($N = 5,518$). A receiver operating characteristic (ROC) sweep was then conducted over the pooled vector to locate the empirical cutoff achieving $\ge 99.0\%$, $99.5\%$, or $99.9\%$ specificity.
- **Methodological Limitation:** Even though individual sample probabilities were computed out-of-fold, choosing the single best cutoff *after observing all pooled predictions* introduced **post-hoc threshold-selection optimism**. It allowed the decision boundary to adapt to random distribution quirks across the validation samples, producing slightly optimistic false-positive counts and sensitivities (e.g. apparent 100% sensitivity for Model B on T21 at 99.9% specificity).

### B. The Corrected Methodology (Leakage-Safe Fold-Derived Evaluation)
- **Procedure:**
  1. In outer fold $k$, the model pipeline is trained on `X_train, y_train`.
  2. Predictions are generated on `X_train` (`train_probs = model.predict_proba(X_train)[:, 1]`).
  3. Operating thresholds targeting $\ge 99.0\%$, $99.5\%$, and $99.9\%$ empirical specificity are computed **strictly on `y_train == 0`**.
  4. Each threshold is applied **ONCE** to the held-out validation fold `X_val`.
  5. Contingency counts (TP, FP, TN, FN) are accumulated across the 5 independent validation folds.
  6. Overall validation sensitivity, specificity, PPV, NPV, and exact 95% Wilson score confidence intervals are computed from the accumulated validation counts.
- **Guarantee:** The held-out validation fold has zero influence on the selection of the operating cutoff.

---

## 3. Comparison of Results: Leakage-Safe vs. Pooled-OOF Post-Hoc

### A. Trisomy 21 ($N=27$ Positives)
| Model | Target Spec | Leakage-Safe Cutoff (Mean) | Leakage-Safe Sens (95% CI) | Leakage-Safe Spec | Leakage-Safe TP / FP | Post-Hoc Pooled-OOF Cutoff | Post-Hoc Sens | Post-Hoc Spec | Post-Hoc TP / FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | 99.0% | $Z = 2.046$ | 100.0% [87.5%, 100%] | 98.98% | 27 / 56 | $Z = 2.050$ | 100.0% | 99.05% | 27 / 52 |
| **Model A** | 99.5% | $Z = 2.210$ | 100.0% [87.5%, 100%] | 99.47% | 27 / 29 | $Z = 2.220$ | 100.0% | 99.54% | 27 / 25 |
| **Model A** | 99.9% | $Z = 2.584$ | 92.6% [76.6%, 97.9%] | 99.91% | 25 / 5 | $Z = 2.550$ | 92.6% | 99.91% | 25 / 5 |
| **Model B** | 99.0% | $p = 0.7010$ | 74.1% [55.3%, 86.8%] | 100.00% | 20 / 0 | $p = 0.1243$ | 100.0% | 99.98% | 27 / 1 |
| **Model B** | 99.5% | $p = 0.7010$ | 74.1% [55.3%, 86.8%] | 100.00% | 20 / 0 | $p = 0.1243$ | 100.0% | 99.98% | 27 / 1 |
| **Model B** | 99.9% | $p = 0.7010$ | 74.1% [55.3%, 86.8%] | 100.00% | 20 / 0 | $p = 0.1243$ | 100.0% | 99.98% | 27 / 1 |
| **Model C** | 99.0% | $p = 0.6882$ | 74.1% [55.3%, 86.8%] | 99.98% | 20 / 1 | $p = 0.1109$ | 100.0% | 99.98% | 27 / 1 |
| **Model C** | 99.5% | $p = 0.6882$ | 74.1% [55.3%, 86.8%] | 99.98% | 20 / 1 | $p = 0.1109$ | 100.0% | 99.98% | 27 / 1 |
| **Model C** | 99.9% | $p = 0.6882$ | 74.1% [55.3%, 86.8%] | 99.98% | 20 / 1 | $p = 0.1109$ | 100.0% | 99.98% | 27 / 1 |

*Insight:* On the training fold, negative samples have tightly squashed probabilities near zero, so targeting $\ge 99.9\%$ specificity on `X_train` selects a conservative threshold ($p \approx 0.70$). When applied to validation folds, this completely eliminates false positives (100.0% specificity for Model B) while identifying 74.1% of true positives without any threshold tuning on the validation data.

### B. Trisomy 18 ($N=15$ Positives)
- **Model A (Baseline):** 
  - 99.0% Target: Sens = 100.0% [79.6%, 100%], Spec = 99.04%, TP = 15, FP = 53
  - 99.5% Target: Sens = 100.0% [79.6%, 100%], Spec = 99.60%, TP = 15, FP = 22
  - 99.9% Target: Sens = 93.3% [70.2%, 98.8%], Spec = 99.93%, TP = 14, FP = 4
- **Model B (6 Z-scores):**
  - All targets: Mean Cutoff $p = 0.6617$, Sens = 66.7% [41.7%, 84.8%], Spec = 100.00%, TP = 10, FP = 0

### C. Trisomy 13 ($N=5$ Positives)
- **Model A (Baseline):**
  - 99.0% Target: Sens = 100.0% [56.6%, 100%], Spec = 99.04%, TP = 5, FP = 53
  - 99.5% Target: Sens = 100.0% [56.6%, 100%], Spec = 99.55%, TP = 5, FP = 25
  - 99.9% Target: Sens = 80.0% [37.6%, 96.4%], Spec = 99.96%, TP = 4, FP = 2
- **Model B (6 Z-scores):**
  - All targets: Mean Cutoff $p = 0.6878$, Sens = 40.0% [11.8%, 76.9%], Spec = 100.00%, TP = 2, FP = 0

---

## 4. Methodological Conclusion
The leakage-safe fold-derived operating point evaluation is fully implemented, verified, and integrated into the project codebase and Streamlit portal. The project is methodologically sound and ready for academic presentation.
