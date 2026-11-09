# Preprocessing & Data Validation Report
**Dataset:** Non-Invasive Prenatal Testing (NIPT) Multi-Sheet Cohort (`pone.0207840.s004 (1).xlsx`)  
**Pipeline Execution Timestamp:** Generated reproducibly via `src/preprocessing.py`  
**Raw Data Integrity:** Original raw XLSX preserved completely untouched.

---

## 1. Cohort & Sample Validation Summary

- **Total Master Samples:** `5,518`
- **Sample ID Integrity:** Validated unique across all three sheets (`chr13`, `chr18`, `chr21`).
- **Cross-Sheet Alignment:** 100% concordance on `Sample ID` ordering and shared metadata (`age`, `week`, `reads`, `fetal`, `peak`, `costday`, `QC`).
- **Missing Values:** `0` missing values across all columns in raw and processed datasets.

---

## 2. Aneuploidy Target Class Distribution

| Condition | Target Column | Negative Cases (`0`) | Positive Cases (`1`) | Positive Prevalence (%) | Statistical Sample Reliability Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trisomy 21 (Down)** | `T21_Target` | 5,491 | **27** | 0.489% | Moderately small positive cohort; suitable for 5-fold stratified CV. |
| **Trisomy 18 (Edwards)** | `T18_Target` | 5,503 | **15** | 0.272% | Very small positive cohort (3 positives per fold in 5-fold CV). |
| **Trisomy 13 (Patau)** | `T13_Target` | 5,513 | **5** | 0.091% | **Extremely rare (N=5 positives).** Cross-validation metrics are subject to extreme statistical variance (1 misclassification shifts sensitivity by 20%). Documented with confidence intervals. |

---

## 3. Quality Control (QC) Distribution & Target Cross-Tabulation

| Library QC Status | Sample Count (% Cohort) | T13 Positives | T18 Positives | T21 Positives | Total Confirmed Trisomies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QC-pass** | 4752 (86.1%) | 4 | 11 | 23 |
| **QC-filtered** | 766 (13.9%) | 1 | 4 | 4 |

> **Critical Decision Note:** 9 out of 47 confirmed trisomy cases fall within the `QC-filtered` subset. Filtering out QC-failed samples prior to modeling would discard ~19% of true positive clinical cases. The `QC` column is preserved in the master dataset for stratified evaluation and quality gating, but is **strictly excluded as an SVM input feature**.

---

## 4. Invalid Fetal-Fraction Inspection & Handling

- **Invalid Count ($< 0.0\%$):** `1` sample.
- **Samples with Fetal Fraction $< 4.0\%$:** `28` samples.

### Detailed Audit of Invalid Sample:
| Sample ID | Raw Fetal Fraction | Library QC | Fragment Peak | Reads | Maternal Age | Gestational Week |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `#666` | `-1.35609%` | `QC-filtered` | `132.47 bp` | `3382248` | `32 yrs` | `17 wks` |

### Imputation & Modeling Policy:
1. **Preservation:** The raw value (`-1.35609%`) is preserved in `master_processed.csv`.
2. **Quality Audit Flag:** An explicit binary flag `fetal_invalid_flag = 1` is recorded.
3. **Model Feature Exclusion:** Per design constraints, `fetal_invalid_flag` is **strictly excluded** from SVM predictor sets.
4. **Transparent Numeric Imputation:** For models requiring non-negative numeric inputs, the preprocessing module exposes a fold-fit median imputer (computed strictly on training folds) to substitute invalid values without data leakage.

---

## 5. Candidate Feature Summary Statistics (Shared Metadata)

| Feature | Min | 25% | Median (50%) | 75% | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fetal` | -1.36 | 8.50 | 10.66 | 13.28 | 34.12 | 11.10 ± 3.61 |
| `age` | 15.00 | 27.00 | 33.00 | 36.00 | 47.00 | 31.83 ± 5.81 |
| `week` | 8.00 | 15.00 | 17.00 | 19.00 | 37.00 | 17.19 ± 3.20 |
| `reads` | 2012449.00 | 3317298.25 | 3761972.50 | 4175854.00 | 18293449.00 | 3796499.83 ± 801116.17 |
| `peak` | 105.68 | 145.92 | 149.13 | 151.63 | 156.22 | 148.13 ± 5.43 |

---

## 6. Output Files and Artifacts Generated

1. `data/processed/master_processed.csv` (5,518 rows × 39 columns)
2. `data/processed/t13_dataset.csv` (5,518 rows × 19 columns)
3. `data/processed/t18_dataset.csv` (5,518 rows × 19 columns)
4. `data/processed/t21_dataset.csv` (5,518 rows × 19 columns)
5. `reports/preprocessing_report.md` (This document)
