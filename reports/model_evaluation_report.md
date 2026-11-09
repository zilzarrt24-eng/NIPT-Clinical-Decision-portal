# NIPT Machine Learning Model Evaluation & Ablation Report
**Cohort:** 5,518 clinical samples (`pone.0207840.s004 (1).xlsx`)
**Three-Tier Comparison:** Model A (Z_run Baseline) vs Model B (6 Z-scores RBF-SVM) vs Model C (Extended Multivariate RBF-SVM)

---

## 1. Executive Summary & Objective Ablation Findings

| Aneuploidy | N Positives | Model A PR-AUC | Model B PR-AUC | Model C PR-AUC | ΔPR-AUC (C - B) | Ablation Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **T13** | 5 | 0.9381 | 0.8268 | 0.8339 | +0.0071 | Model B and Model C showed comparable screening performance (ΔPR-AUC within ±0.01). |
| **T18** | 15 | 0.9957 | 0.9733 | 0.9434 | -0.0300 | Model B outperformed Model C in PR-AUC; adding clinical features did not improve separation. |
| **T21** | 27 | 0.9822 | 0.9893 | 0.9875 | -0.0018 | Model B and Model C showed comparable screening performance (ΔPR-AUC within ±0.01). |

> **Important Clinical Terminology Standard:** All model outputs are evaluated as *model-estimated screening probabilities* and *experimental screening outputs*, not as definitive medical diagnoses.

---

## 2. Leakage-Safe Operating Point Evaluation (Primary Methodology)

### Methodological Rationale & Leakage-Safe Design:
- **Issue with Pooled-OOF Cutoffs:** In the initial audit, operating thresholds (99.0%, 99.5%, 99.9% specificity) were identified by sweeping the pooled out-of-fold predictions across all 5,518 samples and evaluating performance on those same samples. This introduced a subtle post-hoc threshold-selection optimism.
- **Corrected Leakage-Safe Procedure:** In each outer cross-validation fold, the threshold targeting empirical specificity $\ge S$ is determined strictly using only training-fold predictions (`X_train`, `y_train`). That threshold is applied once to the held-out validation fold (`X_val`). Counts (TP, FP, TN, FN) are accumulated across the 5 independent validation folds.
- **Validation Independence:** The validation fold has zero influence on threshold selection, ensuring true out-of-sample generalization.

*(Note: Target specificities at 99.0%, 99.5%, and 99.9% are experimental analytical benchmarks, not clinical standards).*  

### T13 Leakage-Safe Fold-Derived Operating Points
| Model | Target Spec | Mean Fold Cutoff | Sensitivity (95% CI) | Achieved Spec (95% CI) | PPV (95% CI) | NPV (95% CI) | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model A (Baseline) | 99.0% | 1.9600 | 100.0% [56.6%, 100.0%] | 99.04% [98.74%, 99.26%] | 8.6% [3.7%, 18.6%] | 100.000% | 5 / 53 / 5460 / 0 |
| Model A (Baseline) | 99.5% | 2.0420 | 100.0% [56.6%, 100.0%] | 99.55% [99.33%, 99.69%] | 16.7% [7.3%, 33.6%] | 100.000% | 5 / 25 / 5488 / 0 |
| Model A (Baseline) | 99.9% | 3.4080 | 80.0% [37.6%, 96.4%] | 99.96% [99.87%, 99.99%] | 66.7% [30.0%, 90.3%] | 99.982% | 4 / 2 / 5511 / 1 |
| Model B (Z-scores) | 99.0% | 0.6878 | 40.0% [11.8%, 76.9%] | 100.00% [99.93%, 100.00%] | 100.0% [34.2%, 100.0%] | 99.946% | 2 / 0 / 5513 / 3 |
| Model B (Z-scores) | 99.5% | 0.6878 | 40.0% [11.8%, 76.9%] | 100.00% [99.93%, 100.00%] | 100.0% [34.2%, 100.0%] | 99.946% | 2 / 0 / 5513 / 3 |
| Model B (Z-scores) | 99.9% | 0.6878 | 40.0% [11.8%, 76.9%] | 100.00% [99.93%, 100.00%] | 100.0% [34.2%, 100.0%] | 99.946% | 2 / 0 / 5513 / 3 |
| Model C (Extended) | 99.0% | 0.6986 | 40.0% [11.8%, 76.9%] | 100.00% [99.93%, 100.00%] | 100.0% [34.2%, 100.0%] | 99.946% | 2 / 0 / 5513 / 3 |
| Model C (Extended) | 99.5% | 0.6986 | 40.0% [11.8%, 76.9%] | 100.00% [99.93%, 100.00%] | 100.0% [34.2%, 100.0%] | 99.946% | 2 / 0 / 5513 / 3 |
| Model C (Extended) | 99.9% | 0.6986 | 40.0% [11.8%, 76.9%] | 100.00% [99.93%, 100.00%] | 100.0% [34.2%, 100.0%] | 99.946% | 2 / 0 / 5513 / 3 |

### T18 Leakage-Safe Fold-Derived Operating Points
| Model | Target Spec | Mean Fold Cutoff | Sensitivity (95% CI) | Achieved Spec (95% CI) | PPV (95% CI) | NPV (95% CI) | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model A (Baseline) | 99.0% | 1.9680 | 100.0% [79.6%, 100.0%] | 99.04% [98.74%, 99.26%] | 22.1% [13.8%, 33.3%] | 100.000% | 15 / 53 / 5450 / 0 |
| Model A (Baseline) | 99.5% | 2.1600 | 100.0% [79.6%, 100.0%] | 99.60% [99.40%, 99.74%] | 40.5% [26.3%, 56.5%] | 100.000% | 15 / 22 / 5481 / 0 |
| Model A (Baseline) | 99.9% | 2.6360 | 93.3% [70.2%, 98.8%] | 99.93% [99.81%, 99.97%] | 77.8% [54.8%, 91.0%] | 99.982% | 14 / 4 / 5499 / 1 |
| Model B (Z-scores) | 99.0% | 0.6617 | 66.7% [41.7%, 84.8%] | 100.00% [99.93%, 100.00%] | 100.0% [72.2%, 100.0%] | 99.909% | 10 / 0 / 5503 / 5 |
| Model B (Z-scores) | 99.5% | 0.6617 | 66.7% [41.7%, 84.8%] | 100.00% [99.93%, 100.00%] | 100.0% [72.2%, 100.0%] | 99.909% | 10 / 0 / 5503 / 5 |
| Model B (Z-scores) | 99.9% | 0.6617 | 66.7% [41.7%, 84.8%] | 100.00% [99.93%, 100.00%] | 100.0% [72.2%, 100.0%] | 99.909% | 10 / 0 / 5503 / 5 |
| Model C (Extended) | 99.0% | 0.6662 | 66.7% [41.7%, 84.8%] | 100.00% [99.93%, 100.00%] | 100.0% [72.2%, 100.0%] | 99.909% | 10 / 0 / 5503 / 5 |
| Model C (Extended) | 99.5% | 0.6662 | 66.7% [41.7%, 84.8%] | 100.00% [99.93%, 100.00%] | 100.0% [72.2%, 100.0%] | 99.909% | 10 / 0 / 5503 / 5 |
| Model C (Extended) | 99.9% | 0.6662 | 66.7% [41.7%, 84.8%] | 100.00% [99.93%, 100.00%] | 100.0% [72.2%, 100.0%] | 99.909% | 10 / 0 / 5503 / 5 |

### T21 Leakage-Safe Fold-Derived Operating Points
| Model | Target Spec | Mean Fold Cutoff | Sensitivity (95% CI) | Achieved Spec (95% CI) | PPV (95% CI) | NPV (95% CI) | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model A (Baseline) | 99.0% | 2.0460 | 100.0% [87.5%, 100.0%] | 98.98% [98.68%, 99.21%] | 32.5% [23.4%, 43.2%] | 100.000% | 27 / 56 / 5435 / 0 |
| Model A (Baseline) | 99.5% | 2.2100 | 100.0% [87.5%, 100.0%] | 99.47% [99.24%, 99.63%] | 48.2% [35.7%, 61.0%] | 100.000% | 27 / 29 / 5462 / 0 |
| Model A (Baseline) | 99.9% | 2.5840 | 92.6% [76.6%, 97.9%] | 99.91% [99.79%, 99.96%] | 83.3% [66.4%, 92.7%] | 99.964% | 25 / 5 / 5486 / 2 |
| Model B (Z-scores) | 99.0% | 0.7010 | 74.1% [55.3%, 86.8%] | 100.00% [99.93%, 100.00%] | 100.0% [83.9%, 100.0%] | 99.873% | 20 / 0 / 5491 / 7 |
| Model B (Z-scores) | 99.5% | 0.7010 | 74.1% [55.3%, 86.8%] | 100.00% [99.93%, 100.00%] | 100.0% [83.9%, 100.0%] | 99.873% | 20 / 0 / 5491 / 7 |
| Model B (Z-scores) | 99.9% | 0.7010 | 74.1% [55.3%, 86.8%] | 100.00% [99.93%, 100.00%] | 100.0% [83.9%, 100.0%] | 99.873% | 20 / 0 / 5491 / 7 |
| Model C (Extended) | 99.0% | 0.6882 | 74.1% [55.3%, 86.8%] | 99.98% [99.90%, 100.00%] | 95.2% [77.3%, 99.2%] | 99.873% | 20 / 1 / 5490 / 7 |
| Model C (Extended) | 99.5% | 0.6882 | 74.1% [55.3%, 86.8%] | 99.98% [99.90%, 100.00%] | 95.2% [77.3%, 99.2%] | 99.873% | 20 / 1 / 5490 / 7 |
| Model C (Extended) | 99.9% | 0.6882 | 74.1% [55.3%, 86.8%] | 99.98% [99.90%, 100.00%] | 95.2% [77.3%, 99.2%] | 99.873% | 20 / 1 / 5490 / 7 |

---

### Comparative Reference: Previous Pooled-OOF Post-Hoc Analytical Thresholds
*(Retained strictly for methodological transparency and historical comparison; reflects post-hoc threshold selection on pooled OOF data)*

#### T13 Pooled-OOF Post-Hoc Benchmarks (Descriptive Only)
| Model | Target Spec | Cutoff | Sensitivity (95% CI) | Specificity | PPV (95% CI) | NPV | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model A (Baseline) | 99.0% | 1.9600 | 100.0% [56.6%, 100.0%] | 99.00% | 8.3% [3.6%, 18.1%] | 100.000% | 5 / 55 / 5458 / 0 |
| Model A (Baseline) | 99.5% | 2.0500 | 100.0% [56.6%, 100.0%] | 99.51% | 15.6% [6.9%, 31.8%] | 100.000% | 5 / 27 / 5486 / 0 |
| Model A (Baseline) | 99.9% | 2.7900 | 100.0% [56.6%, 100.0%] | 99.96% | 71.4% [35.9%, 91.8%] | 100.000% | 5 / 2 / 5511 / 0 |
| Model B (Z-scores) | 99.0% | 0.0250 | 100.0% [56.6%, 100.0%] | 99.85% | 38.5% [17.7%, 64.5%] | 100.000% | 5 / 8 / 5505 / 0 |
| Model B (Z-scores) | 99.5% | 0.0250 | 100.0% [56.6%, 100.0%] | 99.85% | 38.5% [17.7%, 64.5%] | 100.000% | 5 / 8 / 5505 / 0 |
| Model B (Z-scores) | 99.9% | 0.4264 | 80.0% [37.6%, 96.4%] | 99.98% | 80.0% [37.6%, 96.4%] | 99.982% | 4 / 1 / 5512 / 1 |
| Model C (Extended) | 99.0% | 0.0082 | 100.0% [56.6%, 100.0%] | 99.60% | 18.5% [8.2%, 36.7%] | 100.000% | 5 / 22 / 5491 / 0 |
| Model C (Extended) | 99.5% | 0.0082 | 100.0% [56.6%, 100.0%] | 99.60% | 18.5% [8.2%, 36.7%] | 100.000% | 5 / 22 / 5491 / 0 |
| Model C (Extended) | 99.9% | 0.5304 | 80.0% [37.6%, 96.4%] | 100.00% | 100.0% [51.0%, 100.0%] | 99.982% | 4 / 0 / 5513 / 1 |

#### T18 Pooled-OOF Post-Hoc Benchmarks (Descriptive Only)
| Model | Target Spec | Cutoff | Sensitivity (95% CI) | Specificity | PPV (95% CI) | NPV | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model A (Baseline) | 99.0% | 1.9600 | 100.0% [79.6%, 100.0%] | 99.00% | 21.4% [13.4%, 32.4%] | 100.000% | 15 / 55 / 5448 / 0 |
| Model A (Baseline) | 99.5% | 2.2100 | 100.0% [79.6%, 100.0%] | 99.64% | 42.9% [28.0%, 59.1%] | 100.000% | 15 / 20 / 5483 / 0 |
| Model A (Baseline) | 99.9% | 2.6200 | 100.0% [79.6%, 100.0%] | 99.95% | 83.3% [60.8%, 94.2%] | 100.000% | 15 / 3 / 5500 / 0 |
| Model B (Z-scores) | 99.0% | 0.0163 | 100.0% [79.6%, 100.0%] | 99.91% | 75.0% [53.1%, 88.8%] | 100.000% | 15 / 5 / 5498 / 0 |
| Model B (Z-scores) | 99.5% | 0.0163 | 100.0% [79.6%, 100.0%] | 99.91% | 75.0% [53.1%, 88.8%] | 100.000% | 15 / 5 / 5498 / 0 |
| Model B (Z-scores) | 99.9% | 0.0163 | 100.0% [79.6%, 100.0%] | 99.91% | 75.0% [53.1%, 88.8%] | 100.000% | 15 / 5 / 5498 / 0 |
| Model C (Extended) | 99.0% | 0.0105 | 100.0% [79.6%, 100.0%] | 99.89% | 71.4% [50.0%, 86.2%] | 100.000% | 15 / 6 / 5497 / 0 |
| Model C (Extended) | 99.5% | 0.0105 | 100.0% [79.6%, 100.0%] | 99.89% | 71.4% [50.0%, 86.2%] | 100.000% | 15 / 6 / 5497 / 0 |
| Model C (Extended) | 99.9% | 0.0136 | 93.3% [70.2%, 98.8%] | 99.91% | 73.7% [51.2%, 88.2%] | 99.982% | 14 / 5 / 5498 / 1 |

#### T21 Pooled-OOF Post-Hoc Benchmarks (Descriptive Only)
| Model | Target Spec | Cutoff | Sensitivity (95% CI) | Specificity | PPV (95% CI) | NPV | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model A (Baseline) | 99.0% | 2.0500 | 100.0% [87.5%, 100.0%] | 99.05% | 34.2% [24.7%, 45.2%] | 100.000% | 27 / 52 / 5439 / 0 |
| Model A (Baseline) | 99.5% | 2.2200 | 100.0% [87.5%, 100.0%] | 99.54% | 51.9% [38.7%, 64.9%] | 100.000% | 27 / 25 / 5466 / 0 |
| Model A (Baseline) | 99.9% | 2.5500 | 92.6% [76.6%, 97.9%] | 99.91% | 83.3% [66.4%, 92.7%] | 99.964% | 25 / 5 / 5486 / 2 |
| Model B (Z-scores) | 99.0% | 0.1243 | 100.0% [87.5%, 100.0%] | 99.98% | 96.4% [82.3%, 99.4%] | 100.000% | 27 / 1 / 5490 / 0 |
| Model B (Z-scores) | 99.5% | 0.1243 | 100.0% [87.5%, 100.0%] | 99.98% | 96.4% [82.3%, 99.4%] | 100.000% | 27 / 1 / 5490 / 0 |
| Model B (Z-scores) | 99.9% | 0.1243 | 100.0% [87.5%, 100.0%] | 99.98% | 96.4% [82.3%, 99.4%] | 100.000% | 27 / 1 / 5490 / 0 |
| Model C (Extended) | 99.0% | 0.1109 | 100.0% [87.5%, 100.0%] | 99.98% | 96.4% [82.3%, 99.4%] | 100.000% | 27 / 1 / 5490 / 0 |
| Model C (Extended) | 99.5% | 0.1109 | 100.0% [87.5%, 100.0%] | 99.98% | 96.4% [82.3%, 99.4%] | 100.000% | 27 / 1 / 5490 / 0 |
| Model C (Extended) | 99.9% | 0.1109 | 100.0% [87.5%, 100.0%] | 99.98% | 96.4% [82.3%, 99.4%] | 100.000% | 27 / 1 / 5490 / 0 |

---

## 3. Statistical Uncertainty & Rare Class Limitations

- **Trisomy 13 ($N=5$ Positives):** PRELIMINARY AND EXPLORATORY ONLY. Repeated cross-validation resampling does NOT increase the number of independent positive samples. Each missed case shifts measured sensitivity by 20.0%.
- **Trisomy 18 ($N=15$ Positives):** Small positive cohort; 3 positive cases per fold.
- **Trisomy 21 ($N=27$ Positives):** Moderate positive cohort; 5–6 positive cases per fold.

---

## 4. Unsupervised NIPT Sample Profiling (K-Means)
- **Optimal Clusters Selected:** K = 2
- **Selection Criteria:** K=2 achieved the highest Silhouette Score (0.1593) with a Davies-Bouldin Index of 2.2675.
- **Cluster Distribution:** {"0": 1952, "1": 3566}
- **Non-Diagnostic Policy:** Clusters reflect library cfDNA concentration, sequencing depth, and patient demographics. No disease diagnoses are assigned.
