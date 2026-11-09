"""
Statistical and Clinical Metric Utilities for NIPT Evaluation
============================================================
Provides exact Wilson score confidence intervals, clinical screening metric
calculations, and statistical formatting helpers.
"""

import math
import numpy as np
from typing import Tuple, Dict, Any, List


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes exact Wilson score confidence interval for binomial proportions.
    Essential for small positive cohorts (e.g., T13 with N=5, T18 with N=15)
    where normal approximations completely break down.
    
    Returns:
        (lower_bound, upper_bound) in range [0.0, 1.0]
    """
    if total == 0:
        return (0.0, 0.0)
    
    # Critical value z for given confidence level
    # 0.95 -> 1.95996; 0.90 -> 1.64485; 0.99 -> 2.57583
    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    else:
        # Standard normal inverse approximation
        from scipy.stats import norm
        z = norm.ppf(1 - (1 - confidence) / 2)

    p_hat = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre_adjusted_probability = p_hat + z2 / (2 * total)
    adjusted_std_dev = math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * total)) / total)
    
    lower = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std_dev) / denominator
    
    return (max(0.0, float(lower)), min(1.0, float(upper)))


def compute_contingency_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Any]:
    """
    Calculates sensitivity, specificity, PPV, NPV, and exact 95% Wilson CIs.
    """
    total_pos = tp + fn
    total_neg = tn + fp
    total_pred_pos = tp + fp
    total_pred_neg = tn + fn

    sensitivity = tp / total_pos if total_pos > 0 else 0.0
    specificity = tn / total_neg if total_neg > 0 else 0.0
    ppv = tp / total_pred_pos if total_pred_pos > 0 else 0.0
    npv = tn / total_pred_neg if total_pred_neg > 0 else 0.0

    sens_ci = wilson_score_interval(tp, total_pos, 0.95)
    spec_ci = wilson_score_interval(tn, total_neg, 0.95)
    ppv_ci = wilson_score_interval(tp, total_pred_pos, 0.95) if total_pred_pos > 0 else (0.0, 0.0)
    npv_ci = wilson_score_interval(tn, total_pred_neg, 0.95) if total_pred_neg > 0 else (0.0, 0.0)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": float(sensitivity),
        "sensitivity_ci_95": [float(sens_ci[0]), float(sens_ci[1])],
        "specificity": float(specificity),
        "specificity_ci_95": [float(spec_ci[0]), float(spec_ci[1])],
        "ppv": float(ppv),
        "ppv_ci_95": [float(ppv_ci[0]), float(ppv_ci[1])],
        "npv": float(npv),
        "npv_ci_95": [float(npv_ci[0]), float(npv_ci[1])],
    }
