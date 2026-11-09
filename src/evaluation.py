"""
Evaluation, Cross-Validation, and Model Ablation Module
======================================================
Provides Stratified K-Fold CV, out-of-fold pooling, experimental operating point analysis,
and objective Model B vs Model C ablation benchmarking.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    roc_curve
)

from src.models import (
    get_candidate_features,
    build_calibrated_pipeline,
    sanitize_input_features
)
from src.utils import compute_contingency_metrics


def evaluate_model_a_baseline(
    df: pd.DataFrame,
    aneuploidy: str,
    operating_specificities: List[float] = [0.990, 0.995, 0.999]
) -> Dict[str, Any]:
    """
    Evaluates univariate screening performance of conventional Z_run alone.
    """
    prefix = f"{aneuploidy}_"
    z_col = f"{prefix}Z_run"
    target_col = f"{prefix}Target"

    y_true = df[target_col].values
    z_scores = df[z_col].values

    # Conventional Z_run ROC-AUC, Partial ROC-AUC (max_fpr=0.10, Specificity >= 90%), and PR-AUC
    fpr, tpr, roc_thresh = roc_curve(y_true, z_scores)
    roc_auc = float(roc_auc_score(y_true, z_scores))
    try:
        partial_roc_auc = float(roc_auc_score(y_true, z_scores, max_fpr=0.10))
    except Exception:
        partial_roc_auc = float("nan")

    precision, recall, pr_thresh = precision_recall_curve(y_true, z_scores)
    pr_auc = float(auc(recall, precision))

    # Standard Z = 3.0 cutoff evaluation
    z3_pred = (z_scores >= 3.0).astype(int)
    tp_z3 = int(np.sum((y_true == 1) & (z3_pred == 1)))
    fp_z3 = int(np.sum((y_true == 0) & (z3_pred == 1)))
    tn_z3 = int(np.sum((y_true == 0) & (z3_pred == 0)))
    fn_z3 = int(np.sum((y_true == 1) & (z3_pred == 0)))
    metrics_z3 = compute_contingency_metrics(tp_z3, fp_z3, tn_z3, fn_z3)

    # 1. Previous Pooled-OOF Post-Hoc Analytical Thresholds (retained for comparison/transparency)
    pooled_oof_posthoc_ops = {}
    for target_spec in operating_specificities:
        spec_arr = 1 - fpr
        valid_indices = np.where(spec_arr >= target_spec)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            cutoff = float(roc_thresh[idx])
        else:
            cutoff = float(np.percentile(z_scores[y_true == 0], target_spec * 100))

        y_pred_op = (z_scores >= cutoff).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred_op == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred_op == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred_op == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred_op == 0)))
        op_metrics = compute_contingency_metrics(tp, fp, tn, fn)
        op_metrics["cutoff_z"] = cutoff
        pooled_oof_posthoc_ops[f"{target_spec * 100:.1f}%"] = op_metrics

    # 2. Leakage-Safe Fold-Derived Operating Points (Fold-level threshold selection on train fold only)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_counts_a = {s: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "cutoffs": []} for s in operating_specificities}
    for train_idx, val_idx in skf.split(z_scores, y_true):
        z_train, y_train = z_scores[train_idx], y_true[train_idx]
        z_val, y_val = z_scores[val_idx], y_true[val_idx]
        fpr_tr, _, thresh_tr = roc_curve(y_train, z_train)
        spec_tr = 1 - fpr_tr
        for s in operating_specificities:
            valid_tr = np.where(spec_tr >= s)[0]
            if len(valid_tr) > 0:
                c_s = float(thresh_tr[valid_tr[-1]])
            else:
                c_s = float(np.percentile(z_train[y_train == 0], s * 100))
            pred_v = (z_val >= c_s).astype(int)
            fold_counts_a[s]["tp"] += int(np.sum((y_val == 1) & (pred_v == 1)))
            fold_counts_a[s]["fp"] += int(np.sum((y_val == 0) & (pred_v == 1)))
            fold_counts_a[s]["tn"] += int(np.sum((y_val == 0) & (pred_v == 0)))
            fold_counts_a[s]["fn"] += int(np.sum((y_val == 1) & (pred_v == 0)))
            fold_counts_a[s]["cutoffs"].append(c_s)

    leakage_safe_ops = {}
    for s in operating_specificities:
        fc = fold_counts_a[s]
        m_ls = compute_contingency_metrics(fc["tp"], fc["fp"], fc["tn"], fc["fn"])
        m_ls["mean_fold_cutoff"] = float(np.mean(fc["cutoffs"]))
        m_ls["fold_cutoffs"] = [float(c) for c in fc["cutoffs"]]
        leakage_safe_ops[f"{s * 100:.1f}%"] = m_ls

    return {
        "model": "Model A (Conventional Z_run Baseline)",
        "aneuploidy": aneuploidy,
        "roc_auc": roc_auc,
        "partial_roc_auc": partial_roc_auc,
        "pr_auc": pr_auc,
        "standard_z_cutoff_3_0": metrics_z3,
        "leakage_safe_operating_points": leakage_safe_ops,
        "pooled_oof_posthoc_operating_points": pooled_oof_posthoc_ops,
        "operating_points": leakage_safe_ops  # Primary operating points mapped to leakage-safe
    }


def run_stratified_cv_evaluation(
    df: pd.DataFrame,
    aneuploidy: str,
    feature_set: str,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Executes Stratified K-Fold CV with explicit Platt calibration and fold-level leakage protection.
    Collects out-of-fold (OOF) screening probabilities.
    
    For T13 (N=5 positives): Evaluates 5-fold CV where each test fold contains exactly 1 positive case.
    """
    prefix = f"{aneuploidy}_"
    target_col = f"{prefix}Target"
    feature_cols = get_candidate_features(aneuploidy, feature_set)

    X_raw = sanitize_input_features(df, feature_cols)
    y = df[target_col].values
    sample_ids = df["Sample ID"].values

    total_pos = int(np.sum(y == 1))
    total_samples = len(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_probs = np.zeros(total_samples, dtype=float)
    fold_counts = {s: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "cutoffs": []} for s in [0.990, 0.995, 0.999]}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        X_train, y_train = X_raw.iloc[train_idx], y[train_idx]
        X_val, y_val = X_raw.iloc[val_idx], y[val_idx]

        # Use internal 3-fold Platt calibration on training fold
        # If training fold has fewer than 6 positives, fallback to cv=2 to ensure each inner fold has positive samples
        inner_pos = np.sum(y_train == 1)
        inner_cv = min(3, max(2, inner_pos))

        model = build_calibrated_pipeline(feature_set=feature_set, cv=inner_cv, random_state=random_state + fold_idx)
        model.fit(X_train, y_train)

        # Predict probabilities on both train fold (for leakage-safe threshold selection) and val fold
        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

        # Determine threshold on TRAINING fold only, then apply ONCE to validation fold
        fpr_tr, _, thresh_tr = roc_curve(y_train, train_probs)
        spec_tr = 1 - fpr_tr
        for s in [0.990, 0.995, 0.999]:
            valid_tr = np.where(spec_tr >= s)[0]
            if len(valid_tr) > 0:
                cutoff_s = float(thresh_tr[valid_tr[-1]])
            else:
                cutoff_s = float(np.percentile(train_probs[y_train == 0], s * 100))

            val_pred_s = (val_probs >= cutoff_s).astype(int)
            fold_counts[s]["tp"] += int(np.sum((y_val == 1) & (val_pred_s == 1)))
            fold_counts[s]["fp"] += int(np.sum((y_val == 0) & (val_pred_s == 1)))
            fold_counts[s]["tn"] += int(np.sum((y_val == 0) & (val_pred_s == 0)))
            fold_counts[s]["fn"] += int(np.sum((y_val == 1) & (val_pred_s == 0)))
            fold_counts[s]["cutoffs"].append(cutoff_s)

    # Compute aggregated leakage-safe operating point metrics across validation folds
    leakage_safe_ops = {}
    for s in [0.990, 0.995, 0.999]:
        fc = fold_counts[s]
        m_ls = compute_contingency_metrics(fc["tp"], fc["fp"], fc["tn"], fc["fn"])
        m_ls["mean_fold_cutoff"] = float(np.mean(fc["cutoffs"]))
        m_ls["fold_cutoffs"] = [float(c) for c in fc["cutoffs"]]
        leakage_safe_ops[f"{s * 100:.1f}%"] = m_ls

    # Out-of-fold dataframe
    oof_df = pd.DataFrame({
        "Sample ID": sample_ids,
        "Target": y,
        f"prob_model_{feature_set.lower()}": oof_probs
    })

    # Calculate overall OOF screening metrics
    metrics = compute_model_metrics(
        y_true=y,
        y_prob=oof_probs,
        aneuploidy=aneuploidy,
        model_name=f"Model {feature_set.upper()}",
        leakage_safe_operating_points=leakage_safe_ops
    )

    return metrics, oof_df


def compute_model_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    aneuploidy: str,
    model_name: str,
    operating_specificities: List[float] = [0.990, 0.995, 0.999],
    leakage_safe_operating_points: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive screening metrics from out-of-fold probabilities.
    Includes both:
    1. Leakage-safe fold-derived operating points (primary methodology)
    2. Previous pooled-OOF post-hoc analytical thresholds (retained for comparison/transparency)
    """
    roc_auc = float(roc_auc_score(y_true, y_prob))
    try:
        partial_roc_auc = float(roc_auc_score(y_true, y_prob, max_fpr=0.10))
    except Exception:
        partial_roc_auc = float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision))
    brier = float(brier_score_loss(y_true, y_prob))

    # Metric at default threshold p = 0.5
    pred_05 = (y_prob >= 0.5).astype(int)
    tp_05 = int(np.sum((y_true == 1) & (pred_05 == 1)))
    fp_05 = int(np.sum((y_true == 0) & (pred_05 == 1)))
    tn_05 = int(np.sum((y_true == 0) & (pred_05 == 0)))
    fn_05 = int(np.sum((y_true == 1) & (pred_05 == 0)))
    metrics_05 = compute_contingency_metrics(tp_05, fp_05, tn_05, fn_05)

    # Previous pooled-OOF post-hoc analytical thresholds (for comparison)
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    pooled_oof_posthoc_ops = {}
    for target_spec in operating_specificities:
        spec_arr = 1 - fpr
        valid_indices = np.where(spec_arr >= target_spec)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            cutoff = float(roc_thresh[idx])
        else:
            cutoff = float(np.percentile(y_prob[y_true == 0], target_spec * 100))

        y_pred_op = (y_prob >= cutoff).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred_op == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred_op == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred_op == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred_op == 0)))
        op_metrics = compute_contingency_metrics(tp, fp, tn, fn)
        op_metrics["probability_cutoff"] = cutoff
        pooled_oof_posthoc_ops[f"{target_spec * 100:.1f}%"] = op_metrics

    # Statistical reliability note
    total_pos = int(np.sum(y_true == 1))
    if total_pos <= 5:
        reliability_note = (
            f"PRELIMINARY / EXPLORATORY ONLY: Evaluated on only {total_pos} independent positive samples. "
            "Repeated resampling does NOT increase the number of independent positive cases. "
            "Confidence intervals are wide."
        )
    else:
        reliability_note = f"Evaluated on {total_pos} positive cases and {len(y_true) - total_pos} negative cases."

    res = {
        "model": model_name,
        "aneuploidy": aneuploidy,
        "n_positives": total_pos,
        "n_negatives": len(y_true) - total_pos,
        "roc_auc": roc_auc,
        "partial_roc_auc": partial_roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "default_threshold_0_5": metrics_05,
        "leakage_safe_operating_points": leakage_safe_operating_points if leakage_safe_operating_points is not None else pooled_oof_posthoc_ops,
        "pooled_oof_posthoc_operating_points": pooled_oof_posthoc_ops,
        "operating_points": leakage_safe_operating_points if leakage_safe_operating_points is not None else pooled_oof_posthoc_ops,
        "statistical_reliability_note": reliability_note
    }
    return res


def perform_ablation_comparison(
    metrics_b: Dict[str, Any],
    metrics_c: Dict[str, Any],
    oof_b: pd.DataFrame,
    oof_c: pd.DataFrame,
    aneuploidy: str
) -> Dict[str, Any]:
    """
    Objectively benchmarks Model B (6 Z-scores) against Model C (6 Z-scores + 5 clinical features).
    Evaluates whether clinical/assay features provide incremental predictive value without assuming superiority.
    """
    delta_pr_auc = metrics_c["pr_auc"] - metrics_b["pr_auc"]
    delta_roc_auc = metrics_c["roc_auc"] - metrics_b["roc_auc"]
    delta_brier = metrics_c["brier_score"] - metrics_b["brier_score"]  # Lower Brier is better

    prob_b = oof_b["prob_model_b"].values
    prob_c = oof_c["prob_model_c"].values
    y_true = oof_b["Target"].values
    sample_ids = oof_b["Sample ID"].values

    # Paired disagreement analysis at default p = 0.5
    pred_b_05 = (prob_b >= 0.5).astype(int)
    pred_c_05 = (prob_c >= 0.5).astype(int)

    disagree_mask = (pred_b_05 != pred_c_05)
    disagree_count = int(np.sum(disagree_mask))

    discrepant_samples = []
    if disagree_count > 0:
        for idx in np.where(disagree_mask)[0]:
            discrepant_samples.append({
                "Sample ID": str(sample_ids[idx]),
                "Target": int(y_true[idx]),
                "Prob_Model_B": float(prob_b[idx]),
                "Prob_Model_C": float(prob_c[idx]),
                "Call_Model_B": int(pred_b_05[idx]),
                "Call_Model_C": int(pred_c_05[idx])
            })

    # Objective conclusion
    if delta_pr_auc > 0.01:
        conclusion = "Model C demonstrated moderate incremental PR-AUC improvement over Model B."
    elif delta_pr_auc < -0.01:
        conclusion = "Model B outperformed Model C in PR-AUC; adding clinical features did not improve separation."
    else:
        conclusion = "Model B and Model C showed comparable screening performance (ΔPR-AUC within ±0.01)."

    return {
        "aneuploidy": aneuploidy,
        "pr_auc_model_b": metrics_b["pr_auc"],
        "pr_auc_model_c": metrics_c["pr_auc"],
        "delta_pr_auc": float(delta_pr_auc),
        "roc_auc_model_b": metrics_b["roc_auc"],
        "roc_auc_model_c": metrics_c["roc_auc"],
        "delta_roc_auc": float(delta_roc_auc),
        "brier_score_model_b": metrics_b["brier_score"],
        "brier_score_model_c": metrics_c["brier_score"],
        "delta_brier": float(delta_brier),
        "paired_disagreements_count": disagree_count,
        "discrepant_samples": discrepant_samples,
        "objective_ablation_conclusion": conclusion
    }
