"""
View 5: Model Performance Benchmarking & Operating Points
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve


def load_metrics(aneuploidy: str):
    path = Path("models") / aneuploidy.lower() / "metrics_summary.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_oof_predictions(aneuploidy: str):
    path = Path("models") / aneuploidy.lower() / "oof_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def render_view5(master_df: pd.DataFrame):
    st.header("5. Model Performance & Operating Points")
    st.markdown(
        """
        Rigorous out-of-fold benchmarking comparing:
        - **Model A**: Conventional Z_run Baseline
        - **Model B**: 6 Z-Score RBF-SVM
        - **Model C**: Extended Multivariate RBF-SVM
        """
    )

    aneuploidy = st.selectbox("Select Target Condition for Benchmarking", ["T21", "T18", "T13"], key="perf_aneu")
    metrics = load_metrics(aneuploidy)
    oof_df = load_oof_predictions(aneuploidy)

    if metrics is None or oof_df is None:
        st.warning(f"Training metrics for {aneuploidy} not yet generated on disk. Please execute model training first.")
        return

    # Statistical notice
    if aneuploidy == "T13":
        st.error(
            "⚠️ **EXPLORATORY / PRELIMINARY NOTICE (T13):** Evaluated on only **5 independent positive samples**. "
            "Cross-validation estimates carry large statistical uncertainty. 1 misclassified case alters sensitivity by 20%."
        )

    # KPI Comparison Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Model A (Baseline)")
        st.metric("PR-AUC", f"{metrics['Model_A']['pr_auc']:.4f}")
        st.metric("ROC-AUC", f"{metrics['Model_A']['roc_auc']:.4f}")
    with col2:
        st.markdown("### Model B (6 Z-scores)")
        st.metric("PR-AUC", f"{metrics['Model_B']['pr_auc']:.4f}")
        st.metric("ROC-AUC", f"{metrics['Model_B']['roc_auc']:.4f}")
    with col3:
        st.markdown("### Model C (Extended)")
        st.metric("PR-AUC", f"{metrics['Model_C']['pr_auc']:.4f}")
        st.metric("ROC-AUC", f"{metrics['Model_C']['roc_auc']:.4f}")

    # Section: Comparative ROC & PR Curves
    st.subheader(f"Comparative Discrimination Curves ({aneuploidy})")

    y_true = oof_df["Target"].values
    score_a = oof_df["Z_run"].values
    prob_b = oof_df["Prob_Model_B"].values
    prob_c = oof_df["Prob_Model_C"].values

    fpr_a, tpr_a, _ = roc_curve(y_true, score_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, prob_b)
    fpr_c, tpr_c, _ = roc_curve(y_true, prob_c)

    prec_a, rec_a, _ = precision_recall_curve(y_true, score_a)
    prec_b, rec_b, _ = precision_recall_curve(y_true, prob_b)
    prec_c, rec_c, _ = precision_recall_curve(y_true, prob_c)

    tab_roc, tab_pr = st.tabs(["ROC Curves", "Precision-Recall Curves"])

    with tab_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_a, y=tpr_a, mode="lines", name=f"Model A (AUC = {metrics['Model_A']['roc_auc']:.4f})", line=dict(dash="dash", color="gray")))
        fig_roc.add_trace(go.Scatter(x=fpr_b, y=tpr_b, mode="lines", name=f"Model B (AUC = {metrics['Model_B']['roc_auc']:.4f})", line=dict(color="#1f77b4", width=2)))
        fig_roc.add_trace(go.Scatter(x=fpr_c, y=tpr_c, mode="lines", name=f"Model C (AUC = {metrics['Model_C']['roc_auc']:.4f})", line=dict(color="#2ca02c", width=2)))
        fig_roc.add_shape(type="line", line=dict(dash="dot", color="silver"), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_layout(title=f"ROC Curves ({aneuploidy})", xaxis_title="False Positive Rate (1 - Specificity)", yaxis_title="True Positive Rate (Sensitivity)")
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab_pr:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec_a, y=prec_a, mode="lines", name=f"Model A (PR-AUC = {metrics['Model_A']['pr_auc']:.4f})", line=dict(dash="dash", color="gray")))
        fig_pr.add_trace(go.Scatter(x=rec_b, y=prec_b, mode="lines", name=f"Model B (PR-AUC = {metrics['Model_B']['pr_auc']:.4f})", line=dict(color="#1f77b4", width=2)))
        fig_pr.add_trace(go.Scatter(x=rec_c, y=prec_c, mode="lines", name=f"Model C (PR-AUC = {metrics['Model_C']['pr_auc']:.4f})", line=dict(color="#2ca02c", width=2)))
        fig_pr.update_layout(title=f"Precision-Recall Curves ({aneuploidy})", xaxis_title="Recall (Sensitivity)", yaxis_title="Precision (PPV)")
        st.plotly_chart(fig_pr, use_container_width=True)

    # Section: Leakage-Safe Experimental Operating Points (Primary)
    st.subheader("Leakage-Safe Experimental Operating Points (Primary Methodology)")
    st.info(
        "🛡️ **Leakage-Safe Fold-Derived Evaluation:** "
        "In each outer cross-validation fold, the threshold targeting empirical specificity $\\ge S$ is determined strictly using only training-fold data (`X_train`). "
        "That threshold is applied once to the held-out validation fold (`X_val`). Counts (TP, FP, TN, FN) are accumulated across the 5 independent validation folds. "
        "This completely eliminates post-hoc threshold-selection optimism. "
        "Specificities at 99.0%, 99.5%, and 99.9% are experimental analytical benchmarks, not clinical standards."
    )

    ls_rows = []
    for model_key, model_label in [("Model_A", "Model A (Baseline)"), ("Model_B", "Model B (6 Z-scores)"), ("Model_C", "Model C (Extended)")]:
        ops = metrics[model_key].get("leakage_safe_operating_points", metrics[model_key].get("operating_points", {}))
        for op_point in ["99.0%", "99.5%", "99.9%"]:
            if op_point in ops:
                op = ops[op_point]
                ls_rows.append({
                    "Model": model_label,
                    "Target Specificity": op_point,
                    "Mean Fold Cutoff": f"{op.get('mean_fold_cutoff', 0.0):.4f}",
                    "Sensitivity": f"{op['sensitivity']*100:.1f}%",
                    "Sensitivity (95% CI)": f"[{op['sensitivity_ci_95'][0]*100:.1f}%, {op['sensitivity_ci_95'][1]*100:.1f}%]",
                    "Achieved Specificity": f"{op['specificity']*100:.2f}%",
                    "Specificity (95% CI)": f"[{op['specificity_ci_95'][0]*100:.2f}%, {op['specificity_ci_95'][1]*100:.2f}%]",
                    "PPV": f"{op['ppv']*100:.1f}%",
                    "PPV (95% CI)": f"[{op['ppv_ci_95'][0]*100:.1f}%, {op['ppv_ci_95'][1]*100:.1f}%]",
                    "NPV": f"{op['npv']*100:.3f}%",
                    "TP": op["tp"], "FP": op["fp"], "TN": op["tn"], "FN": op["fn"]
                })

    st.dataframe(pd.DataFrame(ls_rows), use_container_width=True)

    # Section: Post-Hoc Pooled OOF Benchmarks (Comparative Reference)
    with st.expander("Comparative Reference: Post-Hoc Pooled OOF Analytical Benchmarks (Descriptive Only)"):
        st.caption(
            "These benchmarks reflect the previous post-hoc analytical optimization where thresholds were identified across all 5,518 pooled OOF predictions. "
            "They are retained strictly for methodological transparency and historical comparison."
        )
        ph_rows = []
        for model_key, model_label in [("Model_A", "Model A (Baseline)"), ("Model_B", "Model B (6 Z-scores)"), ("Model_C", "Model C (Extended)")]:
            ops = metrics[model_key].get("pooled_oof_posthoc_operating_points", {})
            for op_point in ["99.0%", "99.5%", "99.9%"]:
                if op_point in ops:
                    op = ops[op_point]
                    ph_rows.append({
                        "Model": model_label,
                        "Target Specificity": op_point,
                        "Cutoff": f"{op.get('probability_cutoff', op.get('cutoff_z', 0.0)):.4f}",
                        "Sensitivity": f"{op['sensitivity']*100:.1f}%",
                        "Sensitivity (95% CI)": f"[{op['sensitivity_ci_95'][0]*100:.1f}%, {op['sensitivity_ci_95'][1]*100:.1f}%]",
                        "Achieved Specificity": f"{op['specificity']*100:.2f}%",
                        "PPV": f"{op['ppv']*100:.1f}%",
                        "PPV (95% CI)": f"[{op['ppv_ci_95'][0]*100:.1f}%, {op['ppv_ci_95'][1]*100:.1f}%]",
                        "NPV": f"{op['npv']*100:.3f}%",
                        "TP": op["tp"], "FP": op["fp"], "TN": op["tn"], "FN": op["fn"]
                    })
        st.dataframe(pd.DataFrame(ph_rows), use_container_width=True)
