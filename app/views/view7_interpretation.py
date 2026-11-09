"""
View 7: Model B vs Model C Ablation & Clinical Interpretation
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def load_ablation_summary():
    path = Path("models") / "ablation_summary.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def render_view7(master_df: pd.DataFrame):
    st.header("7. Model B vs Model C Ablation & Interpretation")
    st.markdown(
        """
        ### Objective Research Question:
        *Does adding 5 clinical and assay features (`fetal`, `age`, `week`, `reads`, `peak`) to the 6 chromosome-specific Z-scores 
        provide incremental predictive value for NIPT screening?*
        """
    )

    ablations = load_ablation_summary()
    if ablations is None:
        st.warning("Ablation summary data not found on disk. Run model training first.")
        return

    # Section 1: Objective Ablation Benchmark Table
    st.subheader("Objective Ablation Benchmark Across Chromosomes")
    st.info(
        "⚖️ **Neutral Hypothesis:** We do NOT assume that Model C will outperform Model B. "
        "Adding features to non-linear SVMs can increase variance or noise without improving classification margins."
    )

    ablation_rows = []
    for aneu in ["T13", "T18", "T21"]:
        if aneu in ablations:
            a = ablations[aneu]
            ablation_rows.append({
                "Condition": aneu,
                "Model B PR-AUC": f"{a['pr_auc_model_b']:.4f}",
                "Model C PR-AUC": f"{a['pr_auc_model_c']:.4f}",
                "ΔPR-AUC (C - B)": f"{a['delta_pr_auc']:+.4f}",
                "Model B ROC-AUC": f"{a['roc_auc_model_b']:.4f}",
                "Model C ROC-AUC": f"{a['roc_auc_model_c']:.4f}",
                "ΔROC-AUC (C - B)": f"{a['delta_roc_auc']:+.4f}",
                "ΔBrier Score (C - B)": f"{a['delta_brier']:+.5f}",
                "Disagreements (p=0.5)": a["paired_disagreements_count"],
                "Objective Assessment": a["objective_ablation_conclusion"]
            })

    st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True)

    # Section 2: Disagreement Case Analysis
    st.subheader("Paired Prediction Discordance Analysis")
    st.markdown(
        "Samples where Model B and Model C generated conflicting screening calls at the default decision threshold ($p = 0.5$):"
    )

    aneu_choice = st.selectbox("Inspect Discordant Samples for", ["T21", "T18", "T13"], key="ablation_aneu")
    if aneu_choice in ablations:
        disc = ablations[aneu_choice].get("discrepant_samples", [])
        if len(disc) > 0:
            disc_df = pd.DataFrame(disc)
            st.dataframe(disc_df, use_container_width=True)
        else:
            st.success(f"Zero disagreements observed between Model B and Model C at p = 0.5 for {aneu_choice}.")

    # Section 3: Performance Stratified by Library QC
    st.subheader("Screening Performance Stratified by Assay Library QC")
    st.markdown(
        "Analyzing whether model screening quality degrades within the `QC-filtered` library cohort:"
    )

    qc_tabs = st.tabs(["Trisomy 21", "Trisomy 18", "Trisomy 13"])
    for tab, aneu in zip(qc_tabs, ["T21", "T18", "T13"]):
        with tab:
            oof_path = Path("models") / aneu.lower() / "oof_predictions.csv"
            if oof_path.exists():
                oof_df = pd.read_csv(oof_path)
                strat_rows = []
                for qc_val in ["QC-pass", "QC-filtered"]:
                    sub = oof_df[oof_df["QC"] == qc_val]
                    tp_b = int(np.sum((sub["Target"] == 1) & (sub["Prob_Model_B"] >= 0.5)))
                    fn_b = int(np.sum((sub["Target"] == 1) & (sub["Prob_Model_B"] < 0.5)))
                    fp_b = int(np.sum((sub["Target"] == 0) & (sub["Prob_Model_B"] >= 0.5)))
                    pos = int(np.sum(sub["Target"] == 1))
                    strat_rows.append({
                        "Cohort": qc_val,
                        "Sample Count": len(sub),
                        "Confirmed Trisomies": pos,
                        "Model B Detected": tp_b,
                        "Model B Missed": fn_b,
                        "Model B False Positives": fp_b
                    })
                st.dataframe(pd.DataFrame(strat_rows), use_container_width=True)
            else:
                st.info("Out-of-fold data not yet available.")
