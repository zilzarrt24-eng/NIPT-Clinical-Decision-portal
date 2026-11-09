"""
View 8: Limitations, Statistical Uncertainty & Clinical Disclaimer
"""

import streamlit as st
import pandas as pd


def render_view8(master_df: pd.DataFrame):
    st.header("8. Limitations, Uncertainty & Clinical Disclaimer")

    st.error(
        "⚠️ **CRITICAL STATISTICAL LIMITATION: TRISOMY 13 (N = 5 POSITIVES)**\n\n"
        "The Trisomy 13 cohort contains only **5 confirmed positive cases** out of 5,518 clinical samples (prevalence 0.091%).\n\n"
        "- **Repeated Resampling Clarification:** Cross-validation or repeated resampling does NOT create new independent biological cases. "
        "There remain only 5 independent positive samples in existence.\n"
        "- **High Variance in Estimates:** A single misclassified positive case shifts measured sensitivity by **20.0 percentage points**.\n"
        "- **Wide Confidence Intervals:** Even if an algorithm achieves 5/5 detection (100% point sensitivity), the exact 95% Wilson score "
        "confidence interval spans from **56.6% to 100.0%**.\n"
        "- **Clinical Status:** All T13 machine learning evaluations must be interpreted strictly as preliminary and exploratory."
    )

    st.subheader("Statistical Cohort Summary & Confidence Bounds")
    stat_summary = [
        {
            "Condition": "Trisomy 21 (Down)",
            "Independent Positives": 27,
            "Cohort Prevalence": "0.489%",
            "Impact of 1 Missed Case": "± 3.7%",
            "95% CI Width (at 100% Sens)": "87.5% - 100.0%",
            "Reliability Tier": "Moderate (Stratified 5-Fold CV valid)"
        },
        {
            "Condition": "Trisomy 18 (Edwards)",
            "Independent Positives": 15,
            "Cohort Prevalence": "0.272%",
            "Impact of 1 Missed Case": "± 6.7%",
            "95% CI Width (at 100% Sens)": "79.6% - 100.0%",
            "Reliability Tier": "Small Cohort (Caution warranted)"
        },
        {
            "Condition": "Trisomy 13 (Patau)",
            "Independent Positives": 5,
            "Cohort Prevalence": "0.091%",
            "Impact of 1 Missed Case": "± 20.0%",
            "95% CI Width (at 100% Sens)": "56.6% - 100.0%",
            "Reliability Tier": "Extremely Rare (Exploratory / Preliminary only)"
        }
    ]
    st.dataframe(pd.DataFrame(stat_summary), use_container_width=True)

    st.subheader("Biological & Technical Assay Caveats")
    st.markdown(
        """
        Non-Invasive Prenatal Testing is subject to complex biological and genomic phenomena that cannot be resolved solely by statistical modeling:

        1. **Confined Placental Mosaicism (CPM):**
           cfDNA in maternal plasma derives primarily from the syncytiotrophoblast of the placenta, not the fetus itself. 
           Discrepancies between placental and fetal karyotypes can yield false-positive or false-negative screening calls.
        2. **Maternal Copy Number Variations (CNVs) & Malignancy:**
           Maternal duplications, microdeletions, or occult maternal malignancies can distort chromosome dosage baselines, 
           leading to aberrant Z-scores unrelated to fetal karyotype.
        3. **Vanishing Twin:**
           An unrecognized demised co-twin can shed apoptotic cfDNA for weeks after demise, confounding dosage calculations.
        4. **Extreme Low Fetal Fraction (< 4.0%):**
           Low placental fraction reduces the signal-to-noise ratio, increasing risk of false-negative calls.
        """
    )

    st.subheader("Regulatory & Clinical Decision Support Disclaimer")
    st.warning(
        "🛑 **REGULATORY DISCLAIMER:**\n\n"
        "This software application is an **experimental research and clinical decision support prototype** "
        "designed for computational algorithm comparison and educational walkthroughs. "
        "It is **NOT an FDA-cleared or CE-marked medical device** and is **NOT intended for autonomous clinical diagnosis**.\n\n"
        "All screening outputs must be interpreted by qualified medical geneticists or laboratory directors in conjunction with "
        "ultrasound findings, clinical history, and confirmatory diagnostic testing (such as chorionic villus sampling or amniocentesis)."
    )
