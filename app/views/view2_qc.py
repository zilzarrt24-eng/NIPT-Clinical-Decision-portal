"""
View 2: Quality Control & Data Quality Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render_view2(master_df: pd.DataFrame):
    st.header("2. QC & Data Quality Analysis")
    st.markdown(
        """
        Quality control (QC) is vital in NIPT. Cell-free DNA (cfDNA) circulating in maternal plasma 
        originates from both the mother and the placenta (fetal fraction). Low fetal fraction, inadequate sequencing depth, 
        or atypical library fragmentation can affect test interpretability.
        """
    )

    # Section 1: Fetal Fraction Distribution & Cutoffs
    st.subheader("Fetal Fraction Distribution & Technical Cutoffs")
    cutoff = st.slider("Experimental Fetal Fraction Threshold (%)", min_value=1.0, max_value=8.0, value=4.0, step=0.5)
    
    below_cutoff = master_df[master_df["fetal"] < cutoff]
    st.write(
        f"Samples below {cutoff:.1f}% fetal fraction: **{len(below_cutoff)}** ({len(below_cutoff)/len(master_df)*100:.2f}% of cohort)."
    )

    fig_fetal = px.histogram(
        master_df,
        x="fetal",
        nbins=60,
        title="Fetal Fraction (%) Distribution across 5,518 Samples",
        labels={"fetal": "Fetal Fraction (%)"},
        color_discrete_sequence=["#1f77b4"]
    )
    fig_fetal.add_vline(x=cutoff, line_dash="dash", line_color="red", annotation_text=f"Cutoff {cutoff}%")
    st.plotly_chart(fig_fetal, use_container_width=True)

    # Section 2: Transparent Audit of Invalid Fetal Fraction Sample
    st.subheader("Transparent Audit: Negative Fetal Fraction Sample (#666)")
    invalid_sample = master_df[master_df["fetal_invalid_flag"] == 1]
    
    st.dataframe(
        invalid_sample[[
            "Sample ID", "fetal", "QC", "peak", "reads", "age", "week",
            "T13_Real_state", "T18_Real_state", "T21_Real_state"
        ]],
        use_container_width=True
    )
    
    st.info(
        "🔍 **Transparent Handling Policy:** Sample `#666` presents a biologically non-physical fetal fraction estimate of "
        "**-1.35609%**, accompanied by an abnormally low nucleosomal fragment peak (132.48 bp vs. cohort median 149.13 bp). "
        "The sample was marked `QC-filtered` in the raw assay library. "
        "Per protocol, this value was **NOT silently clamped to zero**. It is preserved in the dataset and flagged with `fetal_invalid_flag=1`. "
        "In the ML pipeline, it is treated as missing and imputed via training-fold median imputation without data leakage."
    )

    # Section 3: Library QC Distribution & Trisomy Cross-Tabulation
    st.subheader("Assay Library QC Flag vs. Confirmed Trisomies")
    qc_summary = []
    for qc_status in ["QC-pass", "QC-filtered"]:
        sub = master_df[master_df["QC"] == qc_status]
        qc_summary.append({
            "Library Status": qc_status,
            "Sample Count": len(sub),
            "% of Cohort": f"{len(sub)/len(master_df)*100:.1f}%",
            "T13 Positives": int(sub["T13_Target"].sum()),
            "T18 Positives": int(sub["T18_Target"].sum()),
            "T21 Positives": int(sub["T21_Target"].sum()),
            "Total Confirmed Trisomies": int(sub["T13_Target"].sum() + sub["T18_Target"].sum() + sub["T21_Target"].sum())
        })
    st.dataframe(pd.DataFrame(qc_summary), use_container_width=True)

    st.warning(
        "🚨 **Critical Preprocessing Decision Rationale:** "
        "Notice that **9 confirmed trisomy cases** (4 T21, 4 T18, 1 T13) fall within the `QC-filtered` cohort (~19% of all positives). "
        "Discarding `QC-filtered` samples would eliminate nearly a fifth of all positive cases. "
        "Therefore, all samples are retained in the dataset, while `QC` is preserved for stratified analysis and excluded as an SVM feature."
    )

    # Section 4: Read Depth vs. Fragment Peak
    st.subheader("Sequencing Depth vs. Fragment Length Peak")
    fig_scatter = px.scatter(
        master_df,
        x="reads",
        y="peak",
        color="QC",
        hover_data=["Sample ID", "fetal", "age", "week"],
        title="Mapped Reads vs. Fragment Peak by QC Status",
        labels={"reads": "Total Mapped Reads", "peak": "Fragment Peak (bp)"},
        opacity=0.6
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
