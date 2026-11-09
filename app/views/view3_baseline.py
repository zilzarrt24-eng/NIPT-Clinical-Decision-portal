"""
View 3: Conventional Z-Score Baseline (Model A)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.utils import compute_contingency_metrics


def render_view3(master_df: pd.DataFrame):
    st.header("3. Conventional Z-Score Baseline (Model A)")
    st.markdown(
        r"""
        Conventional laboratory NIPT relies on univariate Z-score thresholding (`Z_run`). 
        A chromosome dosage metric exceeding a statistical threshold (conventionally $Z \ge 3.0$, representing 3 standard deviations above reference euploid mean) 
        prompts a screen-positive call.
        """
    )

    aneuploidy = st.selectbox("Select Target Chromosome Condition", ["T21", "T18", "T13"])
    z_col = f"{aneuploidy}_Z_run"
    target_col = f"{aneuploidy}_Target"

    y_true = master_df[target_col].values
    z_scores = master_df[z_col].values

    # Interactive Cutoff Slider
    z_cutoff = st.slider("Select Baseline Z-Score Cutoff", min_value=1.5, max_value=6.0, value=3.0, step=0.1)

    # Calculate real-time metrics
    y_pred = (z_scores >= z_cutoff).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    metrics = compute_contingency_metrics(tp, fp, tn, fn)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sensitivity (Recall)", f"{metrics['sensitivity']*100:.1f}%", f"95% CI: [{metrics['sensitivity_ci_95'][0]*100:.1f}%, {metrics['sensitivity_ci_95'][1]*100:.1f}%]")
    with col2:
        st.metric("Specificity", f"{metrics['specificity']*100:.2f}%", f"95% CI: [{metrics['specificity_ci_95'][0]*100:.2f}%, {metrics['specificity_ci_95'][1]*100:.2f}%]")
    with col3:
        st.metric("PPV", f"{metrics['ppv']*100:.1f}%" if (tp+fp)>0 else "N/A", f"True Positives: {tp}")
    with col4:
        st.metric("NPV", f"{metrics['npv']*100:.3f}%", f"False Negatives: {fn}")

    # Confusion Matrix Table
    st.subheader(f"Contingency Matrix at Z = {z_cutoff:.1f}")
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        columns=["Predicted Reference (Z < cutoff)", "Predicted Positive (Z >= cutoff)"],
        index=["Actual Reference (0)", "Actual Trisomy (1)"]
    )
    st.dataframe(cm_df, use_container_width=True)

    # Histogram & Grey Zone
    st.subheader(f"Distribution of {z_col} and Grey Zone Analysis")
    fig_z = px.histogram(
        master_df,
        x=z_col,
        color=master_df[target_col].map({0: "Reference (0)", 1: "Trisomy (1)"}),
        nbins=80,
        title=f"{aneuploidy} Z_run Distribution",
        labels={z_col: f"{aneuploidy} Z_run Score"},
        barmode="overlay",
        opacity=0.75
    )
    fig_z.add_vline(x=z_cutoff, line_dash="dash", line_color="black", annotation_text=f"Selected Z={z_cutoff}")
    fig_z.add_vrect(x0=2.5, x1=3.5, fillcolor="yellow", opacity=0.2, annotation_text="Grey Zone [2.5, 3.5]")
    st.plotly_chart(fig_z, use_container_width=True)

    # Grey zone samples table
    grey_zone = master_df[(master_df[z_col] >= 2.5) & (master_df[z_col] <= 3.5)]
    st.write(f"Samples in grey zone ($2.5 \\le Z \\le 3.5$): **{len(grey_zone)}** (Confirmed Trisomies: {int(grey_zone[target_col].sum())})")
    if len(grey_zone) > 0:
        st.dataframe(grey_zone[["Sample ID", z_col, target_col, "QC", "fetal", "age", "week"]].head(15), use_container_width=True)
