"""
View 6: Unsupervised NIPT Sample Profiling (K-Means)
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_clustering_artifacts():
    kmeans_dir = Path("models") / "kmeans"
    if not kmeans_dir.exists():
        return None, None, None, None

    diag_path = kmeans_dir / "diagnostics.json"
    pca_path = kmeans_dir / "pca_projection.csv"
    medians_path = kmeans_dir / "cluster_medians.csv"
    meta_path = kmeans_dir / "metadata.json"

    diag = json.load(open(diag_path)) if diag_path.exists() else None
    pca_df = pd.read_csv(pca_path) if pca_path.exists() else None
    medians_df = pd.read_csv(medians_path) if medians_path.exists() else None
    meta = json.load(open(meta_path)) if meta_path.exists() else None

    return diag, pca_df, medians_df, meta


def render_view6(master_df: pd.DataFrame):
    st.header("6. NIPT Sample Profiling (Unsupervised Analysis)")
    st.markdown(
        """
        Unsupervised multi-dimensional profiling of NIPT samples across patient demographics (`age`, `week`), 
        sequencing assay quality metrics (`fetal`, `reads`, `peak`), and chromosomal dosage context (`Z_chr_vs_n` for T13, T18, T21).
        """
    )

    st.warning(
        "🛡️ **Strict Non-Diagnostic Constraint:** "
        "K-Means clustering is strictly an unsupervised exploratory tool for grouping technical assay profiles and patient demographics. "
        "It is NEVER used to predict or assign medical diagnoses. "
        "Clusters are NOT labeled with chromosomal aneuploidy or clinical risk categories."
    )

    diag, pca_df, medians_df, meta = load_clustering_artifacts()

    if diag is None or pca_df is None or medians_df is None:
        st.warning("Clustering models and profiling artifacts not yet generated on disk. Please run the training pipeline first.")
        return

    # Section 1: Cluster Selection Diagnostics
    st.subheader("Cluster Count Diagnostics (K = 2 to K = 8)")
    st.write(f"**Diagnosed Optimal Clusters:** `K = {diag['optimal_k_selected']}` ({diag['selection_rationale']})")

    diag_df = pd.DataFrame({
        "K (Clusters)": diag["k_evaluated"],
        "Silhouette Score (Cohesion)": diag["silhouette_scores"],
        "Davies-Bouldin Index (Separation)": diag["davies_bouldin_indices"],
        "Inertia (Within-Cluster SS)": diag["inertias"]
    })

    col1, col2 = st.columns(2)
    with col1:
        fig_sil = px.line(diag_df, x="K (Clusters)", y="Silhouette Score (Cohesion)", markers=True, title="Silhouette Score across K")
        fig_sil.add_vline(x=diag["optimal_k_selected"], line_dash="dash", line_color="green", annotation_text=f"Selected K={diag['optimal_k_selected']}")
        st.plotly_chart(fig_sil, use_container_width=True)

    with col2:
        fig_db = px.line(diag_df, x="K (Clusters)", y="Davies-Bouldin Index (Separation)", markers=True, title="Davies-Bouldin Index across K (Lower is Better)")
        fig_db.add_vline(x=diag["optimal_k_selected"], line_dash="dash", line_color="green", annotation_text=f"Selected K={diag['optimal_k_selected']}")
        st.plotly_chart(fig_db, use_container_width=True)

    # Section 2: 2D PCA Cluster Projection
    st.subheader("2D PCA Projection of Discovered Sample Profiles")
    var_exp = meta.get("pca_variance_ratio", [0, 0])
    fig_pca = px.scatter(
        pca_df,
        x="PCA_Dim1",
        y="PCA_Dim2",
        color=pca_df["Profile_Cluster"].astype(str),
        hover_data=["Sample ID", "QC", "fetal", "age", "week", "reads"],
        title=f"Sample Distribution in 2D PCA Space (Var Exp: Dim1={var_exp[0]*100:.1f}%, Dim2={var_exp[1]*100:.1f}%)",
        labels={"PCA_Dim1": "Principal Component 1", "PCA_Dim2": "Principal Component 2", "color": "Sample Profile"},
        opacity=0.65
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # Section 3: Technical Characterization of Profiles
    st.subheader("Discovered Sample Profile Characteristics (Median Values)")
    st.dataframe(medians_df, use_container_width=True)

    st.markdown("#### Technical Cohort Interpretation:")
    for cluster_id in sorted(medians_df["Profile_Cluster"].unique()):
        row = medians_df[medians_df["Profile_Cluster"] == cluster_id].iloc[0]
        n_samples = meta["cluster_sample_counts"].get(str(cluster_id), "N/A")
        st.markdown(
            f"- **Profile {cluster_id}** (`{n_samples}` samples): "
            f"Median Fetal Fraction = `{row['fetal']:.1f}%`, "
            f"Maternal Age = `{row['age']:.0f} yrs`, "
            f"Gestational Age = `{row['week']:.0f} wks`, "
            f"Mapped Reads = `{row['reads']:,.0f}`, "
            f"Fragment Peak = `{row['peak']:.1f} bp`."
        )
