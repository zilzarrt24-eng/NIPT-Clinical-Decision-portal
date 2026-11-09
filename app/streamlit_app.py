"""
NIPT Clinical Decision Support Portal - Main Application
========================================================
Interactive clinical decision support and machine learning comparison portal
for Non-Invasive Prenatal Testing (NIPT) data.
"""

from pathlib import Path
import sys
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.views.view1_overview import render_view1
from app.views.view2_qc import render_view2
from app.views.view3_baseline import render_view3
from app.views.view4_ml_screening import render_view4
from app.views.view5_performance import render_view5
from app.views.view6_clustering import render_view6
from app.views.view7_interpretation import render_view7
from app.views.view8_limitations import render_view8


st.set_page_config(
    page_title="NIPT Clinical Decision Portal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_master_data():
    csv_path = PROJECT_ROOT / "data" / "processed" / "master_processed.csv"
    if not csv_path.exists():
        st.error(f"Master processed dataset not found at {csv_path}. Please run preprocessing first.")
        st.stop()
    return pd.read_csv(csv_path)


def main():
    master_df = load_master_data()

    # Persistent Top Header Disclaimer
    st.caption("🔬 **Academic Research Prototype** — Experimental screening outputs only — Not a diagnostic medical device.")

    # Sidebar Navigation
    st.sidebar.title("🧬 NIPT Portal")
    st.sidebar.markdown("**Clinical Decision Support System**")
    st.sidebar.markdown("---")

    views = [
        "1. Dataset Overview",
        "2. QC / Data Quality",
        "3. Conventional Z-score Baseline",
        "4. ML Screening",
        "5. Model Performance",
        "6. NIPT Sample Profiling",
        "7. Model B vs C Ablation & Interpretation",
        "8. Limitations and Disclaimer"
    ]

    selected_view = st.sidebar.radio("Navigation", views, index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("Cohort: 5,518 Clinical Samples")
    st.sidebar.caption("Assays: Trisomy 13, 18, 21")
    st.sidebar.caption("Status: Research & Decision Support")
    st.sidebar.info(
        "🔬 **Academic Research Prototype**\n\n"
        "Experimental screening outputs only.\n"
        "Not an FDA/CE cleared diagnostic device."
    )

    # Routing
    if selected_view == "1. Dataset Overview":
        render_view1(master_df)
    elif selected_view == "2. QC / Data Quality":
        render_view2(master_df)
    elif selected_view == "3. Conventional Z-score Baseline":
        render_view3(master_df)
    elif selected_view == "4. ML Screening":
        render_view4(master_df)
    elif selected_view == "5. Model Performance":
        render_view5(master_df)
    elif selected_view == "6. NIPT Sample Profiling":
        render_view6(master_df)
    elif selected_view == "7. Model B vs C Ablation & Interpretation":
        render_view7(master_df)
    elif selected_view == "8. Limitations and Disclaimer":
        render_view8(master_df)

    # Persistent Bottom Footer
    st.markdown("---")
    st.caption(
        "🧬 **NIPT Clinical Decision Support Portal** | Computational Screening Comparison Framework | "
        "Academic Demonstration Prototype — Not a Diagnostic Device"
    )


if __name__ == "__main__":
    main()
