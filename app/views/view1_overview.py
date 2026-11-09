"""
View 1: Dataset Overview
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render_view1(master_df: pd.DataFrame):
    st.header("1. Dataset Overview")
    st.markdown(
        """
        This decision support portal investigates whether multivariate machine learning provides useful screening 
        information beyond the conventional NIPT Z-score baseline.
        The underlying cohort is derived from **5,518 clinical pregnancy samples** tested for **Trisomy 13, 18, and 21**.
        """
    )

    # High-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{len(master_df):,}")
    with col2:
        t21_count = int(master_df["T21_Target"].sum())
        st.metric("Trisomy 21 Cases", f"{t21_count} ({t21_count/len(master_df)*100:.2f}%)")
    with col3:
        t18_count = int(master_df["T18_Target"].sum())
        st.metric("Trisomy 18 Cases", f"{t18_count} ({t18_count/len(master_df)*100:.2f}%)")
    with col4:
        t13_count = int(master_df["T13_Target"].sum())
        st.metric("Trisomy 13 Cases", f"{t13_count} ({t13_count/len(master_df)*100:.2f}%)")

    st.info(
        "⚠️ **Statistical Notice on Rare Positive Classes:** "
        "Trisomy 13 has only **5 confirmed positive cases** (0.091% prevalence). "
        "All T13 evaluations are strictly preliminary and exploratory. "
        "Trisomy 18 has 15 cases (0.272%), and Trisomy 21 has 27 cases (0.489%)."
    )

    st.subheader("Target Distribution Across Chromosomes")
    summary_data = pd.DataFrame({
        "Condition": ["Trisomy 21 (Down)", "Trisomy 18 (Edwards)", "Trisomy 13 (Patau)"],
        "Negative (Euploid)": [len(master_df) - t21_count, len(master_df) - t18_count, len(master_df) - t13_count],
        "Positive (Trisomy)": [t21_count, t18_count, t13_count],
        "Prevalence (%)": [t21_count/len(master_df)*100, t18_count/len(master_df)*100, t13_count/len(master_df)*100]
    })
    st.dataframe(summary_data, use_container_width=True)

    # Interactive Sample Explorer
    st.subheader("Interactive Sample Browser")
    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        filter_opt = st.selectbox(
            "Cohort Quick Filter",
            ["All Samples (Head 50)", "All Confirmed Trisomies (47)", "QC-Filtered Cohort (195)", "Complete Cohort (5,518)"],
            index=0
        )
    with col_filter2:
        search_id = st.text_input("Search by Sample ID (e.g. #1, #666, #1584):", "")
    
    display_cols = [
        "Sample ID", "age", "week", "fetal", "reads", "peak", "QC",
        "T13_Z_run", "T13_Target", "T18_Z_run", "T18_Target", "T21_Z_run", "T21_Target"
    ]
    
    if search_id.strip():
        filtered_df = master_df[master_df["Sample ID"].str.contains(search_id.strip(), case=False, na=False)]
    elif filter_opt == "All Confirmed Trisomies (47)":
        filtered_df = master_df[(master_df["T21_Target"] == 1) | (master_df["T18_Target"] == 1) | (master_df["T13_Target"] == 1)]
    elif filter_opt == "QC-Filtered Cohort (195)":
        filtered_df = master_df[master_df["QC"] == "QC-filtered"]
    elif filter_opt == "Complete Cohort (5,518)":
        filtered_df = master_df
    else:
        filtered_df = master_df.head(50)

    st.dataframe(filtered_df[display_cols], use_container_width=True)
    st.caption(f"Displaying {len(filtered_df):,} of {len(master_df):,} samples.")
