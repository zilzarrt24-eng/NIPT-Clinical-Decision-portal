"""
Model Training and Evaluation Pipeline Orchestrator
==================================================
Coordinates cross-validation, baseline comparisons, Model B vs Model C ablation,
NIPT sample profiling, artifact serialization, and evaluation report generation.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from src.models import (
    build_calibrated_pipeline,
    sanitize_input_features,
    get_candidate_features
)
from src.evaluation import (
    evaluate_model_a_baseline,
    run_stratified_cv_evaluation,
    perform_ablation_comparison
)
from src.clustering import (
    extract_and_scale_profiling_features,
    evaluate_cluster_diagnostics,
    fit_nipt_sample_profiling
)


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def train_and_evaluate_all():
    """
    Executes complete training and evaluation pipeline across T13, T18, and T21.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NIPT CLINICAL DECISION PORTAL: MACHINE LEARNING EVALUATION PIPELINE")
    print("=" * 70)

    # Load preprocessed datasets
    master_df = pd.read_csv(PROCESSED_DIR / "master_processed.csv")
    datasets = {
        "T13": pd.read_csv(PROCESSED_DIR / "t13_dataset.csv"),
        "T18": pd.read_csv(PROCESSED_DIR / "t18_dataset.csv"),
        "T21": pd.read_csv(PROCESSED_DIR / "t21_dataset.csv"),
    }

    all_metrics = {}
    all_ablations = {}

    for aneuploidy in ["T13", "T18", "T21"]:
        print(f"\n[{aneuploidy}] Starting 3-Tier Model Evaluation...")
        df = datasets[aneuploidy]
        aneuploidy_dir = MODELS_DIR / aneuploidy.lower()
        aneuploidy_dir.mkdir(parents=True, exist_ok=True)

        # 1. Model A: Conventional Z_run Baseline
        print(f"  -> [Model A] Conventional Z_run Baseline screening evaluation...")
        metrics_a = evaluate_model_a_baseline(df, aneuploidy)

        # 2. Model B: 6 Z-scores + RBF SVM (Stratified CV)
        print(f"  -> [Model B] 6 Z-Score RBF-SVM with explicit Platt calibration (5-fold CV)...")
        metrics_b, oof_b = run_stratified_cv_evaluation(df, aneuploidy, feature_set="B")

        # 3. Model C: 6 Z-scores + 5 Clinical Features + RBF SVM (Stratified CV)
        print(f"  -> [Model C] Extended Multivariate RBF-SVM (5-fold CV)...")
        metrics_c, oof_c = run_stratified_cv_evaluation(df, aneuploidy, feature_set="C")

        # 4. Model B vs Model C Ablation Analysis
        print(f"  -> Performing objective Model B vs Model C ablation benchmark...")
        ablation = perform_ablation_comparison(metrics_b, metrics_c, oof_b, oof_c, aneuploidy)
        all_ablations[aneuploidy] = ablation

        # Merge OOF predictions
        oof_merged = pd.DataFrame({
            "Sample ID": df["Sample ID"],
            "Target": df[f"{aneuploidy}_Target"],
            "Z_run": df[f"{aneuploidy}_Z_run"],
            "Prob_Model_B": oof_b["prob_model_b"],
            "Prob_Model_C": oof_c["prob_model_c"],
            "QC": df["QC"]
        })
        oof_merged.to_csv(aneuploidy_dir / "oof_predictions.csv", index=False)

        # 5. Fit Production Models on Full Dataset (for interactive inference in Streamlit)
        print(f"  -> Fitting production calibrated pipelines for portal inference...")
        y = df[f"{aneuploidy}_Target"].values
        
        # Production Model B
        feat_b = get_candidate_features(aneuploidy, "B")
        X_b = sanitize_input_features(df, feat_b)
        prod_b = build_calibrated_pipeline("B", cv=3)
        prod_b.fit(X_b, y)
        joblib.dump(prod_b, aneuploidy_dir / "model_b_pipeline.joblib")

        # Production Model C
        feat_c = get_candidate_features(aneuploidy, "C")
        X_c = sanitize_input_features(df, feat_c)
        prod_c = build_calibrated_pipeline("C", cv=3)
        prod_c.fit(X_c, y)
        joblib.dump(prod_c, aneuploidy_dir / "model_c_pipeline.joblib")

        # Store metrics
        metrics_summary = {
            "Model_A": metrics_a,
            "Model_B": metrics_b,
            "Model_C": metrics_c,
            "Ablation_B_vs_C": ablation
        }
        with open(aneuploidy_dir / "metrics_summary.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)

        all_metrics[aneuploidy] = metrics_summary

    # Save overall ablation summary
    with open(MODELS_DIR / "ablation_summary.json", "w") as f:
        json.dump(all_ablations, f, indent=2)

    # 6. Unsupervised NIPT Sample Profiling (K-Means)
    print("\n[Clustering] Executing Unsupervised NIPT Sample Profiling...")
    kmeans_dir = MODELS_DIR / "kmeans"
    kmeans_dir.mkdir(parents=True, exist_ok=True)

    X_scaled, scaler, imputer = extract_and_scale_profiling_features(master_df)
    joblib.dump(scaler, kmeans_dir / "clustering_scaler.joblib")
    joblib.dump(imputer, kmeans_dir / "clustering_imputer.joblib")

    print("  -> Evaluating K = 2 through K = 8 diagnostics (Inertia, Silhouette, Davies-Bouldin)...")
    diag = evaluate_cluster_diagnostics(X_scaled, range(2, 9))
    with open(kmeans_dir / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    
    optimal_k = diag["optimal_k_selected"]
    print(f"  -> Diagnosed optimal cluster count: K = {optimal_k} ({diag['selection_rationale']})")

    kmeans, pca_df, cluster_medians, meta = fit_nipt_sample_profiling(master_df, optimal_k)
    joblib.dump(kmeans, kmeans_dir / "kmeans_model.joblib")
    pca_df.to_csv(kmeans_dir / "pca_projection.csv", index=False)
    cluster_medians.to_csv(kmeans_dir / "cluster_medians.csv", index=False)
    with open(kmeans_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 7. Generate Comprehensive Markdown Report
    print("\n[Reporting] Generating model evaluation and ablation markdown report...")
    generate_markdown_evaluation_report(all_metrics, all_ablations, diag, meta)

    print("\n" + "=" * 70)
    print("ALL MODELS AND ARTIFACTS PRODUCED SUCCESSFULLY.")
    print("=" * 70)


def generate_markdown_evaluation_report(all_metrics, all_ablations, diag, cluster_meta):
    """
    Constructs a detailed evaluation report.
    """
    report_lines = [
        "# NIPT Machine Learning Model Evaluation & Ablation Report",
        "**Cohort:** 5,518 clinical samples (`pone.0207840.s004 (1).xlsx`)",
        "**Three-Tier Comparison:** Model A (Z_run Baseline) vs Model B (6 Z-scores RBF-SVM) vs Model C (Extended Multivariate RBF-SVM)",
        "",
        "---",
        "",
        "## 1. Executive Summary & Objective Ablation Findings",
        "",
        "| Aneuploidy | N Positives | Model A PR-AUC | Model B PR-AUC | Model C PR-AUC | ΔPR-AUC (C - B) | Ablation Assessment |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for aneu in ["T13", "T18", "T21"]:
        m = all_metrics[aneu]
        abl = all_ablations[aneu]
        pos = m["Model_B"]["n_positives"]
        pa = m["Model_A"]["pr_auc"]
        pb = m["Model_B"]["pr_auc"]
        pc = m["Model_C"]["pr_auc"]
        delta = abl["delta_pr_auc"]
        concl = abl["objective_ablation_conclusion"]
        report_lines.append(f"| **{aneu}** | {pos} | {pa:.4f} | {pb:.4f} | {pc:.4f} | {delta:+.4f} | {concl} |")

    report_lines.extend([
        "",
        "> **Important Clinical Terminology Standard:** All model outputs are evaluated as *model-estimated screening probabilities* and *experimental screening outputs*, not as definitive medical diagnoses.",
        "",
        "---",
        "",
        "## 2. Leakage-Safe Operating Point Evaluation (Primary Methodology)",
        "",
        "### Methodological Rationale & Leakage-Safe Design:",
        "- **Issue with Pooled-OOF Cutoffs:** In the initial audit, operating thresholds (99.0%, 99.5%, 99.9% specificity) were identified by sweeping the pooled out-of-fold predictions across all 5,518 samples and evaluating performance on those same samples. This introduced a subtle post-hoc threshold-selection optimism.",
        "- **Corrected Leakage-Safe Procedure:** In each outer cross-validation fold, the threshold targeting empirical specificity $\ge S$ is determined strictly using only training-fold predictions (`X_train`, `y_train`). That threshold is applied once to the held-out validation fold (`X_val`). Counts (TP, FP, TN, FN) are accumulated across the 5 independent validation folds.",
        "- **Validation Independence:** The validation fold has zero influence on threshold selection, ensuring true out-of-sample generalization.",
        "",
        "*(Note: Target specificities at 99.0%, 99.5%, and 99.9% are experimental analytical benchmarks, not clinical standards).*  ",
        ""
    ])

    for aneu in ["T13", "T18", "T21"]:
        report_lines.append(f"### {aneu} Leakage-Safe Fold-Derived Operating Points")
        m = all_metrics[aneu]
        report_lines.append("| Model | Target Spec | Mean Fold Cutoff | Sensitivity (95% CI) | Achieved Spec (95% CI) | PPV (95% CI) | NPV (95% CI) | TP / FP / TN / FN |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for m_key, m_name in [("Model_A", "Model A (Baseline)"), ("Model_B", "Model B (Z-scores)"), ("Model_C", "Model C (Extended)")]:
            ops = m[m_key].get("leakage_safe_operating_points", m[m_key].get("operating_points", {}))
            for op_label in ["99.0%", "99.5%", "99.9%"]:
                if op_label in ops:
                    op = ops[op_label]
                    c_val = f"{op.get('mean_fold_cutoff', 0.0):.4f}"
                    sens_str = f"{op['sensitivity']*100:.1f}% [{op['sensitivity_ci_95'][0]*100:.1f}%, {op['sensitivity_ci_95'][1]*100:.1f}%]"
                    spec_str = f"{op['specificity']*100:.2f}% [{op['specificity_ci_95'][0]*100:.2f}%, {op['specificity_ci_95'][1]*100:.2f}%]"
                    ppv_str = f"{op['ppv']*100:.1f}% [{op['ppv_ci_95'][0]*100:.1f}%, {op['ppv_ci_95'][1]*100:.1f}%]"
                    npv_str = f"{op['npv']*100:.3f}%"
                    counts_str = f"{op['tp']} / {op['fp']} / {op['tn']} / {op['fn']}"
                    report_lines.append(
                        f"| {m_name} | {op_label} | {c_val} | {sens_str} | {spec_str} | {ppv_str} | {npv_str} | {counts_str} |"
                    )
        report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "### Comparative Reference: Previous Pooled-OOF Post-Hoc Analytical Thresholds",
        "*(Retained strictly for methodological transparency and historical comparison; reflects post-hoc threshold selection on pooled OOF data)*",
        ""
    ])

    for aneu in ["T13", "T18", "T21"]:
        report_lines.append(f"#### {aneu} Pooled-OOF Post-Hoc Benchmarks (Descriptive Only)")
        m = all_metrics[aneu]
        report_lines.append("| Model | Target Spec | Cutoff | Sensitivity (95% CI) | Specificity | PPV (95% CI) | NPV | TP / FP / TN / FN |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for m_key, m_name in [("Model_A", "Model A (Baseline)"), ("Model_B", "Model B (Z-scores)"), ("Model_C", "Model C (Extended)")]:
            ops = m[m_key].get("pooled_oof_posthoc_operating_points", {})
            for op_label in ["99.0%", "99.5%", "99.9%"]:
                if op_label in ops:
                    op = ops[op_label]
                    c_val = f"{op.get('probability_cutoff', op.get('cutoff_z', 0.0)):.4f}"
                    sens_str = f"{op['sensitivity']*100:.1f}% [{op['sensitivity_ci_95'][0]*100:.1f}%, {op['sensitivity_ci_95'][1]*100:.1f}%]"
                    spec_str = f"{op['specificity']*100:.2f}%"
                    ppv_str = f"{op['ppv']*100:.1f}% [{op['ppv_ci_95'][0]*100:.1f}%, {op['ppv_ci_95'][1]*100:.1f}%]"
                    npv_str = f"{op['npv']*100:.3f}%"
                    counts_str = f"{op['tp']} / {op['fp']} / {op['tn']} / {op['fn']}"
                    report_lines.append(
                        f"| {m_name} | {op_label} | {c_val} | {sens_str} | {spec_str} | {ppv_str} | {npv_str} | {counts_str} |"
                    )
        report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## 3. Statistical Uncertainty & Rare Class Limitations",
        "",
        "- **Trisomy 13 ($N=5$ Positives):** PRELIMINARY AND EXPLORATORY ONLY. Repeated cross-validation resampling does NOT increase the number of independent positive samples. Each missed case shifts measured sensitivity by 20.0%.",
        "- **Trisomy 18 ($N=15$ Positives):** Small positive cohort; 3 positive cases per fold.",
        "- **Trisomy 21 ($N=27$ Positives):** Moderate positive cohort; 5–6 positive cases per fold.",
        "",
        "---",
        "",
        "## 4. Unsupervised NIPT Sample Profiling (K-Means)",
        f"- **Optimal Clusters Selected:** K = {diag['optimal_k_selected']}",
        f"- **Selection Criteria:** {diag['selection_rationale']}",
        f"- **Cluster Distribution:** {json.dumps(cluster_meta['cluster_sample_counts'])}",
        "- **Non-Diagnostic Policy:** Clusters reflect library cfDNA concentration, sequencing depth, and patient demographics. No disease diagnoses are assigned.",
        ""
    ])

    report_path = REPORTS_DIR / "model_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    train_and_evaluate_all()
