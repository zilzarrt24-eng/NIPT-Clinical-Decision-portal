"""
Pipeline Execution Script for NIPT Data Preprocessing
=====================================================
Executes validation, merges raw sheets, generates modeling datasets,
and writes comprehensive validation reports.
"""

from pathlib import Path
import sys
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import NIPTPreprocessor


def main():
    raw_path = PROJECT_ROOT / "data" / "raw" / "pone.0207840.s004 (1).xlsx"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "reports"

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NIPT CLINICAL DECISION PORTAL: DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Reading raw data from: {raw_path}")

    preprocessor = NIPTPreprocessor(raw_path)

    # 1. Ingestion & Validation
    print("\n[Step 1/5] Ingesting and validating raw sheets (chr13, chr18, chr21)...")
    raw_dfs = preprocessor.load_and_validate_raw_sheets()
    print("  -> Passed: Uniqueness of Sample IDs verified.")
    print("  -> Passed: Cross-sheet sample concordance verified (5,518 identical samples).")
    print("  -> Passed: Shared metadata consistency verified.")

    # 2. Build Master Processed Dataset
    print("\n[Step 2/5] Merging into master sample-level dataframe...")
    master_df = preprocessor.build_master_dataset(raw_dfs)
    master_out_path = processed_dir / "master_processed.csv"
    master_df.to_csv(master_out_path, index=False)
    print(f"  -> Saved master processed dataset: {master_out_path} ({master_df.shape[0]} rows, {master_df.shape[1]} cols)")

    # 3. Build Aneuploidy Modeling Datasets
    print("\n[Step 3/5] Building chromosome-specific modeling datasets (T13, T18, T21)...")
    t13_df, t13_features, t13_target = preprocessor.build_aneuploidy_modeling_dataset(master_df, "T13")
    t18_df, t18_features, t18_target = preprocessor.build_aneuploidy_modeling_dataset(master_df, "T18")
    t21_df, t21_features, t21_target = preprocessor.build_aneuploidy_modeling_dataset(master_df, "T21")

    t13_path = processed_dir / "t13_dataset.csv"
    t18_path = processed_dir / "t18_dataset.csv"
    t21_path = processed_dir / "t21_dataset.csv"

    t13_df.to_csv(t13_path, index=False)
    t18_df.to_csv(t18_path, index=False)
    t21_df.to_csv(t21_path, index=False)

    print(f"  -> T13 Dataset: {t13_path} ({t13_df.shape[0]} rows, {t13_df.shape[1]} cols | Positives: {t13_df[t13_target].sum()})")
    print(f"  -> T18 Dataset: {t18_path} ({t18_df.shape[0]} rows, {t18_df.shape[1]} cols | Positives: {t18_df[t18_target].sum()})")
    print(f"  -> T21 Dataset: {t21_path} ({t21_df.shape[0]} rows, {t21_df.shape[1]} cols | Positives: {t21_df[t21_target].sum()})")

    # 4. Generate Preprocessing Report
    print("\n[Step 4/5] Generating validation and preprocessing audit report...")
    report_content = preprocessor.generate_preprocessing_report(master_df, t13_df, t18_df, t21_df)
    report_path = reports_dir / "preprocessing_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  -> Saved report: {report_path}")

    # 5. Sanity Verification Checks
    print("\n[Step 5/5] Performing post-generation integrity checks...")
    assert len(master_df) == 5518, "Master row count mismatch!"
    assert len(t13_df) == 5518, "T13 row count mismatch!"
    assert len(t18_df) == 5518, "T18 row count mismatch!"
    assert len(t21_df) == 5518, "T21 row count mismatch!"
    assert master_df["fetal_invalid_flag"].sum() == 1, "Expected 1 invalid fetal fraction flag!"
    assert t13_df["T13_Target"].sum() == 5, "Expected 5 T13 positives!"
    assert t18_df["T18_Target"].sum() == 15, "Expected 15 T18 positives!"
    assert t21_df["T21_Target"].sum() == 27, "Expected 27 T21 positives!"
    print("  -> All integrity checks passed successfully!")
    print("=" * 60)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 60)


if __name__ == "__main__":
    main()
