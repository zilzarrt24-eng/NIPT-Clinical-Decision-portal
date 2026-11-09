"""
Model Definitions and Pipeline Factory for NIPT Screening
=========================================================
Implements Model B (6 Z-score features) and Model C (extended multivariate)
using non-linear RBF SVM with explicit Platt calibration via CalibratedClassifierCV.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


def get_candidate_features(aneuploidy: str, feature_set: str) -> List[str]:
    """
    Returns candidate predictor names for a specific aneuploidy and model level.
    Strictly excludes: Sample ID, Real_state, Target, Group, QC, costday,
    fetal_invalid_flag, and Z_run.
    """
    prefix = f"{aneuploidy}_"
    z_score_features = [
        f"{prefix}Z_baseline_vs_n",
        f"{prefix}Z_baseline_vs_p",
        f"{prefix}Z_chr_vs_n",
        f"{prefix}Z_chr_vs_p",
        f"{prefix}Z_sample_vs_n",
        f"{prefix}Z_sample_vs_p"
    ]
    
    if feature_set.upper() == "B":
        return z_score_features
    elif feature_set.upper() == "C":
        clinical_features = ["fetal", "age", "week", "reads", "peak"]
        return z_score_features + clinical_features
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Expected 'B' or 'C'.")


def build_svm_pipeline(feature_set: str, random_state: int = 42) -> Pipeline:
    """
    Constructs an uncalibrated SVM pipeline.
    
    Uses SVC(probability=False) to ensure calibration is handled explicitly
    via CalibratedClassifierCV without conflicting internal CV in SVC.
    """
    svc_base = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,  # Explicitly False per design modification
        random_state=random_state
    )

    if feature_set.upper() == "B":
        steps = [
            ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
            ("classifier", svc_base)
        ]
    elif feature_set.upper() == "C":
        steps = [
            # Handle invalid fetal fraction values (< 0 flagged as NaN) transparently
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
            ("classifier", svc_base)
        ]
    else:
        raise ValueError(f"Invalid feature_set: {feature_set}")

    return Pipeline(steps)


def build_calibrated_pipeline(feature_set: str, cv: int = 3, random_state: int = 42) -> CalibratedClassifierCV:
    """
    Wraps the uncalibrated SVM pipeline with CalibratedClassifierCV using Platt sigmoid calibration.
    """
    base_pipeline = build_svm_pipeline(feature_set=feature_set, random_state=random_state)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=cv
    )
    return calibrated_model


def sanitize_input_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Extracts features and transparently replaces negative/invalid fetal fraction with NaN
    so the pipeline imputer can handle it without silent clamping or data leakage.
    """
    X = df[feature_cols].copy()
    if "fetal" in X.columns:
        # Replace non-physical fetal fractions (< 0) with NaN for fold-fit imputation
        X.loc[X["fetal"] < 0, "fetal"] = np.nan
    return X
