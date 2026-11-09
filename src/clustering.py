"""
Unsupervised NIPT Sample Profiling Module
=========================================
Performs exploratory, unsupervised multi-dimensional profiling of NIPT samples
across assay quality, patient gestational metrics, and chromosomal dosage context.

CRITICAL CLINICAL SAFEGUARDS:
- Strictly exploratory and unsupervised.
- Never used to predict or assign medical diagnoses.
- Clusters are never labeled with disease names (e.g. Down syndrome, high-risk fetus).
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


PROFILING_FEATURES = [
    "age",
    "week",
    "fetal",
    "reads",
    "peak",
    "T13_Z_chr_vs_n",
    "T18_Z_chr_vs_n",
    "T21_Z_chr_vs_n"
]


def extract_and_scale_profiling_features(master_df: pd.DataFrame) -> Tuple[np.ndarray, RobustScaler, SimpleImputer]:
    """
    Extracts the 8 non-diagnostic profiling features, handles invalid fetal values,
    and applies RobustScaler.
    """
    X = master_df[PROFILING_FEATURES].copy()
    
    # Transparently handle invalid negative fetal values with NaN for imputation
    if "fetal" in X.columns:
        X.loc[X["fetal"] < 0, "fetal"] = np.nan

    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = RobustScaler(with_centering=True, with_scaling=True)
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, scaler, imputer


def evaluate_cluster_diagnostics(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> Dict[str, Any]:
    """
    Evaluates K = 2 through K = 8 across Inertia, Silhouette Score, and Davies-Bouldin Index.
    Does NOT hard-code cluster count before evaluation.
    """
    inertias = []
    silhouette_scores = []
    davies_bouldin_indices = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=15, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(float(kmeans.inertia_))
        sil = float(silhouette_score(X_scaled, labels, sample_size=3000, random_state=42))
        db = float(davies_bouldin_score(X_scaled, labels))
        
        silhouette_scores.append(sil)
        davies_bouldin_indices.append(db)

    # Automated selection criterion:
    # Pick K that maximizes Silhouette Score within clinical plausibility window (e.g. K in [3, 6])
    # or balances high silhouette and low Davies-Bouldin index
    best_idx = int(np.argmax(silhouette_scores))
    optimal_k = list(k_range)[best_idx]

    return {
        "k_evaluated": list(k_range),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_indices": davies_bouldin_indices,
        "optimal_k_selected": optimal_k,
        "selection_rationale": (
            f"K={optimal_k} achieved the highest Silhouette Score ({silhouette_scores[best_idx]:.4f}) "
            f"with a Davies-Bouldin Index of {davies_bouldin_indices[best_idx]:.4f}."
        )
    }


def fit_nipt_sample_profiling(
    master_df: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[KMeans, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fits final KMeans clustering using the diagnosed optimal K.
    Produces 2D PCA projections and technical (non-diagnostic) profile summaries.
    """
    X_scaled, scaler, imputer = extract_and_scale_profiling_features(master_df)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 2D PCA projection for visualization
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "Sample ID": master_df["Sample ID"].values,
        "PCA_Dim1": pca_coords[:, 0],
        "PCA_Dim2": pca_coords[:, 1],
        "Profile_Cluster": cluster_labels,
        "QC": master_df["QC"].values,
        "fetal": master_df["fetal"].values,
        "reads": master_df["reads"].values,
        "peak": master_df["peak"].values,
        "age": master_df["age"].values,
        "week": master_df["week"].values
    })

    # Cluster characterization in original feature space (median & IQR)
    df_imputed = pd.DataFrame(
        imputer.transform(master_df[PROFILING_FEATURES].replace([np.inf, -np.inf], np.nan)),
        columns=PROFILING_FEATURES
    )
    df_imputed["Profile_Cluster"] = cluster_labels

    cluster_medians = df_imputed.groupby("Profile_Cluster").median().reset_index()
    cluster_counts = df_imputed["Profile_Cluster"].value_counts().sort_index().to_dict()

    metadata = {
        "n_clusters": n_clusters,
        "pca_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cluster_sample_counts": cluster_counts,
        "features_used": PROFILING_FEATURES,
        "non_diagnostic_disclaimer": (
            "These clusters describe technical assay profiles and patient demographics. "
            "They do NOT indicate fetal chromosomal aneuploidy or medical diagnoses."
        )
    }

    return kmeans, pca_df, cluster_medians, metadata
