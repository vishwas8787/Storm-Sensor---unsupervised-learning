"""Run the complete reproducible NOAA extreme-weather clustering analysis.

Implements Unit III & IV Unsupervised Learning:
- NOAA Storm Events feature engineering and preprocessing documentation
- Hierarchical Clustering: Ward, Complete, Average, Single across k=3..6
- Density-Based Clustering: DBSCAN parameter grid and k-distance knee analysis
- PCA 2D/3D Dimensionality Reduction, scree analysis, and loadings biplot
- Internal validation: Silhouette, Davies-Bouldin, Calinski-Harabasz, Cophenetic correlation
- Adjusted Rand Index (ARI) stability matrix
- Cluster profiling in original units and responsible AI interpretation report
- Export consolidated JSON dataset for the interactive web frontend
"""

import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.eda import run_eda
from src.feature_engineering import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, engineer_features
from src.hierarchical import run_hierarchical
from src.dbscan import run_dbscan
from src.dimensionality_reduction import fit_pca, plot_pca, plot_scree
from src.evaluation import stability_table, plot_silhouettes, final_comparison
from src.interpretation import write_profiles, write_responsible_interpretation
from src.visualization import plot_stability
from src.preprocessing import preprocess_data


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed" / "cleaned_storm_data.csv"
RAW = ROOT / "data" / "raw" / "StormEvents_details_2010_2020.csv"
PLOTS = ROOT / "outputs" / "plots"
METRICS = ROOT / "outputs" / "metrics"
PROFILES = ROOT / "outputs" / "cluster_profiles"
RANDOM_STATE = 42
SAMPLE_SIZE = 2500

LOAD_COLUMNS = [
    "EVENT_ID", "YEAR", "EVENT_TYPE", "STATE", "CZ_NAME", "BEGIN_DATE_TIME", "END_DATE_TIME",
    "INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT",
    "PROPERTY_DAMAGE_USD", "CROP_DAMAGE_USD", "BEGIN_LAT", "BEGIN_LON", "MAGNITUDE"
]


def load_analysis_data():
    """Load preprocessed NOAA dataset or trigger preprocessing if raw is newer."""
    processed_is_current = (
        PROCESSED.exists()
        and (not RAW.exists() or PROCESSED.stat().st_mtime >= RAW.stat().st_mtime)
    )
    if processed_is_current:
        source = PROCESSED
    elif RAW.exists():
        print("Processed dataset is missing or older than the raw NOAA file; preprocessing first.")
        raw_data = pd.read_csv(RAW, low_memory=False)
        preprocess_data(raw_data)
        del raw_data
        source = PROCESSED
    else:
        source = RAW

    if not source.exists():
        raise FileNotFoundError(
            "NOAA details CSV was not found. Place StormEvents_details_2010_2020.csv in data/raw."
        )

    print(f"Loading analysis columns from {source.name}...")
    df = pd.read_csv(source, usecols=lambda col: col in LOAD_COLUMNS, low_memory=False)
    for col in ["BEGIN_DATE_TIME", "END_DATE_TIME"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    for directory in [PLOTS, METRICS, PROFILES]:
        directory.mkdir(parents=True, exist_ok=True)

    full = load_analysis_data()
    print("Loaded full dataset shape:", full.shape)
    total_rows = len(full)
    year_min, year_max = int(full["YEAR"].min()), int(full["YEAR"].max())

    # Reproducible random sample for pairwise distance-based clustering
    sample = full.sample(n=min(SAMPLE_SIZE, len(full)), random_state=RANDOM_STATE).copy()
    sample = engineer_features(sample).reset_index(drop=True)

    # Run Exploratory Data Analysis
    print("Running EDA...")
    run_eda(full, PLOTS, METRICS, sample)
    del full

    # Feature matrix preparation
    raw_X = sample[FEATURE_COLUMNS].copy()
    missing_counts = raw_X.isna().sum()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_X)
    X = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

    # Preprocessing documentation
    pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "missing_in_sample": missing_counts.values,
        "missing_percentage": (missing_counts / len(sample) * 100).values,
        "transformation": ["log1p(count)" if "INJURIES" in f or "DEATHS" in f else
                            "log1p(USD)" if "DAMAGE" in f else
                            "log1p(hours)" if "DURATION" in f else
                            "cyclical sin/cos(2*pi*m/12)" for f in FEATURE_COLUMNS],
        "imputation": ["none (defaults to 0.0)" if "INJURIES" in f or "DEATHS" in f else
                       "0.0 USD (unquantified/negligible damage)" if "DAMAGE" in f else
                       "sample median duration" if "DURATION" in f else "month derived" for f in FEATURE_COLUMNS],
        "standardization": "StandardScaler (mean=0, std=1)"
    }).to_csv(METRICS / "clustering_feature_preprocessing.csv", index=False)

    print(f"Clustering sample: {len(sample)} rows; features scaled: {X.shape[1]}")

    # Hierarchical Clustering
    print("Executing Hierarchical Clustering experiments...")
    hierarchy, hierarchy_labels = run_hierarchical(X.to_numpy(), sample, PLOTS, METRICS)

    # Density-Based Clustering (DBSCAN)
    print("Executing DBSCAN experiments...")
    density, density_labels = run_dbscan(X.to_numpy(), PLOTS, METRICS)

    # Evaluation & Stability
    print("Evaluating metrics and cluster stability...")
    plot_silhouettes(hierarchy, density, PLOTS)
    comparison = final_comparison(hierarchy, density, METRICS)

    label_sets = {**hierarchy_labels, **density_labels}
    ari_matrix, ari_names = stability_table(label_sets, METRICS)
    plot_stability(ari_matrix, ari_names, PLOTS / "stability_ari.png")

    # Model Selection: Hierarchical
    # Screen out singleton / chaining solutions (smallest cluster < 1% of sample)
    min_cluster_threshold = max(5, int(0.01 * len(sample)))
    valid_h = hierarchy[hierarchy["smallest_cluster"] >= min_cluster_threshold]
    if valid_h.empty:
        valid_h = hierarchy.sort_values(["smallest_cluster", "silhouette_evaluation_sample"], ascending=[False, False])
        best_h_name = valid_h.iloc[0]["configuration"]
        hierarchical_note = "Selected highest cluster size balance due to small outlier splits."
    else:
        # Prefer Ward linkage with highest silhouette among non-singleton solutions
        ward_candidates = valid_h[valid_h["method"] == "ward"]
        if not ward_candidates.empty:
            best_h_name = ward_candidates.sort_values("silhouette_evaluation_sample", ascending=False).iloc[0]["configuration"]
            hierarchical_note = f"Selected highest Silhouette among Ward linkage settings meeting minimum cluster threshold (>= {min_cluster_threshold} points)."
        else:
            best_h_name = valid_h.sort_values("silhouette_evaluation_sample", ascending=False).iloc[0]["configuration"]
            hierarchical_note = f"Selected highest Silhouette among non-singleton hierarchical solutions."

    selected_h_row = hierarchy.loc[hierarchy["configuration"] == best_h_name].iloc[0]

    # Model Selection: DBSCAN
    # Require at least 2 non-noise clusters, smallest cluster >= 15 points (to avoid trivial density split), and noise < 25%
    min_dbscan_cluster_size = 15
    valid_d = density[
        (density["clusters_excluding_noise"] >= 2) &
        (density["noise_percentage"] < 25.0) &
        (density["smallest_cluster"] >= min_dbscan_cluster_size) &
        density["silhouette_non_noise_evaluation_sample"].notna()
    ]
    if valid_d.empty:
        best_d_name = density.sort_values("noise_percentage").iloc[0]["configuration"]
        dbscan_note = "Selected lowest noise configuration as diagnostic."
    else:
        best_d_name = valid_d.sort_values("silhouette_non_noise_evaluation_sample", ascending=False).iloc[0]["configuration"]
        dbscan_note = f"Selected highest non-noise Silhouette among settings with >= 2 clusters, smallest cluster >= {min_dbscan_cluster_size}, and < 25% noise."

    selected_d_row = density.loc[density["configuration"] == best_d_name].iloc[0]

    # Mark selected in comparison
    comparison["selected_for_profile"] = comparison["configuration"].isin([best_h_name, best_d_name])
    comparison.to_csv(METRICS / "final_comparison.csv", index=False)

    # Dimensionality Reduction: PCA
    print("Fitting PCA and generating projection views...")
    pca, coords, loadings = fit_pca(X, n_components=3)
    loadings.to_csv(METRICS / "pca_loadings.csv", index=False)

    explained = pd.DataFrame({
        "component": ["PC1", "PC2", "PC3"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    explained.to_csv(METRICS / "pca_explained_variance.csv", index=False)
    plot_scree(pca, PLOTS / "pca_scree.png")

    # Plot PCA for chosen models with biplot arrows
    plot_pca(
        coords,
        hierarchy_labels[best_h_name],
        f"Hierarchical Clustering in PCA Space ({best_h_name})",
        PLOTS / "hierarchical_pca.png",
        loadings=loadings
    )
    plot_pca(
        coords,
        density_labels[best_d_name],
        f"DBSCAN Clustering in PCA Space ({best_d_name})\nNoise = -1",
        PLOTS / "dbscan_pca.png",
        loadings=loadings
    )

    # Write Profiles & Interpretation
    print("Generating cluster profiles and responsible interpretation...")
    h_profiles, h_comp = write_profiles(sample, hierarchy_labels[best_h_name], "hierarchical", PROFILES, PLOTS)
    d_profiles, d_comp = write_profiles(sample, density_labels[best_d_name], "dbscan", PROFILES, PLOTS)
    write_responsible_interpretation(METRICS)

    # Pairwise ARI between selected models and neighbors
    ward_k3_k4_ari = adjusted_rand_score(hierarchy_labels.get("ward_k3", []), hierarchy_labels.get("ward_k4", []))
    cross_method_ari = adjusted_rand_score(hierarchy_labels[best_h_name], density_labels[best_d_name])

    # Save selection notes
    (METRICS / "selection_notes.txt").write_text(
        f"NOAA Storm Events Grouping - Model Selection Notes\n"
        f"===================================================\n"
        f"Dataset Period: {year_min}-{year_max} | Total Records: {total_rows:,}\n"
        f"Evaluation Sample: {len(sample):,} events (random_state={RANDOM_STATE})\n\n"
        f"Feature Space: 7 standardized features (casualties, damage, duration, cyclical seasonality)\n"
        f"PCA 2-Component Variance Explained: {explained['cumulative_variance_ratio'].iloc[1]*100:.2f}%\n"
        f"PCA 3-Component Variance Explained: {explained['cumulative_variance_ratio'].iloc[2]*100:.2f}%\n\n"
        f"Selected Hierarchical Model: {best_h_name}\n"
        f"- Selection Rule: {hierarchical_note}\n"
        f"- Clusters: {selected_h_row['actual_clusters']}\n"
        f"- Silhouette Score: {selected_h_row['silhouette_evaluation_sample']:.4f}\n"
        f"- Davies-Bouldin Index: {selected_h_row['davies_bouldin_index']:.4f}\n"
        f"- Calinski-Harabasz Index: {selected_h_row['calinski_harabasz_index']:.2f}\n"
        f"- Smallest Cluster Size: {selected_h_row['smallest_cluster']} events\n"
        f"- ARI Stability (Ward k3 vs k4): {ward_k3_k4_ari:.4f}\n\n"
        f"Selected DBSCAN Model: {best_d_name}\n"
        f"- Selection Rule: {dbscan_note}\n"
        f"- Clusters (excl. noise): {selected_d_row['clusters_excluding_noise']}\n"
        f"- Noise Count: {selected_d_row['noise_count']} ({selected_d_row['noise_percentage']:.2f}%)\n"
        f"- Non-noise Silhouette: {selected_d_row['silhouette_non_noise_evaluation_sample']:.4f}\n"
        f"- Davies-Bouldin Index: {selected_d_row['davies_bouldin_index']:.4f}\n"
        f"- Calinski-Harabasz Index: {selected_d_row['calinski_harabasz_index']:.2f}\n\n"
        f"Cross-Method Agreement (Hierarchical vs DBSCAN ARI): {cross_method_ari:.4f}\n"
        f"Note: Exploratory data partition, not an immutable physical taxonomy.\n",
        encoding="utf-8"
    )

    # Attach cluster assignments and PCA coordinates to sample for dashboard export
    sample["HIERARCHICAL_CLUSTER"] = hierarchy_labels[best_h_name]
    sample["DBSCAN_CLUSTER"] = density_labels[best_d_name]
    sample["PC1"] = np.round(coords[:, 0], 3)
    sample["PC2"] = np.round(coords[:, 1], 3)
    sample["PC3"] = np.round(coords[:, 2], 3)

    # Centroids in standardized feature space for live prediction
    h_labels = hierarchy_labels[best_h_name]
    d_labels = density_labels[best_d_name]

    h_centroids = {}
    for c in np.unique(h_labels):
        h_centroids[str(c)] = X.iloc[h_labels == c].mean().to_dict()

    d_centroids = {}
    for c in np.unique(d_labels):
        if c != -1:
            d_centroids[str(c)] = X.iloc[d_labels == c].mean().to_dict()

    # Export consolidated dashboard data for fast web frontend loading
    dashboard_payload = {
        "summary": {
            "total_records": total_rows,
            "sample_size": len(sample),
            "years": f"{year_min}-{year_max}",
            "best_hierarchical": best_h_name,
            "hierarchical_silhouette": float(selected_h_row["silhouette_evaluation_sample"]),
            "hierarchical_db": float(selected_h_row["davies_bouldin_index"]),
            "hierarchical_ch": float(selected_h_row["calinski_harabasz_index"]),
            "hierarchical_clusters": int(selected_h_row["actual_clusters"]),
            "best_dbscan": best_d_name,
            "dbscan_silhouette": float(selected_d_row["silhouette_non_noise_evaluation_sample"]),
            "dbscan_db": float(selected_d_row["davies_bouldin_index"]),
            "dbscan_ch": float(selected_d_row["calinski_harabasz_index"]),
            "dbscan_clusters": int(selected_d_row["clusters_excluding_noise"]),
            "dbscan_noise_pct": float(selected_d_row["noise_percentage"]),
            "pca_variance_2d": float(explained["cumulative_variance_ratio"].iloc[1]),
            "pca_variance_3d": float(explained["cumulative_variance_ratio"].iloc[2]),
            "cross_method_ari": float(cross_method_ari),
        },
        "model_params": {
            "features": FEATURE_COLUMNS,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "pca_components": pca.components_.tolist(),
            "pca_mean": pca.mean_.tolist(),
            "hierarchical_centroids": h_centroids,
            "dbscan_centroids": d_centroids,
            "selected_hierarchical": best_h_name,
            "selected_dbscan": best_d_name,
        },
        "pca_loadings": loadings.to_dict(orient="records"),
        "pca_variance": explained.to_dict(orient="records"),
        "hierarchical_results": hierarchy.to_dict(orient="records"),
        "dbscan_results": density.to_dict(orient="records"),
        "feature_preprocessing": pd.read_csv(METRICS / "clustering_feature_preprocessing.csv").to_dict(orient="records"),
        "stability_names": ari_names,
        "stability_matrix": ari_matrix.round(3).tolist(),
        "sample_points": sample[[
            "EVENT_ID", "EVENT_TYPE", "STATE", "CZ_NAME", "YEAR", "DURATION_HOURS",
            "TOTAL_INJURIES", "TOTAL_DEATHS", "PROPERTY_DAMAGE_USD", "CROP_DAMAGE_USD",
            "BEGIN_LAT", "BEGIN_LON", "PC1", "PC2", "PC3",
            "HIERARCHICAL_CLUSTER", "DBSCAN_CLUSTER"
        ]].dropna(subset=["PC1", "PC2"]).head(1200).to_dict(orient="records")
    }

    def clean_for_json(obj):
        if isinstance(obj, float):
            return None if (np.isnan(obj) or np.isinf(obj)) else obj
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return clean_for_json(obj.tolist())
        elif isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.floating)):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        return obj

    with open(ROOT / "outputs" / "dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(dashboard_payload), f)

    print("\nAnalysis complete!")
    print(f"Best Hierarchical: {best_h_name} (Silhouette: {selected_h_row['silhouette_evaluation_sample']:.4f})")
    print(f"Best DBSCAN: {best_d_name} (Silhouette: {selected_d_row['silhouette_non_noise_evaluation_sample']:.4f}, Noise: {selected_d_row['noise_percentage']:.1f}%)")
    print(f"PCA Variance Explained: {explained['cumulative_variance_ratio'].iloc[1]*100:.1f}% (2-Comp), {explained['cumulative_variance_ratio'].iloc[2]*100:.1f}% (3-Comp)")
    print(f"Dashboard data exported to: outputs/dashboard_data.json")


if __name__ == "__main__":
    main()
