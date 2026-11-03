"""Run the complete local NOAA extreme-weather clustering analysis."""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.eda import run_eda
from src.feature_engineering import FEATURE_COLUMNS, engineer_features
from src.hierarchical import run_hierarchical
from src.dbscan import run_dbscan
from src.dimensionality_reduction import fit_pca, plot_pca
from src.evaluation import stability_table, plot_silhouettes, final_comparison
from src.interpretation import write_profiles
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
    "EVENT_ID", "YEAR", "EVENT_TYPE", "STATE", "BEGIN_DATE_TIME", "END_DATE_TIME",
    "INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT",
    "PROPERTY_DAMAGE_USD", "CROP_DAMAGE_USD",
]


def load_analysis_data():
    processed_is_current = (
        PROCESSED.exists()
        and (not RAW.exists() or PROCESSED.stat().st_mtime >= RAW.stat().st_mtime)
    )
    if processed_is_current:
        source = PROCESSED
    elif RAW.exists():
        print("Processed dataset is missing or older than the raw NOAA file; preprocessing it first.")
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
    print(f"Loading analysis columns from {source}")
    df = pd.read_csv(source, usecols=lambda col: col in LOAD_COLUMNS, low_memory=False)
    for column in ["BEGIN_DATE_TIME", "END_DATE_TIME"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    for directory in [PLOTS, METRICS, PROFILES]:
        directory.mkdir(parents=True, exist_ok=True)
    full = load_analysis_data()
    print("Loaded shape:", full.shape)
    total_rows = len(full)
    year_min, year_max = int(full["YEAR"].min()), int(full["YEAR"].max())

    # EDA uses all rows; expensive clustering uses one reproducible random sample.
    sample = full.sample(n=min(SAMPLE_SIZE, len(full)), random_state=RANDOM_STATE).copy()
    sample = engineer_features(sample).reset_index(drop=True)
    run_eda(full, PLOTS, METRICS, sample)
    del full

    raw_X = sample[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    missing_counts = raw_X.isna().sum()
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputed = imputer.fit_transform(raw_X)
    feature_names = imputer.get_feature_names_out(FEATURE_COLUMNS)
    scaler = StandardScaler()
    X = scaler.fit_transform(imputed)
    X = pd.DataFrame(X, columns=feature_names)
    pd.DataFrame({"feature": FEATURE_COLUMNS, "missing_in_sample": missing_counts.values,
                  "missing_percentage": (missing_counts / len(sample) * 100).values,
                  "imputation": "sample median; missingness indicators added"})\
        .to_csv(METRICS / "clustering_feature_preprocessing.csv", index=False)
    print(f"Clustering sample: {len(sample)} rows; model features after imputation/scaling: {X.shape[1]}")

    hierarchy, hierarchy_labels = run_hierarchical(X.to_numpy(), sample, PLOTS, METRICS)
    density, density_labels = run_dbscan(X.to_numpy(), PLOTS, METRICS)
    plot_silhouettes(hierarchy, density, PLOTS)
    comparison = final_comparison(hierarchy, density, METRICS)
    label_sets = {**hierarchy_labels, **density_labels}
    ari, ari_names = stability_table(label_sets, METRICS)
    plot_stability(ari, ari_names, PLOTS / "stability_ari.png")

    valid_h = hierarchy.dropna(subset=["silhouette_evaluation_sample"])
    if valid_h.empty:
        raise RuntimeError("No hierarchical configuration produced a valid silhouette score.")
    minimum_h_size = max(2, int(np.ceil(0.02 * len(sample))))
    valid_h = valid_h[valid_h["smallest_cluster"] >= minimum_h_size]
    if valid_h.empty:
        valid_h = hierarchy.dropna(subset=["silhouette_evaluation_sample"]).sort_values(
            ["smallest_cluster", "silhouette_evaluation_sample"], ascending=[False, False]
        ).head(1)
        hierarchical_note = "No setting met the 2% minimum-cluster-size screen; selected the largest minimum cluster and treat its silhouette as descriptive only."
    else:
        hierarchical_note = f"Selected highest silhouette among settings whose smallest cluster has at least {minimum_h_size} observations (2% of sample), screening out singleton splits."
    best_h_name = valid_h.sort_values("silhouette_evaluation_sample", ascending=False).iloc[0]["configuration"]
    valid_d = density[(density["clusters_excluding_noise"] >= 2) &
                      (density["noise_percentage"] < 95) &
                      density["silhouette_non_noise_evaluation_sample"].notna()]
    dbscan_note = "Selected highest non-noise silhouette among settings with at least two clusters and under 95% noise."
    if valid_d.empty:
        candidates = density[density["clusters_excluding_noise"] >= 1].sort_values(
            ["noise_percentage", "clusters_excluding_noise"], ascending=[True, False])
        if candidates.empty:
            best_d_name = density.sort_values("noise_percentage").iloc[0]["configuration"]
            dbscan_note = "No setting formed a non-noise cluster; plotted the setting with the least noise as a diagnostic only."
        else:
            best_d_name = candidates.iloc[0]["configuration"]
            dbscan_note = "No setting met the silhouette/noise selection rule; plotted the least-noise clustering as diagnostic only."
    else:
        best_d_name = valid_d.sort_values("silhouette_non_noise_evaluation_sample", ascending=False).iloc[0]["configuration"]

    comparison["selected_for_profile"] = comparison["configuration"].isin([best_h_name, best_d_name])
    comparison.to_csv(METRICS / "final_comparison.csv", index=False)
    selected_h_row = hierarchy.loc[hierarchy["configuration"] == best_h_name].iloc[0]
    selected_d_row = density.loc[density["configuration"] == best_d_name].iloc[0]
    ward_neighbours = [name for name in hierarchy_labels if name.startswith("ward_k") and name != best_h_name]
    ward_neighbour = min(
        ward_neighbours,
        key=lambda name: abs(int(name.rsplit("k", 1)[1]) - int(best_h_name.rsplit("k", 1)[1])),
    ) if ward_neighbours and best_h_name.startswith("ward_k") else ""
    if not ward_neighbour:
        ward_neighbour = "ward_k4" if best_h_name != "ward_k4" else "ward_k3"
    ward_neighbour_ari = adjusted_rand_score(hierarchy_labels[best_h_name], hierarchy_labels[ward_neighbour])
    same_min_samples = density[
        (density["min_samples"] == selected_d_row["min_samples"])
        & (density["configuration"] != best_d_name)
    ].copy()
    if same_min_samples.empty:
        same_min_samples = density[density["configuration"] != best_d_name].copy()
    same_min_samples["eps_difference"] = (same_min_samples["eps"] - selected_d_row["eps"]).abs()
    dbscan_neighbour = same_min_samples.sort_values("eps_difference").iloc[0]["configuration"]
    dbscan_neighbour_ari = adjusted_rand_score(
        density_labels[best_d_name], density_labels[dbscan_neighbour]
    )

    pca, coords, loadings = fit_pca(X)
    loadings.to_csv(METRICS / "pca_loadings.csv", index=False)
    explained = pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_ratio": pca.explained_variance_ratio_})
    explained.to_csv(METRICS / "pca_explained_variance.csv", index=False)
    for name, labels, title, filename in [
        (best_h_name, hierarchy_labels[best_h_name], "Hierarchical clustering shown in PCA space", "hierarchical_pca.png"),
        (best_d_name, density_labels[best_d_name], "DBSCAN clustering shown in PCA space (noise = -1)", "dbscan_pca.png"),
    ]:
        plot_pca(coords, labels, f"{title}\n{ name }", PLOTS / filename)
        write_profiles(sample, labels, "hierarchical" if name == best_h_name else "dbscan", PROFILES, PLOTS)

    (METRICS / "selection_notes.txt").write_text(
        f"Dataset years: {year_min}-{year_max}; {total_rows} events.\n"
        f"Analysis sample: {len(sample)} (random_state={RANDOM_STATE}; maximum_sample_size={SAMPLE_SIZE}).\n"
        f"Clustering was fit in standardized imputed feature space, not PCA space.\n"
        f"PCA PC1+PC2 explained variance: {pca.explained_variance_ratio_.sum():.4f}.\n"
        f"Selected hierarchical configuration: {best_h_name}. {hierarchical_note}\n"
        f"Hierarchical silhouette (deterministic evaluation sample): {selected_h_row['silhouette_evaluation_sample']:.4f}.\n"
        f"ARI against {ward_neighbour}: {ward_neighbour_ari:.4f}.\n"
        f"Selected DBSCAN configuration: {best_d_name}. {dbscan_note}\n"
        f"DBSCAN non-noise silhouette: {selected_d_row['silhouette_non_noise_evaluation_sample']:.4f}; noise: {selected_d_row['noise_count']} ({selected_d_row['noise_percentage']:.2f}%).\n"
        f"ARI against {dbscan_neighbour}: {dbscan_neighbour_ari:.4f}.\n"
        "Silhouette scores for DBSCAN use non-noise points only and do not score noise assignments.\n"
        "Clusters are exploratory patterns in this sample, not permanent weather categories.\n",
        encoding="utf-8",
    )
    print("\nBest hierarchical:", best_h_name)
    print("Best DBSCAN:", best_d_name, "-", dbscan_note)
    print("PCA variance explained by first two components:", round(explained["explained_variance_ratio"].sum(), 4))
    print("Outputs written under outputs/ (sample size:", len(sample), ")")
    print("\nComparison preview:\n", comparison.sort_values("silhouette", ascending=False, na_position="last").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
