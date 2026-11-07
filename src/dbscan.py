"""DBSCAN sensitivity experiments, k-distance diagnostic, and density evaluation.

Explores density-based clustering behavior across eps and min_samples:
- k-nearest neighbor distance curve to identify the density threshold (elbow)
- Parameter sensitivity grid (eps in [0.8 .. 2.0], min_samples in [5, 8, 12, 15])
- Noise detection (% unclustered events) and core vs border point identification
- Non-noise cluster separation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

EPS_VALUES = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
MIN_SAMPLES_VALUES = (5, 8, 12, 15)


def run_dbscan(X, out_dir, metrics_dir, silhouette_limit=2500):
    """Run DBSCAN parameter grid and save diagnostics."""
    out_dir, metrics_dir = Path(out_dir), Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. K-nearest neighbor distance curve
    k = 8
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth_distances = np.sort(distances[:, -1])

    # Find approximate knee (maximum perpendicular distance from chord line)
    n_pts = len(kth_distances)
    x_coords = np.arange(n_pts)
    line_vec = np.array([n_pts - 1, kth_distances[-1] - kth_distances[0]])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    pt_vecs = np.vstack([x_coords, kth_distances - kth_distances[0]]).T
    proj = np.dot(pt_vecs, line_vec_norm)
    proj_vecs = np.outer(proj, line_vec_norm)
    dist_to_line = np.linalg.norm(pt_vecs - proj_vecs, axis=1)
    knee_idx = int(np.argmax(dist_to_line))
    knee_eps = round(float(kth_distances[knee_idx]), 2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(kth_distances, color="#1f77b4", lw=2, label=f"{k}-NN Distance Curve")
    ax.axhline(knee_eps, color="#d62728", linestyle="--", lw=1.5, label=f"Detected Knee (eps ~ {knee_eps})")
    ax.scatter([knee_idx], [knee_eps], color="#d62728", s=60, zorder=5)
    ax.set_title(f"{k}-Nearest Neighbor Distance Curve (DBSCAN Knee Diagnostic)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Observations Sorted by k-NN Distance", fontsize=10)
    ax.set_ylabel("Distance to k-th Nearest Neighbor", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(str(out_dir / "dbscan_k_distance.png"), dpi=160)
    plt.close(fig)

    eval_idx = np.linspace(0, len(X) - 1, min(len(X), silhouette_limit), dtype=int)
    rows = []
    label_sets = {}

    for min_samples in MIN_SAMPLES_VALUES:
        for eps in EPS_VALUES:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1).fit(X)
            labels = db.labels_

            name = f"dbscan_eps{eps:g}_min{min_samples}"
            label_sets[name] = labels

            noise_mask = (labels == -1)
            noise_count = int(noise_mask.sum())
            noise_pct = float(noise_mask.mean() * 100.0)

            core_mask = np.zeros_like(labels, dtype=bool)
            core_mask[db.core_sample_indices_] = True
            core_count = int(core_mask.sum())
            border_count = int((~noise_mask & ~core_mask).sum())

            non_noise_labels = labels[~noise_mask]
            unique_clusters = np.unique(non_noise_labels)
            num_clusters = len(unique_clusters)

            sizes = pd.Series(non_noise_labels).value_counts() if num_clusters > 0 else pd.Series(dtype=int)

            # Evaluate non-noise sample metrics
            eval_labels = labels[eval_idx]
            eval_non_noise = eval_labels != -1
            eval_unique = np.unique(eval_labels[eval_non_noise])

            if len(eval_unique) >= 2 and eval_non_noise.sum() > len(eval_unique):
                X_eval_clean = X[eval_idx][eval_non_noise]
                lbls_clean = eval_labels[eval_non_noise]
                sil = float(silhouette_score(X_eval_clean, lbls_clean))
                dbi = float(davies_bouldin_score(X_eval_clean, lbls_clean))
                chi = float(calinski_harabasz_score(X_eval_clean, lbls_clean))
            else:
                sil, dbi, chi = np.nan, np.nan, np.nan

            rows.append({
                "configuration": name,
                "eps": eps,
                "min_samples": min_samples,
                "clusters_excluding_noise": num_clusters,
                "noise_count": noise_count,
                "noise_percentage": round(noise_pct, 2),
                "core_points": core_count,
                "border_points": border_count,
                "largest_cluster": int(sizes.max()) if len(sizes) else 0,
                "smallest_cluster": int(sizes.min()) if len(sizes) else 0,
                "silhouette_non_noise_evaluation_sample": round(sil, 4) if pd.notna(sil) else np.nan,
                "davies_bouldin_index": round(dbi, 4) if pd.notna(dbi) else np.nan,
                "calinski_harabasz_index": round(chi, 2) if pd.notna(chi) else np.nan,
            })

    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / "dbscan_results.csv", index=False)

    # Plot Parameter Sensitivity: Noise % and Number of Clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for min_samples, group in results.groupby("min_samples"):
        ax1.plot(group["eps"], group["noise_percentage"], marker="o", lw=1.8, label=f"min_samples={min_samples}")
        ax2.plot(group["eps"], group["clusters_excluding_noise"], marker="s", lw=1.8, label=f"min_samples={min_samples}")

    ax1.set_title("Noise Percentage Sensitivity", fontsize=11, fontweight="bold")
    ax1.set_xlabel("eps", fontsize=10)
    ax1.set_ylabel("Noise Observations (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.set_title("Discovered Clusters Sensitivity (Excl. Noise)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("eps", fontsize=10)
    ax2.set_ylabel("Number of Clusters", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.suptitle("DBSCAN Parameter Sensitivity Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_dir / "dbscan_parameter_analysis.png"), dpi=160)
    plt.close(fig)

    return results, label_sets
