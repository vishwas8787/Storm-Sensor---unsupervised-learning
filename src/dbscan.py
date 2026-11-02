"""DBSCAN sensitivity experiments and k-distance diagnostic."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from pathlib import Path


EPS_VALUES = (0.8, 1.2, 1.6, 2.0)
MIN_SAMPLES_VALUES = (8, 15)


def run_dbscan(X, out_dir, metrics_dir, silhouette_limit=1200):
    out_dir, metrics_dir = Path(out_dir), Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    k = 8
    distances, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    kth = np.sort(distances[:, -1])
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(kth, color="#3973ac")
    ax.set(title=f"{k}-nearest-neighbour distance curve", xlabel="Observations sorted by distance", ylabel="Distance")
    fig.tight_layout(); fig.savefig(out_dir / "dbscan_k_distance.png", dpi=160); plt.close(fig)

    eval_idx = np.linspace(0, len(X) - 1, min(len(X), silhouette_limit), dtype=int)
    rows, label_sets = [], {}
    for min_samples in MIN_SAMPLES_VALUES:
        for eps in EPS_VALUES:
            labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1).fit_predict(X)
            mask = labels[eval_idx] != -1
            kept_labels = labels[eval_idx][mask]
            unique = np.unique(kept_labels)
            score = silhouette_score(X[eval_idx][mask], kept_labels) if 1 < len(unique) < len(kept_labels) else np.nan
            name = f"dbscan_eps{eps:g}_min{min_samples}"
            label_sets[name] = labels
            nonnoise = labels[labels != -1]
            sizes = pd.Series(nonnoise).value_counts()
            rows.append({"configuration": name, "eps": eps, "min_samples": min_samples,
                         "clusters_excluding_noise": len(np.unique(nonnoise)),
                         "noise_count": int((labels == -1).sum()), "noise_percentage": float((labels == -1).mean() * 100),
                         "largest_cluster": int(sizes.max()) if len(sizes) else 0,
                         "smallest_cluster": int(sizes.min()) if len(sizes) else 0,
                         "silhouette_non_noise_evaluation_sample": score,
                         "silhouette_sample_non_noise_count": int(mask.sum())})
    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / "dbscan_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    for min_samples, group in results.groupby("min_samples"):
        ax.plot(group["eps"], group["noise_percentage"], marker="o", label=f"min_samples={min_samples}")
    ax.set(title="DBSCAN noise sensitivity", xlabel="eps", ylabel="Noise observations (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "dbscan_parameter_analysis.png", dpi=160); plt.close(fig)
    return results, label_sets
