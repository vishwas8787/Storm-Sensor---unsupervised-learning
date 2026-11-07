"""Hierarchical clustering experiments on the reproducible analysis sample.

Evaluates Single, Complete, Average, and Ward linkage algorithms across multiple
cluster counts (k=3, 4, 5, 6) using:
- Cophenetic Correlation Coefficient (fidelity to original metric space)
- Silhouette Score (intra-cluster cohesion vs nearest-cluster separation)
- Davies-Bouldin Index (cluster separation ratio; lower is better)
- Calinski-Harabasz Index (variance ratio criterion; higher is better)
- Cluster size distribution (screening out trivial singleton branches)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

LINKAGES = ["ward", "complete", "average", "single"]


def run_hierarchical(X, sample_df, out_dir, metrics_dir, cluster_counts=(3, 4, 5, 6), silhouette_limit=2500):
    """Run hierarchical clustering on standardized feature matrix X."""
    out_dir, metrics_dir = Path(out_dir), Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    n = len(X)
    eval_idx = np.linspace(0, n - 1, min(n, silhouette_limit), dtype=int)
    pdist_eval = pdist(X[eval_idx])

    rows = []
    label_sets = {}
    cophenetic_rows = []

    for method in LINKAGES:
        tree = linkage(X, method=method, metric="euclidean")

        # Compute Cophenetic Correlation Coefficient on the evaluation subset
        tree_eval = linkage(X[eval_idx], method=method, metric="euclidean")
        cpcc, _ = cophenet(tree_eval, pdist_eval)
        cophenetic_rows.append({"linkage": method, "cophenetic_correlation": float(cpcc)})

        # Generate truncated dendrogram
        fig, ax = plt.subplots(figsize=(12, 5.5))
        dend_data = dendrogram(
            tree,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90,
            leaf_font_size=9,
            color_threshold=0.7 * max(tree[:, 2]),
            ax=ax,
        )
        ax.set_title(
            f"{method.title()} Linkage Dendrogram (Truncated p=30, n={n})\n"
            f"Cophenetic Correlation: {cpcc:.4f}",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_xlabel("Clusters or Merged Observations", fontsize=10)
        ax.set_ylabel("Euclidean Distance", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        save_file = out_dir / f"dendrogram_{method}.png"
        fig.savefig(str(save_file), dpi=160)
        plt.close(fig)

        for k in cluster_counts:
            labels = fcluster(tree, t=k, criterion="maxclust")
            distinct = np.unique(labels)
            evaluation_labels = labels[eval_idx]
            evaluation_clusters = np.unique(evaluation_labels)

            if 1 < len(evaluation_clusters) < len(eval_idx):
                sil = float(silhouette_score(X[eval_idx], evaluation_labels))
                db = float(davies_bouldin_score(X[eval_idx], evaluation_labels))
                ch = float(calinski_harabasz_score(X[eval_idx], evaluation_labels))
            else:
                sil, db, ch = np.nan, np.nan, np.nan

            key = f"{method}_k{k}"
            label_sets[key] = labels
            sizes = pd.Series(labels).value_counts()

            rows.append({
                "configuration": key,
                "method": method,
                "requested_clusters": k,
                "actual_clusters": len(distinct),
                "cophenetic_correlation": round(float(cpcc), 4),
                "silhouette_evaluation_sample": round(sil, 4) if pd.notna(sil) else np.nan,
                "davies_bouldin_index": round(db, 4) if pd.notna(db) else np.nan,
                "calinski_harabasz_index": round(ch, 2) if pd.notna(ch) else np.nan,
                "smallest_cluster": int(sizes.min()),
                "largest_cluster": int(sizes.max()),
                "cluster_size_ratio": round(int(sizes.max()) / max(1, int(sizes.min())), 2),
            })

    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / "hierarchical_results.csv", index=False)

    cophenetic_df = pd.DataFrame(cophenetic_rows).sort_values("cophenetic_correlation", ascending=False)
    cophenetic_df.to_csv(metrics_dir / "cophenetic_correlations.csv", index=False)

    return results, label_sets
