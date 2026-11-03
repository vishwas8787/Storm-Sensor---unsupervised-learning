"""Hierarchical clustering experiments on the reproducible analysis sample."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from pathlib import Path


LINKAGES = ["single", "complete", "average", "ward"]


def run_hierarchical(X, sample_df, out_dir, metrics_dir, cluster_counts=(3, 4, 5), silhouette_limit=1200):
    out_dir, metrics_dir = Path(out_dir), Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    n = len(X)
    eval_idx = np.linspace(0, n - 1, min(n, silhouette_limit), dtype=int)
    rows, label_sets = [], {}
    for method in LINKAGES:
        tree = linkage(X, method=method, metric="euclidean", optimal_ordering=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        dendrogram(tree, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, ax=ax)
        ax.set_title(f"{method.title()} linkage dendrogram (truncated; n={n})")
        ax.set_xlabel("Merged observations or groups"); ax.set_ylabel("Euclidean distance")
        fig.tight_layout(); fig.savefig(out_dir / f"dendrogram_{method}.png", dpi=160); plt.close(fig)
        for k in cluster_counts:
            labels = fcluster(tree, t=k, criterion="maxclust")
            distinct = np.unique(labels)
            evaluation_labels = labels[eval_idx]
            evaluation_clusters = np.unique(evaluation_labels)
            score = (
                silhouette_score(X[eval_idx], evaluation_labels)
                if 1 < len(evaluation_clusters) < len(eval_idx)
                else np.nan
            )
            key = f"{method}_k{k}"
            label_sets[key] = labels
            sizes = pd.Series(labels).value_counts()
            rows.append({"configuration": key, "method": method, "requested_clusters": k,
                         "actual_clusters": len(distinct), "silhouette_evaluation_sample": score,
                         "smallest_cluster": int(sizes.min()), "largest_cluster": int(sizes.max())})
    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / "hierarchical_results.csv", index=False)
    return results, label_sets
