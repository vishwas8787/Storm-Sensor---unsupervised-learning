"""Evaluation metrics and stability analysis across clustering settings."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def stability_table(label_sets, metrics_dir):
    """Compute pairwise Adjusted Rand Index matrix across all tested configurations."""
    names = list(label_sets)
    matrix = np.zeros((len(names), len(names)))
    rows = []

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            score = adjusted_rand_score(label_sets[a], label_sets[b])
            matrix[i, j] = score
            if j > i:
                rows.append({
                    "configuration_a": a,
                    "configuration_b": b,
                    "adjusted_rand_index": round(score, 4),
                })

    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(metrics_dir) / "stability_results.csv", index=False)
    return matrix, names


def plot_silhouettes(hierarchical, dbscan, plot_dir):
    """Generate a horizontal bar plot comparing silhouette scores."""
    h = hierarchical[["configuration", "silhouette_evaluation_sample"]].rename(
        columns={"silhouette_evaluation_sample": "silhouette"}
    )
    h["method"] = "Hierarchical"

    d = dbscan[["configuration", "silhouette_non_noise_evaluation_sample"]].rename(
        columns={"silhouette_non_noise_evaluation_sample": "silhouette"}
    )
    d["method"] = "DBSCAN (Non-noise)"

    data = pd.concat([h, d], ignore_index=True).dropna().sort_values("silhouette", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(data))))
    if len(data):
        colors = ["#2ca02c" if m == "Hierarchical" else "#1f77b4" for m in data["method"]]
        bars = ax.barh(data["configuration"], data["silhouette"], color=colors, alpha=0.85)
        ax.set_xlim([0, 1.0])
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No configuration had a valid silhouette score", ha="center", va="center")

    ax.set_title("Silhouette Scores across Tested Configurations\n(DBSCAN evaluates non-noise observations)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Silhouette Coefficient", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / "silhouette_plot.png"), dpi=160)
    plt.close(fig)


def final_comparison(hierarchical, dbscan, metrics_dir):
    """Create a consolidated multi-metric comparison table."""
    h = hierarchical[[
        "configuration", "actual_clusters", "silhouette_evaluation_sample",
        "davies_bouldin_index", "calinski_harabasz_index"
    ]].copy()
    h["method"] = "Hierarchical"
    h["noise_percentage"] = 0.0
    h.rename(columns={
        "actual_clusters": "clusters",
        "silhouette_evaluation_sample": "silhouette",
    }, inplace=True)

    d = dbscan[[
        "configuration", "clusters_excluding_noise", "noise_percentage",
        "silhouette_non_noise_evaluation_sample", "davies_bouldin_index", "calinski_harabasz_index"
    ]].copy()
    d["method"] = "DBSCAN"
    d.rename(columns={
        "clusters_excluding_noise": "clusters",
        "silhouette_non_noise_evaluation_sample": "silhouette",
    }, inplace=True)

    result = pd.concat([h, d], ignore_index=True)[[
        "configuration", "method", "clusters", "noise_percentage",
        "silhouette", "davies_bouldin_index", "calinski_harabasz_index"
    ]]
    result.to_csv(Path(metrics_dir) / "final_comparison.csv", index=False)
    return result
