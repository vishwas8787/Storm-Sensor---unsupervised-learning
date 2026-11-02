"""Comparable metrics and agreement across clustering settings."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from pathlib import Path


def stability_table(label_sets, metrics_dir):
    names = list(label_sets)
    matrix = np.zeros((len(names), len(names)))
    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix[i, j] = adjusted_rand_score(label_sets[a], label_sets[b])
            if j > i:
                rows.append({"configuration_a": a, "configuration_b": b, "adjusted_rand_index": matrix[i, j]})
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(metrics_dir) / "stability_results.csv", index=False)
    return matrix, names


def plot_silhouettes(hierarchical, dbscan, plot_dir):
    h = hierarchical[["configuration", "silhouette_evaluation_sample"]].rename(columns={"silhouette_evaluation_sample": "silhouette"})
    d = dbscan[["configuration", "silhouette_non_noise_evaluation_sample"]].rename(columns={"silhouette_non_noise_evaluation_sample": "silhouette"})
    data = pd.concat([h, d], ignore_index=True).dropna().sort_values("silhouette")
    fig, ax = plt.subplots(figsize=(11, max(5, .32 * len(data))))
    if len(data):
        ax.barh(data["configuration"], data["silhouette"], color="#3973ac")
    else:
        ax.text(.5, .5, "No configuration had a valid silhouette score", ha="center", va="center")
    ax.set(title="Silhouette scores (DBSCAN scores exclude noise)", xlabel="Silhouette", ylabel="")
    fig.tight_layout(); fig.savefig(Path(plot_dir) / "silhouette_plot.png", dpi=160); plt.close(fig)


def final_comparison(hierarchical, dbscan, metrics_dir):
    h = hierarchical[["configuration", "actual_clusters", "silhouette_evaluation_sample"]].copy()
    h["method"] = "Hierarchical"
    h["noise_percentage"] = 0.0
    h.rename(columns={"actual_clusters": "clusters", "silhouette_evaluation_sample": "silhouette"}, inplace=True)
    d = dbscan[["configuration", "clusters_excluding_noise", "noise_percentage", "silhouette_non_noise_evaluation_sample"]].copy()
    d["method"] = "DBSCAN"
    d.rename(columns={"clusters_excluding_noise": "clusters", "silhouette_non_noise_evaluation_sample": "silhouette"}, inplace=True)
    result = pd.concat([h, d], ignore_index=True)[["configuration", "method", "clusters", "noise_percentage", "silhouette"]]
    result.to_csv(Path(metrics_dir) / "final_comparison.csv", index=False)
    return result
