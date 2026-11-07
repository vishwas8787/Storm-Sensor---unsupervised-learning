"""PCA projection, scree analysis, and loadings biplot for clustering visualization.

PCA is utilized strictly for dimensionality reduction and visualization of
high-dimensional clusters, NOT for fitting the clusters themselves.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_pca(X, n_components=3):
    """Fit PCA on standardized features and return model, coords, and loadings."""
    model = PCA(n_components=n_components, random_state=42)
    coords = model.fit_transform(X)

    cols = [f"PC{i+1}_loading" for i in range(n_components)]
    loadings = pd.DataFrame(model.components_.T, columns=cols)
    loadings.insert(0, "feature", X.columns)

    return model, coords, loadings


def plot_scree(pca_model, out_path):
    """Generate Scree plot and Cumulative Explained Variance curve."""
    var_ratio = pca_model.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    components = [f"PC{i+1}" for i in range(len(var_ratio))]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color = "#1f77b4"
    ax1.bar(components, var_ratio * 100, color=color, alpha=0.75, label="Individual Variance (%)")
    ax1.set_xlabel("Principal Components", fontsize=10)
    ax1.set_ylabel("Explained Variance (%)", color=color, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#d62728"
    ax2.plot(components, cum_var * 100, color=color, marker="o", lw=2, label="Cumulative Variance (%)")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color=color, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim([0, 105])

    plt.title(
        f"PCA Explained Variance Scree Plot (Total 2-Comp: {cum_var[1]*100:.1f}%)",
        fontsize=12,
        fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


def plot_pca(coords, labels, title, path, loadings=None):
    """Plot 2D PCA scatter of clustered events with cluster color-mapping."""
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    classes, encoded = np.unique(labels, return_inverse=True)

    # Use distinct colormap
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20", max(1, len(classes)))
    norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5), cmap.N)

    # Plot points
    points = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=encoded,
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.65,
        edgecolors="none",
    )

    # If loadings provided, add vector biplot arrows
    if loadings is not None:
        arrow_scale = np.max(np.abs(coords[:, :2])) * 0.75
        for _, row in loadings.iterrows():
            feat = row["feature"]
            x = row["PC1_loading"] * arrow_scale
            y = row["PC2_loading"] * arrow_scale
            ax.arrow(0, 0, x, y, color="black", alpha=0.6, head_width=0.15, length_includes_head=True)
            ax.text(x * 1.1, y * 1.1, feat.replace("LOG_", ""), color="#222", fontsize=8, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=10)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Custom colorbar labels
    colorbar = fig.colorbar(points, ax=ax, ticks=range(len(classes)))
    label_names = [f"Noise (-1)" if v == -1 else f"Cluster {v}" for v in classes]
    colorbar.ax.set_yticklabels(label_names)
    colorbar.set_label("Assigned Cluster")

    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)
