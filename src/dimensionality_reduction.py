"""PCA projection for visualisation, not for fitting the clusters."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path


def fit_pca(X):
    model = PCA(n_components=2, random_state=42)
    coords = model.fit_transform(X)
    loadings = pd.DataFrame(model.components_.T, columns=["PC1_loading", "PC2_loading"])
    loadings.insert(0, "feature", X.columns)
    return model, coords, loadings


def plot_pca(coords, labels, title, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    classes, encoded = np.unique(labels, return_inverse=True)
    cmap = plt.get_cmap("tab20", max(1, len(classes)))
    norm = BoundaryNorm(np.arange(-.5, len(classes) + .5), cmap.N)
    points = ax.scatter(coords[:, 0], coords[:, 1], c=encoded, cmap=cmap, norm=norm, s=10, alpha=.65)
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    colorbar = fig.colorbar(points, ax=ax, ticks=range(len(classes)))
    colorbar.ax.set_yticklabels([str(value) for value in classes])
    colorbar.set_label("Cluster label (DBSCAN noise = -1)")
    fig.tight_layout(); fig.savefig(Path(path), dpi=160); plt.close(fig)
