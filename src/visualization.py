"""Visualization helpers for model stability and cluster comparison."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_stability(matrix, names, path):
    """Plot Adjusted Rand Index (ARI) heatmap comparing clustering configurations."""
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(names)), max(8, 0.55 * len(names))))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7.5)
    ax.set_yticklabels(names, fontsize=7.5)

    ax.set_title("Adjusted Rand Index (ARI) Stability Matrix\n(Values closer to 1.0 indicate identical cluster partitions)", fontsize=11, fontweight="bold")

    # Add numeric annotations if small enough
    if len(names) <= 15:
        for i in range(len(names)):
            for j in range(len(names)):
                val = matrix[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=6.5)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Adjusted Rand Index (ARI)", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(Path(path)), dpi=160)
    plt.close(fig)
