"""Save compact, presentation-ready analysis charts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_stability(matrix, names, path):
    fig, ax = plt.subplots(figsize=(max(9, .55 * len(names)), max(8, .5 * len(names))))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_title("Adjusted Rand Index between tested configurations")
    fig.colorbar(image, ax=ax, label="ARI (1 = identical assignments)")
    fig.tight_layout(); fig.savefig(Path(path), dpi=160); plt.close(fig)
