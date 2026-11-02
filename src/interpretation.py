"""Cluster profiles in original units and event-type composition."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


PROFILE_COLUMNS = ["TOTAL_INJURIES", "TOTAL_DEATHS", "PROPERTY_DAMAGE_USD", "CROP_DAMAGE_USD", "DURATION_HOURS"]


def write_profiles(df, labels, name, output_dir, plot_dir):
    output_dir, plot_dir = Path(output_dir), Path(plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True); plot_dir.mkdir(parents=True, exist_ok=True)
    work = df.copy(); work["cluster"] = labels
    work["EVENT_TYPE"] = work["EVENT_TYPE"].fillna("Unknown")
    numeric = work.groupby("cluster")[PROFILE_COLUMNS].agg(["count", "mean", "median"])
    numeric.columns = [f"{a}_{b}" for a, b in numeric.columns]
    numeric.insert(0, "event_count", work.groupby("cluster").size())
    numeric.to_csv(output_dir / f"{name}_profiles.csv")
    composition = pd.crosstab(work["cluster"], work["EVENT_TYPE"], normalize="index")
    composition.to_csv(output_dir / f"{name}_event_type_distribution.csv")

    medians = work.groupby("cluster")[PROFILE_COLUMNS].median()
    scaled = (medians - medians.mean()) / medians.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(scaled, cmap="vlag", center=0, annot=True, fmt=".1f", ax=ax)
    ax.set(title=f"{name.title()} median profiles (standardised for display)", xlabel="Feature", ylabel="Cluster")
    fig.tight_layout(); fig.savefig(plot_dir / f"{name}_cluster_profile_heatmap.png", dpi=160); plt.close(fig)
    return numeric, composition
