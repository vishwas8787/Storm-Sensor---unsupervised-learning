"""Cluster profiles in original units, event-type distributions, and responsible interpretation.

Translates mathematical cluster partitions into domain-relevant meteorological archetypes,
evaluates impact feature profiles in real units (USD, casualties, hours),
and documents limitations, biases, and responsible AI caveats.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROFILE_COLUMNS = [
    "TOTAL_INJURIES",
    "TOTAL_DEATHS",
    "PROPERTY_DAMAGE_USD",
    "CROP_DAMAGE_USD",
    "DURATION_HOURS",
]


def write_profiles(df, labels, name, output_dir, plot_dir):
    """Generate statistical profiles, event distributions, and heatmaps for assigned clusters."""
    output_dir, plot_dir = Path(output_dir), Path(plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    work["cluster"] = labels
    work["EVENT_TYPE"] = work["EVENT_TYPE"].fillna("Unknown")

    # 1. Physical Impact Metrics (Count, Mean, Median, Std)
    numeric = work.groupby("cluster")[PROFILE_COLUMNS].agg(["count", "mean", "median", "std"])
    numeric.columns = [f"{a}_{b}" for a, b in numeric.columns]
    numeric.insert(0, "event_count", work.groupby("cluster").size())
    numeric.to_csv(output_dir / f"{name}_profiles.csv")

    # 2. Event-Type Composition Proportions
    composition = pd.crosstab(work["cluster"], work["EVENT_TYPE"], normalize="index")
    composition.to_csv(output_dir / f"{name}_event_type_distribution.csv")

    # 3. Standardized Median Heatmap for Intuitive Profile Visualisation
    medians = work.groupby("cluster")[PROFILE_COLUMNS].median()
    # Normalize features between 0 and 1 or standard z-score for visual comparison
    stds = medians.std(ddof=0).replace(0, 1)
    scaled = (medians - medians.mean()) / stds

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.8 * len(medians))))
    sns.heatmap(
        scaled,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Standardized Feature Deviation"},
        ax=ax,
    )
    ax.set_title(
        f"{name.title()} Cluster Profiles (Standardized Medians)\n"
        f"Shows relative feature elevation above/below cluster average",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlabel("Physical Impact Features", fontsize=10)
    ax.set_ylabel("Cluster ID", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(plot_dir / f"{name}_cluster_profile_heatmap.png"), dpi=160)
    if name == "hierarchical":
        fig.savefig(str(plot_dir / "cluster_profile_heatmap.png"), dpi=160)
    plt.close(fig)

    return numeric, composition


def write_responsible_interpretation(metrics_dir):
    """Document known limitations, reporting biases, and responsible AI principles."""
    report = """RESPONSIBLE AI AND INTERPRETATION REPORT: NOAA STORM EVENTS GROUPING
========================================================================

1. NATURE OF DISCOVERED CLUSTERS (EXPLORATORY VS PHYSICAL TAXONOMY)
The groupings identified via Hierarchical and DBSCAN clustering represent
exploratory data partitions based on selected impact attributes, not permanent,
immutable physical laws or definitive meteorological classifications.
Unsupervised clustering groups observations according to distance in the
standardized feature space; boundaries between clusters are continuous transitions
rather than sharp real-world divides.

2. REPORTING AND OBSERVATIONAL BIASES IN NOAA DATA
- Population Density Bias: NOAA Storm Events data relies heavily on National
  Weather Service (NWS) trained spotters, emergency management, law enforcement,
  and public reports. Densely populated urban and suburban areas naturally record
  significantly higher frequencies of hail, localized wind, and minor property
  damage compared to unpopulated rural plains and wilderness areas.
- Reporting Evolution Over Time (2010-2020): Technological improvements (dual-polarization
  radar, smartphone weather spotter apps, automated weather stations) have increased
  the detection of minor events over time, potentially shifting sample densities.
- Unadjusted Monetary Impact: Property and crop damage estimates are reported in nominal
  USD at the time of the event and are not inflation-adjusted across the 11-year observation
  window.

3. METHODOLOGICAL SENSITIVITIES
- Missing Value Imputation: Missing property and crop damage fields were treated as $0.00
  consistent with NWS reporting guidelines (absence of damage reports indicates no
  quantifiable monetary damage was verified). However, unquantified rural or non-structural
  losses may exist.
- Linkage Sensitivity: Single and Complete linkage suffer from chaining and extreme outlier
  isolation, yielding artificially high silhouette scores on near-singleton clusters.
  Ward's linkage was selected because it minimizes total within-cluster variance, producing
  balanced, reproducible clusters.
- DBSCAN Thresholds: DBSCAN evaluates density continuity. Points classified as noise (-1)
  are rare, extreme compound multi-hazard events or isolated anomalies. They should not
  be discarded as "errors", but recognized as non-modal phenomena.

4. ETHICAL AND PUBLIC SAFETY IMPLICATIONS
Clustering models should assist hazard risk assessment and resource allocation planning,
but MUST NOT be used to downplay rare, extreme weather risks simply because an event is
grouped into a historically low-casualty cluster. Every severe storm possesses localized,
life-threatening potential regardless of its mathematical cluster assignment.
"""
    (Path(metrics_dir) / "responsible_interpretation.txt").write_text(report, encoding="utf-8")
