"""Exploratory summaries and plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def run_eda(df, plot_dir, metrics_dir, correlation_df=None):
    plot_dir, metrics_dir = Path(plot_dir), Path(metrics_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_report = metrics_dir / "missing_values_report.csv"
    if preprocessing_report.exists():
        report = pd.read_csv(preprocessing_report, index_col=0)
        missing = report["Missing_Percentage"].sort_values(ascending=False)
    else:
        missing = df.isna().mean().mul(100).sort_values(ascending=False)
    missing.to_frame("Missing_Percentage").to_csv(metrics_dir / "missingness_by_feature.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    missing.head(20).sort_values().plot.barh(ax=ax, color="#3973ac")
    ax.set(title="Features with the most missing values", xlabel="Missing (%)", ylabel="")
    fig.tight_layout(); fig.savefig(plot_dir / "missing_values.png", dpi=160); plt.close(fig)

    counts = df["EVENT_TYPE"].fillna("Unknown").value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7)); counts.plot.barh(ax=ax, color="#4c956c")
    ax.set(title="Most frequently reported event types", xlabel="Records", ylabel="")
    fig.tight_layout(); fig.savefig(plot_dir / "event_distribution.png", dpi=160); plt.close(fig)

    corr_source = df if correlation_df is None else correlation_df
    numeric = ["TOTAL_INJURIES", "TOTAL_DEATHS", "PROPERTY_DAMAGE_USD", "CROP_DAMAGE_USD", "DURATION_HOURS"]
    corr = corr_source[numeric].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax)
    ax.set_title("Spearman correlations among impact and duration features")
    fig.tight_layout(); fig.savefig(plot_dir / "correlation_heatmap.png", dpi=160); plt.close(fig)
    return counts
