"""Exploratory data analysis: summary tables and Figures 1-4."""

import matplotlib
matplotlib.use("Agg")  # save to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FEATURES, FIG_DIR, NUTRIENTS, TARGET

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PALETTE = "crest"


def summary_table(df):
    """Descriptive statistics for every feature."""
    return df[FEATURES].astype("float64").describe().T.round(2)


def plot_class_balance(df):
    """Figure 1: samples per crop."""
    counts = df[TARGET].astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    sns.barplot(x=counts.index, y=counts.values, color="#2a7f62", ax=ax)
    ax.set_xlabel("Crop")
    ax.set_ylabel("Samples")
    ax.set_title("Samples per crop (22 classes)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_class_balance.png", dpi=200)
    plt.close(fig)
    return counts


def plot_distributions(df):
    """Figure 2: histogram and density curve for each feature."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col].astype("float64"), kde=True, ax=ax, color="#3b6ea5", bins=30)
        ax.set_title(col)
        ax.set_xlabel("")
    axes.flat[-1].axis("off")  # 7 features on a 2x4 grid
    fig.suptitle("Feature distributions", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_distributions.png", dpi=200)
    plt.close(fig)


def plot_correlation(df):
    """Figure 3: correlation heat map; returns the strongest pair."""
    corr = df[FEATURES].astype("float64").corr()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature correlation (Pearson)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_correlation.png", dpi=200)
    plt.close(fig)

    mask = ~np.eye(len(corr), dtype=bool)
    stacked = corr.where(mask).abs().stack()
    (a, b) = stacked.idxmax()
    return {"pair": (a, b), "r": round(float(corr.loc[a, b]), 2)}


def crop_resource_profile(df):
    """Median N, P, K and rainfall per crop (the re-ranker's requirement table)."""
    return (df.groupby(df[TARGET].astype(str))[NUTRIENTS + ["rainfall"]]
              .median().astype("float64").round(1))


def plot_resource_profile(profile):
    """Figure 4: each crop's typical inputs, scaled 0-1 per column."""
    scaled = (profile - profile.min()) / (profile.max() - profile.min())
    order = scaled.sum(axis=1).sort_values().index  # lowest needs first
    fig, ax = plt.subplots(figsize=(5.4, 6.4))
    sns.heatmap(scaled.loc[order], annot=profile.loc[order], fmt=".0f",
                cmap="YlOrBr", cbar_kws={"label": "relative need (0-1)"}, ax=ax)
    ax.set_title("Typical input needs per crop\n(numbers = median value)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_resource_profile.png", dpi=200)
    plt.close(fig)
    return list(order)


def run_eda(df):
    """Run every EDA step and return the numbers used in the report."""
    stats = summary_table(df)
    counts = plot_class_balance(df)
    plot_distributions(df)
    strongest = plot_correlation(df)
    profile = crop_resource_profile(df)
    order = plot_resource_profile(profile)
    return {
        "summary": stats,
        "class_counts": counts,
        "strongest_correlation": strongest,
        "profile": profile,
        "cheapest_to_most_expensive": order,
    }
