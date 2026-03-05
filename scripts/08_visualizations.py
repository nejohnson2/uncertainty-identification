"""Generate all publication-ready figures from saved results.

All figures read from saved data — no recomputation. Designed to be
run independently of the analysis pipeline.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import polars as pl
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.uncertainty_lexicon import get_category_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tables"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"

# Publication style
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (8, 5),
})

# Accessible color palette
PALETTE = sns.color_palette("colorblind")


def fig1_uncertainty_density_curve():
    """Main result: uncertainty rate across reasoning position."""
    path = RESULTS_DIR / "uncertainty_by_decile.csv"
    if not path.exists():
        logger.warning("Skipping fig1: %s not found", path)
        return

    df = pl.read_csv(path)
    fig, ax = plt.subplots()

    ax.plot(df["position_decile"], df["uncertainty_rate"],
            marker="o", linewidth=2, color=PALETTE[0], markersize=8)
    ax.fill_between(df["position_decile"], df["uncertainty_rate"],
                     alpha=0.15, color=PALETTE[0])

    ax.set_xlabel("Position Decile in Reasoning Chain")
    ax.set_ylabel("Uncertainty Rate (proportion of sentences)")
    ax.set_title("Uncertainty Expression Across Reasoning Position")
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)

    fig.savefig(FIGURES_DIR / "fig1_uncertainty_density.png")
    fig.savefig(FIGURES_DIR / "fig1_uncertainty_density.pdf")
    plt.close(fig)
    logger.info("Saved fig1")


def fig2_category_by_position():
    """Category breakdown by position decile."""
    path = RESULTS_DIR / "uncertainty_by_decile.csv"
    if not path.exists():
        logger.warning("Skipping fig2: %s not found", path)
        return

    df = pl.read_csv(path)
    cat_names = get_category_names()
    cat_cols = [f"cat_{c}_rate" for c in cat_names if f"cat_{c}_rate" in df.columns]

    if not cat_cols:
        logger.warning("Skipping fig2: no category rate columns found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = df["position_decile"].to_numpy()

    for i, col in enumerate(cat_cols):
        label = col.replace("cat_", "").replace("_rate", "").replace("_", " ").title()
        ax.plot(x, df[col].to_numpy(), marker="o", linewidth=1.5,
                color=PALETTE[i % len(PALETTE)], label=label, markersize=5)

    ax.set_xlabel("Position Decile in Reasoning Chain")
    ax.set_ylabel("Category Rate")
    ax.set_title("Uncertainty Categories Across Reasoning Position")
    ax.set_xticks(range(1, 11))
    ax.legend(loc="best", fontsize=9)

    fig.savefig(FIGURES_DIR / "fig2_category_by_position.png")
    fig.savefig(FIGURES_DIR / "fig2_category_by_position.pdf")
    plt.close(fig)
    logger.info("Saved fig2")


def fig3_confidence_filtering():
    """Paired bars: reasoning vs response uncertainty rates."""
    path = RESULTS_DIR / "filtering_paired_tests.csv"
    if not path.exists():
        logger.warning("Skipping fig3: %s not found", path)
        return

    df = pl.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))

    tests = df["test"].to_list()
    r_means = df["reasoning_mean"].to_numpy()
    resp_means = df["response_mean"].to_numpy()
    p_values = df["p_value"].to_numpy()

    x = np.arange(len(tests))
    width = 0.35

    bars1 = ax.bar(x - width / 2, r_means, width, label="Reasoning",
                    color=PALETTE[0], alpha=0.85)
    bars2 = ax.bar(x + width / 2, resp_means, width, label="Response",
                    color=PALETTE[1], alpha=0.85)

    # Add significance stars
    for i, p in enumerate(p_values):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = "ns"
        max_h = max(r_means[i], resp_means[i])
        ax.text(x[i], max_h + 0.005, star, ha="center", fontsize=10)

    labels = [t.replace("_", " ").title() for t in tests]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Uncertainty Rate")
    ax.set_title("Confidence Filtering: Reasoning vs Response Uncertainty")
    ax.legend()

    fig.savefig(FIGURES_DIR / "fig3_confidence_filtering.png")
    fig.savefig(FIGURES_DIR / "fig3_confidence_filtering.pdf")
    plt.close(fig)
    logger.info("Saved fig3")


def fig4_model_comparison():
    """Overlaid density curves for model families."""
    path = RESULTS_DIR / "uncertainty_by_decile_model.csv"
    if not path.exists():
        logger.warning("Skipping fig4: %s not found", path)
        return

    df = pl.read_csv(path)
    fig, ax = plt.subplots()

    families = df["model_family"].unique().to_list()
    for i, family in enumerate(sorted(families)):
        subset = df.filter(pl.col("model_family") == family)
        ax.plot(subset["position_decile"], subset["uncertainty_rate"],
                marker="o", linewidth=2, color=PALETTE[i % len(PALETTE)],
                label=family.title(), markersize=6)

    ax.set_xlabel("Position Decile in Reasoning Chain")
    ax.set_ylabel("Uncertainty Rate")
    ax.set_title("Uncertainty by Position: Model Family Comparison")
    ax.set_xticks(range(1, 11))
    ax.legend()

    fig.savefig(FIGURES_DIR / "fig4_model_comparison.png")
    fig.savefig(FIGURES_DIR / "fig4_model_comparison.pdf")
    plt.close(fig)
    logger.info("Saved fig4")


def fig5_filtering_distribution():
    """Histogram of per-interaction filtering ratios."""
    path = RESULTS_DIR / "filtering_metrics_summary.csv"
    if not path.exists():
        logger.warning("Skipping fig5: %s not found", path)
        return

    df = pl.read_csv(path)
    if "filtering_ratio" not in df.columns:
        logger.warning("Skipping fig5: filtering_ratio column not found")
        return

    fig, ax = plt.subplots()
    ratios = df["filtering_ratio"].to_numpy()

    ax.hist(ratios, bins=50, color=PALETTE[0], alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(np.median(ratios), color=PALETTE[2], linestyle="-", linewidth=1.5,
               label=f"Median = {np.median(ratios):.2f}")

    ax.set_xlabel("Filtering Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Confidence Filtering Ratios")
    ax.legend()

    fig.savefig(FIGURES_DIR / "fig5_filtering_distribution.png")
    fig.savefig(FIGURES_DIR / "fig5_filtering_distribution.pdf")
    plt.close(fig)
    logger.info("Saved fig5")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig1_uncertainty_density_curve()
    fig2_category_by_position()
    fig3_confidence_filtering()
    fig4_model_comparison()
    fig5_filtering_distribution()

    logger.info("\nAll visualizations complete. Saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
