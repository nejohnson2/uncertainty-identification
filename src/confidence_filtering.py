"""Confidence filtering: compare uncertainty in reasoning vs. response."""

import logging

import polars as pl
from scipy import stats

from src.uncertainty_lexicon import analyze_text, get_category_names

logger = logging.getLogger(__name__)


def compute_text_uncertainty_rate(text: str, use_spacy: bool = True) -> dict:
    """Compute uncertainty metrics for a single text."""
    results = analyze_text(text, use_spacy=use_spacy)
    if not results:
        return {"n_sentences": 0, "uncertainty_rate": 0.0, "mean_count": 0.0, "categories": {}}

    n = len(results)
    n_uncertain = sum(1 for r in results if r.has_uncertainty)
    total_markers = sum(r.total_markers for r in results)

    # Category-level rates
    cat_counts = {}
    for cat in get_category_names():
        cat_counts[cat] = sum(1 for r in results if r.categories.get(cat, 0) > 0)

    return {
        "n_sentences": n,
        "uncertainty_rate": n_uncertain / n,
        "mean_count": total_markers / n,
        "categories": {cat: count / n for cat, count in cat_counts.items()},
    }


def compute_filtering_metrics(df: pl.DataFrame, use_spacy: bool = True) -> pl.DataFrame:
    """Compute paired reasoning vs response uncertainty metrics.

    For each interaction, computes uncertainty rate in reasoning and response,
    then calculates the filtering ratio.
    """
    from tqdm import tqdm

    records = []
    for i in tqdm(range(len(df)), desc="Computing filtering metrics"):
        row = df.row(i, named=True)
        reasoning = row.get("reasoning", "")
        response = row.get("response", "")

        if not reasoning or not response:
            continue

        r_metrics = compute_text_uncertainty_rate(str(reasoning), use_spacy=use_spacy)
        resp_metrics = compute_text_uncertainty_rate(str(response), use_spacy=use_spacy)

        if r_metrics["n_sentences"] == 0:
            continue

        record = {
            "interaction_id": i,
            "reasoning_n_sentences": r_metrics["n_sentences"],
            "reasoning_uncertainty_rate": r_metrics["uncertainty_rate"],
            "reasoning_mean_count": r_metrics["mean_count"],
            "response_n_sentences": resp_metrics["n_sentences"],
            "response_uncertainty_rate": resp_metrics["uncertainty_rate"],
            "response_mean_count": resp_metrics["mean_count"],
        }

        # Filtering ratio: how much uncertainty is removed
        if r_metrics["uncertainty_rate"] > 0:
            record["filtering_ratio"] = (
                (r_metrics["uncertainty_rate"] - resp_metrics["uncertainty_rate"])
                / r_metrics["uncertainty_rate"]
            )
        else:
            record["filtering_ratio"] = 0.0

        # Category-level survival rates
        for cat in get_category_names():
            r_rate = r_metrics["categories"].get(cat, 0.0)
            resp_rate = resp_metrics["categories"].get(cat, 0.0)
            record[f"reasoning_{cat}_rate"] = r_rate
            record[f"response_{cat}_rate"] = resp_rate

        records.append(record)

    return pl.DataFrame(records)


def run_paired_tests(filtering_df: pl.DataFrame) -> dict:
    """Run Wilcoxon signed-rank tests comparing reasoning vs response uncertainty."""
    results = {}

    # Overall uncertainty rate
    reasoning_rates = filtering_df["reasoning_uncertainty_rate"].to_numpy()
    response_rates = filtering_df["response_uncertainty_rate"].to_numpy()
    stat, p = stats.wilcoxon(reasoning_rates, response_rates, alternative="greater")
    results["overall"] = {
        "statistic": float(stat),
        "p_value": float(p),
        "reasoning_mean": float(reasoning_rates.mean()),
        "response_mean": float(response_rates.mean()),
        "n": len(reasoning_rates),
    }

    # Per-category tests
    for cat in get_category_names():
        r_col = f"reasoning_{cat}_rate"
        resp_col = f"response_{cat}_rate"
        if r_col in filtering_df.columns and resp_col in filtering_df.columns:
            r_vals = filtering_df[r_col].to_numpy()
            resp_vals = filtering_df[resp_col].to_numpy()
            # Only test if there's variance
            if (r_vals != resp_vals).any():
                stat, p = stats.wilcoxon(r_vals, resp_vals, alternative="greater")
                results[cat] = {
                    "statistic": float(stat),
                    "p_value": float(p),
                    "reasoning_mean": float(r_vals.mean()),
                    "response_mean": float(resp_vals.mean()),
                }

    return results
