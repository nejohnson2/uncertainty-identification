"""Confidence filtering analysis: compare uncertainty in reasoning vs response.

Tests whether uncertainty present in reasoning traces gets suppressed
in the final response.
"""

import json
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import load_full
from src.preprocessing import preprocess
from src.confidence_filtering import compute_filtering_metrics, run_paired_tests
from src.uncertainty_lexicon import get_category_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tables"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and preprocess ──────────────────────────────────────────
    logger.info("Loading dataset...")
    df = load_full().collect()
    df = preprocess(df)

    # Filter to rows with both reasoning and response
    df = df.filter(
        pl.col("response").is_not_null()
        & (pl.col("response").str.len_chars() > 0)
    )
    logger.info("Rows with both reasoning and response: %d", len(df))

    # ── Compute filtering metrics ────────────────────────────────────
    filtering_df = compute_filtering_metrics(df, use_spacy=True)
    logger.info("Filtering metrics computed for %d interactions", len(filtering_df))

    # ── Paired statistical tests ─────────────────────────────────────
    test_results = run_paired_tests(filtering_df)

    logger.info("\n=== Paired Test Results ===")
    for test_name, result in test_results.items():
        logger.info("  %s:", test_name)
        logger.info("    Reasoning mean: %.4f", result.get("reasoning_mean", 0))
        logger.info("    Response mean:  %.4f", result.get("response_mean", 0))
        logger.info("    Wilcoxon W: %.1f, p=%.2e", result["statistic"], result["p_value"])

    # ── Filtering ratio statistics ───────────────────────────────────
    fr = filtering_df["filtering_ratio"]
    logger.info("\n=== Filtering Ratio ===")
    logger.info("  Mean: %.3f", fr.mean())
    logger.info("  Median: %.3f", fr.median())
    logger.info("  Std: %.3f", fr.std())
    logger.info("  %% positive (uncertainty reduced): %.1f%%",
                filtering_df.filter(pl.col("filtering_ratio") > 0).height / len(filtering_df) * 100)
    logger.info("  %% negative (uncertainty increased): %.1f%%",
                filtering_df.filter(pl.col("filtering_ratio") < 0).height / len(filtering_df) * 100)

    # ── Save results ─────────────────────────────────────────────────
    filtering_df.write_parquet(DATA_DIR / "filtering_metrics.parquet")
    filtering_df.write_csv(RESULTS_DIR / "filtering_metrics_summary.csv")

    # Save test results
    test_records = []
    for test_name, result in test_results.items():
        test_records.append({
            "test": test_name,
            "reasoning_mean": result.get("reasoning_mean", 0),
            "response_mean": result.get("response_mean", 0),
            "statistic": result["statistic"],
            "p_value": result["p_value"],
        })
    pl.DataFrame(test_records).write_csv(RESULTS_DIR / "filtering_paired_tests.csv")

    logger.info("\nConfidence filtering analysis complete")


if __name__ == "__main__":
    main()
