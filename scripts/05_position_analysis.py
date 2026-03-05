"""Core positional uncertainty analysis.

Processes all reasoning traces, builds sentence-level DataFrame,
and computes positional aggregates. This is the main analysis script.
"""

import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import load_full
from src.preprocessing import preprocess
from src.position_analysis import build_sentence_dataframe, aggregate_by_decile, aggregate_by_decile_and_group

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
    logger.info("Rows for analysis: %d", len(df))

    # ── Build sentence-level DataFrame ───────────────────────────────
    logger.info("Building sentence-level DataFrame...")
    sentence_df = build_sentence_dataframe(df, reasoning_col="reasoning", use_spacy=True)
    logger.info("Total sentence records: %d", len(sentence_df))

    # Add metadata from interaction-level columns
    # Create an interaction metadata lookup
    meta_cols = ["model_family", "nsfw_flag", "reasoning_len_chars"]
    if all(c in df.columns for c in meta_cols):
        meta = df.select([
            pl.arange(0, len(df)).alias("interaction_id"),
            *[pl.col(c) for c in meta_cols],
        ])
        sentence_df = sentence_df.join(meta, on="interaction_id", how="left")

    # Optionally join topic labels if available
    topic_path = DATA_DIR / "prompt_topics.parquet"
    if topic_path.exists():
        logger.info("Joining topic labels...")
        topics = pl.read_parquet(topic_path)
        # Would need alignment logic — simplified here
        logger.info("Topic labels available but alignment requires matching")

    # ── Save sentence-level data ─────────────────────────────────────
    sentence_path = DATA_DIR / "sentence_level.parquet"
    sentence_df.write_parquet(sentence_path)
    logger.info("Sentence-level data saved to %s", sentence_path)

    # ── Aggregate by decile ──────────────────────────────────────────
    decile_agg = aggregate_by_decile(sentence_df)
    decile_agg.write_csv(RESULTS_DIR / "uncertainty_by_decile.csv")
    logger.info("\n=== Uncertainty by Position Decile ===")
    for row in decile_agg.iter_rows(named=True):
        logger.info("  Decile %2d: rate=%.3f, mean_count=%.3f (n=%d)",
                     row["position_decile"],
                     row["uncertainty_rate"],
                     row["mean_uncertainty_count"],
                     row["n_sentences"])

    # ── Aggregate by decile x model family ───────────────────────────
    if "model_family" in sentence_df.columns:
        model_agg = aggregate_by_decile_and_group(sentence_df, "model_family")
        model_agg.write_csv(RESULTS_DIR / "uncertainty_by_decile_model.csv")
        logger.info("Decile x model family aggregates saved")

    # ── Aggregate by decile x NSFW flag ──────────────────────────────
    if "nsfw_flag" in sentence_df.columns:
        nsfw_agg = aggregate_by_decile_and_group(sentence_df, "nsfw_flag")
        nsfw_agg.write_csv(RESULTS_DIR / "uncertainty_by_decile_nsfw.csv")
        logger.info("Decile x NSFW aggregates saved")

    # ── Summary statistics ───────────────────────────────────────────
    overall_rate = sentence_df["has_uncertainty"].mean()
    logger.info("\nOverall uncertainty rate: %.3f", overall_rate)
    logger.info("Total sentences analyzed: %d", len(sentence_df))
    logger.info("Total interactions: %d", sentence_df["interaction_id"].n_unique())

    logger.info("\nPositional analysis complete")


if __name__ == "__main__":
    main()
