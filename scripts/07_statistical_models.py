"""Statistical modeling: mixed-effects regression for uncertainty analysis.

Fits positional and filtering models, runs sensitivity analyses,
and saves model summaries.
"""

import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.statistical_models import fit_positional_model, fit_filtering_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tables"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load sentence-level data ─────────────────────────────────────
    sentence_path = DATA_DIR / "sentence_level.parquet"
    if not sentence_path.exists():
        logger.error("Sentence-level data not found. Run 05_position_analysis.py first.")
        return

    logger.info("Loading sentence-level data...")
    sentence_df = pl.read_parquet(sentence_path)
    logger.info("Loaded %d sentence records", len(sentence_df))

    # ── Model 1: Positional uncertainty ──────────────────────────────
    logger.info("\n=== Fitting Positional Model ===")
    try:
        pos_result = fit_positional_model(sentence_df)
        logger.info("\n%s", pos_result.summary())

        # Save summary
        with open(RESULTS_DIR / "model_positional_summary.txt", "w") as f:
            f.write(str(pos_result.summary()))
    except Exception as e:
        logger.error("Positional model failed: %s", e)

    # ── Model 2: Confidence filtering ────────────────────────────────
    filtering_path = DATA_DIR / "filtering_metrics.parquet"
    if filtering_path.exists():
        logger.info("\n=== Fitting Filtering Model ===")
        filtering_df = pl.read_parquet(filtering_path)

        # Join metadata if available
        try:
            filt_result = fit_filtering_model(filtering_df)
            logger.info("\n%s", filt_result.summary())

            with open(RESULTS_DIR / "model_filtering_summary.txt", "w") as f:
                f.write(str(filt_result.summary()))
        except Exception as e:
            logger.error("Filtering model failed: %s", e)
    else:
        logger.warning("Filtering data not found. Run 06_confidence_filtering.py first.")

    # ── Sensitivity: different min reasoning lengths ─────────────────
    logger.info("\n=== Sensitivity Analysis: Min Reasoning Length ===")
    for min_sents in [3, 5, 10, 20]:
        filtered = sentence_df.filter(pl.col("total_sentences") >= min_sents)
        n_interactions = filtered["interaction_id"].n_unique()
        overall_rate = filtered["has_uncertainty"].mean()
        logger.info("  min_sentences=%2d: %6d interactions, %8d sentences, rate=%.4f",
                     min_sents, n_interactions, len(filtered), overall_rate)

    logger.info("\nStatistical modeling complete")


if __name__ == "__main__":
    main()
