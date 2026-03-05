"""Exploratory data analysis for the real-slop dataset.

Outputs descriptive statistics and determines effective sample size
(rows with reasoning traces).
"""

import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import load_full, PARQUET_PATH
from src.preprocessing import preprocess, extract_model_family, extract_model_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tables"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", PARQUET_PATH)
    lf = load_full()
    df = lf.collect()
    logger.info("Total rows: %d", len(df))

    # ── Basic statistics ─────────────────────────────────────────────
    stats = {}
    stats["total_rows"] = len(df)

    # Null counts per column
    for col in df.columns:
        null_count = df[col].null_count()
        non_null = len(df) - null_count
        stats[f"{col}_non_null"] = non_null
        stats[f"{col}_null_pct"] = round(null_count / len(df) * 100, 2)
        logger.info("Column %-15s: %d non-null (%.1f%% null)", col, non_null, null_count / len(df) * 100)

    # ── Reasoning trace analysis ──────────────────────────────────────
    has_reasoning = df.filter(
        pl.col("reasoning").is_not_null()
        & (pl.col("reasoning").str.len_chars() > 0)
    )
    stats["rows_with_reasoning"] = len(has_reasoning)
    stats["reasoning_pct"] = round(len(has_reasoning) / len(df) * 100, 2)
    logger.info("\n=== CRITICAL: Rows with reasoning traces: %d (%.1f%%) ===",
                len(has_reasoning), len(has_reasoning) / len(df) * 100)

    if len(has_reasoning) > 0:
        # Reasoning length distribution
        reasoning_lens = has_reasoning.select(
            pl.col("reasoning").str.len_chars().alias("len")
        )["len"]

        for q, label in [(0.25, "p25"), (0.5, "median"), (0.75, "p75"), (0.95, "p95")]:
            val = reasoning_lens.quantile(q)
            stats[f"reasoning_len_{label}"] = val
            logger.info("Reasoning length %s: %d chars", label, val)

        stats["reasoning_len_mean"] = round(reasoning_lens.mean(), 1)
        stats["reasoning_len_min"] = reasoning_lens.min()
        stats["reasoning_len_max"] = reasoning_lens.max()

    # ── Model distribution ────────────────────────────────────────────
    model_counts = (
        df.group_by("model")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    logger.info("\n=== Model distribution (top 20) ===")
    for row in model_counts.head(20).iter_rows(named=True):
        logger.info("  %-40s %d", row["model"], row["count"])

    stats["n_unique_models"] = model_counts.height

    # Model family distribution
    df_with_family = df.with_columns(
        extract_model_family(pl.col("model")).alias("model_family"),
        extract_model_provider(pl.col("model")).alias("model_provider"),
    )
    family_counts = (
        df_with_family.group_by("model_family")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    logger.info("\n=== Model family distribution ===")
    for row in family_counts.iter_rows(named=True):
        logger.info("  %-10s %d", row["model_family"], row["count"])

    # Model family x reasoning availability
    family_reasoning = (
        df_with_family
        .with_columns(
            (pl.col("reasoning").is_not_null() & (pl.col("reasoning").str.len_chars() > 0))
            .alias("has_reasoning")
        )
        .group_by("model_family")
        .agg([
            pl.len().alias("total"),
            pl.col("has_reasoning").sum().alias("with_reasoning"),
        ])
        .with_columns(
            (pl.col("with_reasoning") / pl.col("total") * 100).round(1).alias("pct_reasoning")
        )
        .sort("total", descending=True)
    )
    logger.info("\n=== Reasoning availability by model family ===")
    for row in family_reasoning.iter_rows(named=True):
        logger.info("  %-10s %d/%d (%.1f%%)",
                     row["model_family"], row["with_reasoning"], row["total"], row["pct_reasoning"])

    # ── Response analysis ─────────────────────────────────────────────
    has_response = df.filter(
        pl.col("response").is_not_null()
        & (pl.col("response").str.len_chars() > 0)
    )
    stats["rows_with_response"] = len(has_response)

    # Both reasoning AND response (needed for confidence filtering analysis)
    has_both = df.filter(
        pl.col("reasoning").is_not_null()
        & (pl.col("reasoning").str.len_chars() > 0)
        & pl.col("response").is_not_null()
        & (pl.col("response").str.len_chars() > 0)
    )
    stats["rows_with_both"] = len(has_both)
    logger.info("\n=== Rows with BOTH reasoning and response: %d ===", len(has_both))

    # ── Tool usage ────────────────────────────────────────────────────
    has_tools = df.filter(
        pl.col("tools").is_not_null()
        & (pl.col("tools").str.len_chars() > 0)
    )
    stats["rows_with_tools"] = len(has_tools)

    has_tool_calls = df.filter(
        pl.col("tool_calls").is_not_null()
        & (pl.col("tool_calls").str.len_chars() > 0)
    )
    stats["rows_with_tool_calls"] = len(has_tool_calls)

    # ── Save EDA summary ──────────────────────────────────────────────
    summary_df = pl.DataFrame({
        "metric": list(stats.keys()),
        "value": [str(v) for v in stats.values()],
    })
    summary_path = RESULTS_DIR / "eda_summary.csv"
    summary_df.write_csv(summary_path)
    logger.info("\nEDA summary saved to %s", summary_path)

    # Save model distribution
    model_counts.write_csv(RESULTS_DIR / "model_distribution.csv")
    family_reasoning.write_csv(RESULTS_DIR / "family_reasoning_availability.csv")

    logger.info("\n=== EDA Complete ===")


if __name__ == "__main__":
    main()
