"""Build and validate the uncertainty lexicon.

Runs the lexical uncertainty detector on a sample of reasoning traces,
reports detection rates per category, and outputs example matches
for manual review.
"""

import logging
import random
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import load_full
from src.preprocessing import preprocess
from src.uncertainty_lexicon import analyze_text, get_category_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tables"
SAMPLE_SIZE = 500  # Number of reasoning traces to analyze for validation


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and preprocess ──────────────────────────────────────────
    logger.info("Loading dataset...")
    df = load_full().collect()
    df = preprocess(df)

    # Sample for validation
    random.seed(42)
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, seed=42)
    else:
        df_sample = df
    logger.info("Analyzing %d reasoning traces", len(df_sample))

    # ── Run lexicon-based detection ──────────────────────────────────
    category_names = get_category_names()
    all_sentence_results = []
    trace_stats = []

    for i in tqdm(range(len(df_sample)), desc="Analyzing traces"):
        row = df_sample.row(i, named=True)
        reasoning = row.get("reasoning", "")
        if not reasoning:
            continue

        results = analyze_text(str(reasoning), use_spacy=True)
        n_sentences = len(results)
        n_uncertain = sum(1 for r in results if r.has_uncertainty)

        trace_stats.append({
            "n_sentences": n_sentences,
            "n_uncertain_sentences": n_uncertain,
            "uncertainty_rate": n_uncertain / n_sentences if n_sentences > 0 else 0.0,
        })

        for r in results:
            all_sentence_results.append({
                "text": r.text[:200],  # Truncate for storage
                "has_uncertainty": r.has_uncertainty,
                "total_markers": r.total_markers,
                **{f"cat_{cat}": r.categories.get(cat, 0) for cat in category_names},
                "matched_phrases": "; ".join(r.matched_phrases[:5]),
            })

    # ── Report statistics ────────────────────────────────────────────
    trace_df = pl.DataFrame(trace_stats)
    sent_df = pl.DataFrame(all_sentence_results)

    logger.info("\n=== Trace-Level Statistics ===")
    logger.info("Traces analyzed: %d", len(trace_df))
    logger.info("Mean sentences per trace: %.1f", trace_df["n_sentences"].mean())
    logger.info("Mean uncertainty rate per trace: %.3f", trace_df["uncertainty_rate"].mean())
    logger.info("Traces with ANY uncertainty: %d (%.1f%%)",
                trace_df.filter(pl.col("n_uncertain_sentences") > 0).height,
                trace_df.filter(pl.col("n_uncertain_sentences") > 0).height / len(trace_df) * 100)

    logger.info("\n=== Sentence-Level Statistics ===")
    logger.info("Total sentences: %d", len(sent_df))
    logger.info("Sentences with uncertainty: %d (%.1f%%)",
                sent_df.filter(pl.col("has_uncertainty")).height,
                sent_df.filter(pl.col("has_uncertainty")).height / len(sent_df) * 100)

    logger.info("\n=== Category Breakdown ===")
    for cat in category_names:
        col = f"cat_{cat}"
        n_with = sent_df.filter(pl.col(col) > 0).height
        pct = n_with / len(sent_df) * 100
        logger.info("  %-25s %d sentences (%.2f%%)", cat, n_with, pct)

    # ── Save for review ──────────────────────────────────────────────
    # Save category rates
    cat_rates = []
    for cat in category_names:
        col = f"cat_{cat}"
        n_with = sent_df.filter(pl.col(col) > 0).height
        cat_rates.append({
            "category": cat,
            "n_sentences_with": n_with,
            "pct_of_sentences": round(n_with / len(sent_df) * 100, 3),
        })
    pl.DataFrame(cat_rates).write_csv(RESULTS_DIR / "lexicon_category_rates.csv")

    # Save example matches for manual review (top 50 uncertain sentences)
    examples = (
        sent_df
        .filter(pl.col("has_uncertainty"))
        .sort("total_markers", descending=True)
        .head(100)
    )
    examples.write_csv(RESULTS_DIR / "lexicon_example_matches.csv")

    # Save trace-level stats
    trace_df.write_csv(RESULTS_DIR / "lexicon_trace_stats.csv")

    logger.info("\nResults saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
