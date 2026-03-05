"""LLM-based uncertainty classification on a stratified sample.

Samples sentences from reasoning traces (stratified by position decile),
classifies them using a local Ollama model, and compares with lexical labels.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.uncertainty_lexicon import analyze_text, get_category_names
from src.uncertainty_classifier import classify_batch, compute_agreement, DEFAULT_MODEL
from src.data_loading import load_full
from src.preprocessing import preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tables"
SAMPLE_PER_DECILE = 250  # 250 x 10 = 2500 total sentences


def main():
    parser = argparse.ArgumentParser(description="LLM-based uncertainty classification")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and preprocess ──────────────────────────────────────────
    logger.info("Loading dataset...")
    df = load_full().collect()
    df = preprocess(df)

    # Sample a subset of traces to extract sentences from
    df_sample = df.sample(n=min(2000, len(df)), seed=42)

    # ── Extract sentences with position info ─────────────────────────
    logger.info("Extracting sentences from reasoning traces...")
    sentences = []
    for i in tqdm(range(len(df_sample)), desc="Extracting sentences"):
        row = df_sample.row(i, named=True)
        reasoning = row.get("reasoning", "")
        if not reasoning:
            continue

        results = analyze_text(str(reasoning), use_spacy=True)
        total = len(results)
        for idx, r in enumerate(results):
            norm_pos = idx / max(total - 1, 1)
            decile = min(int(norm_pos * 10) + 1, 10)
            sentences.append({
                "sentence": r.text,
                "position_decile": decile,
                "lexical_has_uncertainty": r.has_uncertainty,
                "lexical_categories": r.categories,
            })

    logger.info("Total sentences extracted: %d", len(sentences))

    # ── Stratified sample by position decile ─────────────────────────
    random.seed(42)
    by_decile = {}
    for s in sentences:
        d = s["position_decile"]
        by_decile.setdefault(d, []).append(s)

    sampled = []
    for d in sorted(by_decile.keys()):
        pool = by_decile[d]
        n = min(SAMPLE_PER_DECILE, len(pool))
        sampled.extend(random.sample(pool, n))
        logger.info("Decile %d: sampled %d / %d", d, n, len(pool))

    logger.info("Total stratified sample: %d sentences", len(sampled))

    # ── Classify with Ollama ─────────────────────────────────────────
    logger.info("Using model: %s", args.model)
    sentence_texts = [s["sentence"] for s in sampled]
    llm_results = classify_batch(sentence_texts, model=args.model)

    # ── Compare lexical vs LLM labels ────────────────────────────────
    lexical_labels = [s["lexical_has_uncertainty"] for s in sampled]
    llm_labels = [r.has_uncertainty for r in llm_results]

    agreement = compute_agreement(lexical_labels, llm_labels)
    logger.info("\n=== Inter-Method Agreement ===")
    logger.info("Cohen's Kappa: %.3f", agreement["cohens_kappa"])
    logger.info("LLM positive rate: %.3f", agreement["llm_positive_rate"])
    logger.info("Lexical positive rate: %.3f", agreement["lexical_positive_rate"])

    report = agreement["classification_report"]
    if "uncertainty" in report:
        logger.info("Lexical precision (vs LLM ground truth): %.3f", report["uncertainty"]["precision"])
        logger.info("Lexical recall (vs LLM ground truth): %.3f", report["uncertainty"]["recall"])

    # ── Save results ─────────────────────────────────────────────────
    comparison_records = []
    for s, llm_r in zip(sampled, llm_results):
        comparison_records.append({
            "sentence": s["sentence"][:200],
            "position_decile": s["position_decile"],
            "lexical_uncertain": s["lexical_has_uncertainty"],
            "llm_uncertain": llm_r.has_uncertainty,
            "llm_type": llm_r.uncertainty_type or "",
            "llm_confidence": llm_r.confidence,
            "agree": s["lexical_has_uncertainty"] == llm_r.has_uncertainty,
        })

    comparison_df = pl.DataFrame(comparison_records)
    comparison_df.write_csv(RESULTS_DIR / "classifier_comparison.csv")

    # Save agreement stats
    agreement_stats = pl.DataFrame({
        "metric": ["cohens_kappa", "llm_positive_rate", "lexical_positive_rate", "n_samples"],
        "value": [
            agreement["cohens_kappa"],
            agreement["llm_positive_rate"],
            agreement["lexical_positive_rate"],
            agreement["n_samples"],
        ],
    })
    agreement_stats.write_csv(RESULTS_DIR / "classifier_agreement.csv")

    # Analyze disagreements
    disagreements = comparison_df.filter(~pl.col("agree"))
    logger.info("\nDisagreements: %d / %d (%.1f%%)",
                len(disagreements), len(comparison_df),
                len(disagreements) / len(comparison_df) * 100)

    # False negatives (LLM says uncertain, lexicon misses)
    fn = comparison_df.filter(pl.col("llm_uncertain") & ~pl.col("lexical_uncertain"))
    logger.info("False negatives (lexicon misses): %d", len(fn))
    if len(fn) > 0:
        fn.head(20).write_csv(RESULTS_DIR / "lexicon_false_negatives.csv")

    logger.info("\nResults saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
