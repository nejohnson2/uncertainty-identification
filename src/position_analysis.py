"""Positional uncertainty analysis: measure uncertainty across reasoning chain positions."""

import logging
from dataclasses import dataclass

import polars as pl
from tqdm import tqdm

from src.uncertainty_lexicon import analyze_text, get_category_names

logger = logging.getLogger(__name__)


@dataclass
class SentenceRecord:
    """A single sentence with positional and uncertainty metadata."""
    interaction_id: int
    sentence_idx: int
    total_sentences: int
    normalized_position: float
    position_decile: int
    text: str
    has_uncertainty: bool
    uncertainty_count: int
    categories: dict[str, int]


def compute_sentence_records(
    reasoning_text: str,
    interaction_id: int,
    use_spacy: bool = True,
) -> list[SentenceRecord]:
    """Analyze a single reasoning trace and return per-sentence records."""
    results = analyze_text(reasoning_text, use_spacy=use_spacy)
    if not results:
        return []

    total = len(results)
    records = []
    for idx, r in enumerate(results):
        norm_pos = idx / max(total - 1, 1)
        decile = min(int(norm_pos * 10) + 1, 10)
        records.append(SentenceRecord(
            interaction_id=interaction_id,
            sentence_idx=idx,
            total_sentences=total,
            normalized_position=norm_pos,
            position_decile=decile,
            text=r.text,
            has_uncertainty=r.has_uncertainty,
            uncertainty_count=r.total_markers,
            categories=r.categories,
        ))
    return records


def build_sentence_dataframe(
    df: pl.DataFrame,
    reasoning_col: str = "reasoning",
    id_col: str | None = None,
    use_spacy: bool = True,
) -> pl.DataFrame:
    """Process all reasoning traces and build a sentence-level DataFrame.

    Args:
        df: DataFrame with reasoning traces.
        reasoning_col: Column name containing reasoning text.
        id_col: Optional column to use as interaction ID. If None, uses row index.
        use_spacy: Whether to use POS-aware detection.

    Returns:
        DataFrame with one row per sentence, including positional and
        uncertainty metadata.
    """
    category_names = get_category_names()

    all_rows = []
    reasoning_series = df[reasoning_col]
    ids = df[id_col] if id_col else range(len(df))

    for i, (iid, text) in enumerate(tqdm(
        zip(ids, reasoning_series),
        total=len(df),
        desc="Analyzing reasoning traces",
    )):
        if text is None or not str(text).strip():
            continue

        records = compute_sentence_records(str(text), iid, use_spacy=use_spacy)
        for rec in records:
            row = {
                "interaction_id": rec.interaction_id,
                "sentence_idx": rec.sentence_idx,
                "total_sentences": rec.total_sentences,
                "normalized_position": rec.normalized_position,
                "position_decile": rec.position_decile,
                "sentence_text": rec.text,
                "has_uncertainty": rec.has_uncertainty,
                "uncertainty_count": rec.uncertainty_count,
            }
            for cat in category_names:
                row[f"cat_{cat}"] = rec.categories.get(cat, 0)
            all_rows.append(row)

    logger.info("Built %d sentence records from %d interactions", len(all_rows), len(df))
    return pl.DataFrame(all_rows)


def aggregate_by_decile(sentence_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate uncertainty statistics by position decile."""
    category_names = get_category_names()
    cat_cols = [f"cat_{c}" for c in category_names]

    agg_exprs = [
        pl.len().alias("n_sentences"),
        pl.col("has_uncertainty").mean().alias("uncertainty_rate"),
        pl.col("uncertainty_count").mean().alias("mean_uncertainty_count"),
    ]
    for col in cat_cols:
        agg_exprs.append(
            (pl.col(col) > 0).mean().alias(f"{col}_rate")
        )

    return (
        sentence_df
        .group_by("position_decile")
        .agg(agg_exprs)
        .sort("position_decile")
    )


def aggregate_by_decile_and_group(
    sentence_df: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """Aggregate uncertainty by position decile and a grouping variable."""
    return (
        sentence_df
        .group_by(["position_decile", group_col])
        .agg([
            pl.len().alias("n_sentences"),
            pl.col("has_uncertainty").mean().alias("uncertainty_rate"),
            pl.col("uncertainty_count").mean().alias("mean_uncertainty_count"),
        ])
        .sort(["position_decile", group_col])
    )
