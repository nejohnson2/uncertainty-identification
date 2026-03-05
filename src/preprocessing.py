"""Preprocessing: filter, clean, and parse the raw dataset."""

import json
import logging
import re
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "cleaned.parquet"

# Minimal NSFW keyword list for flagging (not exhaustive — just a starting signal)
_NSFW_KEYWORDS = re.compile(
    r"\b(nsfw|explicit|sexual|erotic|porn|hentai|xxx)\b",
    re.IGNORECASE,
)


def extract_model_family(model_col: pl.Expr) -> pl.Expr:
    """Map full model identifier to model family."""
    return (
        pl.when(model_col.str.contains("opus"))
        .then(pl.lit("claude-opus"))
        .when(model_col.str.contains("sonnet"))
        .then(pl.lit("claude-sonnet"))
        .when(model_col.str.contains("haiku"))
        .then(pl.lit("claude-haiku"))
        .when(model_col.str.contains("gpt"))
        .then(pl.lit("gpt"))
        .when(model_col.str.contains("gemini"))
        .then(pl.lit("gemini"))
        .when(model_col.str.contains("kimi"))
        .then(pl.lit("kimi"))
        .when(model_col.str.contains("deepseek"))
        .then(pl.lit("deepseek"))
        .when(model_col.str.contains("glm"))
        .then(pl.lit("glm"))
        .when(model_col.str.contains("qwen"))
        .then(pl.lit("qwen"))
        .when(model_col.str.contains("minimax"))
        .then(pl.lit("minimax"))
        .otherwise(pl.lit("other"))
    )


def extract_model_provider(model_col: pl.Expr) -> pl.Expr:
    """Map model identifier to provider (anthropic, openai, google, other)."""
    return (
        pl.when(model_col.str.contains("claude|opus|sonnet|haiku"))
        .then(pl.lit("anthropic"))
        .when(model_col.str.contains("gpt"))
        .then(pl.lit("openai"))
        .when(model_col.str.contains("gemini"))
        .then(pl.lit("google"))
        .otherwise(pl.lit("other"))
    )


def extract_user_prompt(messages_str: str) -> str:
    """Extract the last user message from the messages JSON string."""
    try:
        msgs = json.loads(messages_str)
        if isinstance(msgs, list):
            user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            if user_msgs:
                content = user_msgs[-1].get("content", "")
                if isinstance(content, list):
                    # Handle content blocks format
                    text_parts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    return " ".join(text_parts)
                return str(content)
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return ""


def count_turns(messages_str: str) -> int:
    """Count number of conversation turns from messages JSON."""
    try:
        msgs = json.loads(messages_str)
        if isinstance(msgs, list):
            return len(msgs)
    except (json.JSONDecodeError, TypeError):
        pass
    return 0


def preprocess(df: pl.DataFrame, min_reasoning_sentences: int = 5) -> pl.DataFrame:
    """Clean and filter the dataset for analysis.

    Filters to rows with non-null reasoning, extracts metadata,
    and flags NSFW content.
    """
    logger.info("Starting preprocessing on %d rows", len(df))

    # Filter to rows with non-null, non-empty reasoning
    df = df.filter(
        pl.col("reasoning").is_not_null()
        & (pl.col("reasoning").str.len_chars() > 0)
    )
    logger.info("Rows with reasoning: %d", len(df))

    # Deduplicate on (messages, model, response) — removes exact copies
    # while preserving cross-model comparisons and keeping one copy of re-generations
    n_before = len(df)
    df = df.unique(subset=["messages", "model", "response"])
    n_removed = n_before - len(df)
    logger.info("Deduplication: removed %d rows (%.1f%%), %d remaining",
                n_removed, n_removed / n_before * 100, len(df))

    # Extract model family and provider
    df = df.with_columns(
        extract_model_family(pl.col("model")).alias("model_family"),
        extract_model_provider(pl.col("model")).alias("model_provider"),
    )

    # Extract user prompt and turn count (using map_elements for JSON parsing)
    df = df.with_columns(
        pl.col("messages").map_elements(extract_user_prompt, return_dtype=pl.Utf8).alias("user_prompt"),
        pl.col("messages").map_elements(count_turns, return_dtype=pl.Int64).alias("n_turns"),
    )

    # Compute text lengths
    df = df.with_columns(
        pl.col("reasoning").str.len_chars().alias("reasoning_len_chars"),
        pl.col("response").fill_null("").str.len_chars().alias("response_len_chars"),
        pl.col("user_prompt").str.len_chars().alias("prompt_len_chars"),
    )

    # Flag NSFW content (check both prompt and response)
    df = df.with_columns(
        (
            pl.col("user_prompt").str.contains(_NSFW_KEYWORDS.pattern)
            | pl.col("response").fill_null("").str.contains(_NSFW_KEYWORDS.pattern)
        ).alias("nsfw_flag"),
    )

    logger.info("Preprocessing complete: %d rows", len(df))
    return df


def save_processed(df: pl.DataFrame, path: Path | None = None) -> Path:
    """Save the processed dataset."""
    path = path or PROCESSED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Saved processed data to %s", path)
    return path
