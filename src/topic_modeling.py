"""Topic modeling for user prompts using BERTopic."""

import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fit_topic_model(
    prompts: list[str],
    n_topics: str = "auto",
    embedding_model: str = "all-MiniLM-L6-v2",
    min_topic_size: int = 50,
    seed: int = 42,
):
    """Fit a BERTopic model on user prompts.

    Args:
        prompts: List of user prompt strings.
        n_topics: Number of topics or "auto" for automatic detection.
        embedding_model: Sentence transformer model name.
        min_topic_size: Minimum cluster size for a topic.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (BERTopic model, topics list, probabilities).
    """
    from bertopic import BERTopic
    from sklearn.cluster import MiniBatchKMeans
    from umap import UMAP

    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        min_topic_size=min_topic_size,
        nr_topics=n_topics if n_topics != "auto" else None,
        verbose=True,
    )

    logger.info("Fitting BERTopic on %d prompts...", len(prompts))
    topics, probs = topic_model.fit_transform(prompts)
    logger.info("Found %d topics", len(set(topics)) - (1 if -1 in topics else 0))

    return topic_model, topics, probs


def get_topic_info(topic_model) -> pl.DataFrame:
    """Extract topic info as a Polars DataFrame (CSV-safe, no nested columns)."""
    info = topic_model.get_topic_info()
    # Drop list/object columns that can't be written to CSV
    safe_cols = [c for c in info.columns if info[c].dtype.name not in ("object",) and not hasattr(info[c].iloc[0] if len(info) > 0 else None, "__iter__") or isinstance(info[c].iloc[0] if len(info) > 0 else "", str)]
    # Simpler approach: keep only scalar columns
    scalar_info = info[["Topic", "Count", "Name"]].copy()
    # Add top representation words as a single string
    if "Representation" in info.columns:
        scalar_info["Representation"] = info["Representation"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
    return pl.from_pandas(scalar_info)


def assign_topics_to_df(
    df: pl.DataFrame,
    topics: list[int],
    topic_labels: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Add topic assignments to the main DataFrame."""
    df = df.with_columns(
        pl.Series("topic_id", topics),
    )

    if topic_labels:
        df = df.with_columns(
            pl.col("topic_id")
            .map_elements(lambda x: topic_labels.get(x, "other"), return_dtype=pl.Utf8)
            .alias("topic_label"),
        )

    return df


def save_topic_model(topic_model, path: Path):
    """Save the fitted BERTopic model. Falls back gracefully on pickle errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import shutil
        if path.exists():
            shutil.rmtree(path)
        topic_model.save(str(path), serialization="safetensors", save_ctfidf=True)
        logger.info("Saved topic model to %s", path)
    except Exception as e:
        logger.warning("Could not save topic model (pickle/serialization issue): %s", e)
        logger.warning("Topic assignments and info CSV are saved — model can be refit if needed.")
