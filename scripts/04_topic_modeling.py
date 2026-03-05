"""Topic modeling on user prompts using BERTopic.

Extracts user prompts, generates embeddings, fits BERTopic,
and saves topic assignments. Designed to run on the cluster.
"""

import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import load_full
from src.preprocessing import preprocess
from src.topic_modeling import fit_topic_model, get_topic_info, assign_topics_to_df, save_topic_model

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

    # Extract non-empty user prompts
    prompts = df.filter(
        pl.col("user_prompt").is_not_null()
        & (pl.col("user_prompt").str.len_chars() > 10)
    )["user_prompt"].to_list()
    logger.info("Prompts for topic modeling: %d", len(prompts))

    # ── Fit BERTopic ─────────────────────────────────────────────────
    topic_model, topics, probs = fit_topic_model(
        prompts,
        embedding_model="all-MiniLM-L6-v2",
        min_topic_size=50,
        seed=42,
    )

    # ── Save topic info ──────────────────────────────────────────────
    topic_info = get_topic_info(topic_model)
    topic_info.write_csv(RESULTS_DIR / "topic_info.csv")
    logger.info("\n=== Topic Summary ===")
    for row in topic_info.head(20).iter_rows(named=True):
        logger.info("  Topic %3d (%5d docs): %s",
                     row.get("Topic", -1),
                     row.get("Count", 0),
                     row.get("Name", "")[:60])

    # Save topic model
    save_topic_model(topic_model, DATA_DIR / "topic_model")

    # Save topic assignments
    # Note: topics list corresponds to the filtered prompts, need to align
    prompt_topics = pl.DataFrame({
        "user_prompt": prompts,
        "topic_id": topics,
    })
    prompt_topics.write_parquet(DATA_DIR / "prompt_topics.parquet")
    logger.info("Topic assignments saved to %s", DATA_DIR / "prompt_topics.parquet")

    logger.info("\nTopic modeling complete")


if __name__ == "__main__":
    main()
