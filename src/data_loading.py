"""Download and load the real-slop dataset from Hugging Face."""

import logging
from pathlib import Path

import polars as pl
from datasets import load_dataset

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PARQUET_PATH = DATA_DIR / "real_slop.parquet"
HF_DATASET_ID = "Solenopsisbot/real-slop"


def download_dataset(force: bool = False) -> Path:
    """Download dataset from HuggingFace and save as local parquet."""
    if PARQUET_PATH.exists() and not force:
        logger.info("Dataset already exists at %s", PARQUET_PATH)
        return PARQUET_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s ...", HF_DATASET_ID)
    ds = load_dataset(HF_DATASET_ID, split="train")
    ds.to_parquet(str(PARQUET_PATH))
    logger.info("Saved to %s (%d rows)", PARQUET_PATH, len(ds))
    return PARQUET_PATH


def load_full(path: Path | None = None) -> pl.LazyFrame:
    """Load the full dataset as a Polars LazyFrame for memory-efficient processing."""
    path = path or PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `make download` first."
        )
    return pl.scan_parquet(path)


def load_sample(frac: float = 0.05, seed: int = 42, path: Path | None = None) -> pl.DataFrame:
    """Load a random sample of the dataset into memory."""
    lf = load_full(path)
    total = lf.select(pl.len()).collect().item()
    n = int(total * frac)
    logger.info("Sampling %d rows (%.1f%% of %d)", n, frac * 100, total)
    return lf.collect().sample(n=n, seed=seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_dataset()
