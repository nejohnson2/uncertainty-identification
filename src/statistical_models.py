"""Statistical models: mixed-effects regression for uncertainty analysis."""

import logging

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


def fit_positional_model(
    sentence_df: pl.DataFrame,
    formula: str | None = None,
) -> object:
    """Fit a mixed-effects logistic regression for positional uncertainty.

    Model: uncertainty ~ position + position^2 + model_family + log(reasoning_len)
           + nsfw_flag + topic + (1|interaction_id)

    Args:
        sentence_df: Sentence-level DataFrame with uncertainty and position columns.
        formula: Optional custom formula. If None, uses default.

    Returns:
        Fitted MixedLM result object.
    """
    pdf = sentence_df.to_pandas()

    # Create derived columns
    pdf["position_sq"] = pdf["normalized_position"] ** 2
    pdf["has_uncertainty_int"] = pdf["has_uncertainty"].astype(int)

    if "total_sentences" in pdf.columns:
        pdf["log_reasoning_len"] = np.log1p(pdf["total_sentences"])

    if formula is None:
        formula = "has_uncertainty_int ~ normalized_position + position_sq"
        if "model_family" in pdf.columns:
            formula += " + C(model_family)"
        if "log_reasoning_len" in pdf.columns:
            formula += " + log_reasoning_len"
        if "nsfw_flag" in pdf.columns:
            pdf["nsfw_int"] = pdf["nsfw_flag"].astype(int)
            formula += " + nsfw_int"
        if "topic_label" in pdf.columns:
            formula += " + C(topic_label)"

    logger.info("Fitting model: %s", formula)
    logger.info("N observations: %d", len(pdf))

    model = smf.mixedlm(
        formula,
        data=pdf,
        groups=pdf["interaction_id"],
    )
    result = model.fit(reml=False)
    logger.info("Model converged: %s", result.converged)
    return result


def fit_filtering_model(
    filtering_df: pl.DataFrame,
    formula: str | None = None,
) -> object:
    """Fit a linear mixed model for the filtering ratio.

    Model: filtering_ratio ~ model_family + topic + nsfw_flag + log(reasoning_len)
           + (1|model)
    """
    pdf = filtering_df.to_pandas()

    if "reasoning_n_sentences" in pdf.columns:
        pdf["log_reasoning_len"] = np.log1p(pdf["reasoning_n_sentences"])

    if formula is None:
        formula = "filtering_ratio ~ 1"
        if "model_family" in pdf.columns:
            formula += " + C(model_family)"
        if "log_reasoning_len" in pdf.columns:
            formula += " + log_reasoning_len"
        if "nsfw_flag" in pdf.columns:
            pdf["nsfw_int"] = pdf["nsfw_flag"].astype(int)
            formula += " + nsfw_int"
        if "topic_label" in pdf.columns:
            formula += " + C(topic_label)"

    logger.info("Fitting filtering model: %s", formula)

    # Use model variant as grouping variable if available
    groups = pdf.get("model_family", pdf.get("model", pd.Series(["all"] * len(pdf))))

    model = smf.mixedlm(formula, data=pdf, groups=groups)
    result = model.fit(reml=False)
    logger.info("Model converged: %s", result.converged)
    return result


def extract_results_table(result) -> pl.DataFrame:
    """Extract model results as a clean Polars DataFrame."""
    summary = result.summary().tables[1]
    return pl.from_pandas(summary.reset_index() if hasattr(summary, 'reset_index') else summary)
