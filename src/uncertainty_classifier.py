"""LLM-based uncertainty classification using Ollama (local).

Used to validate and augment the lexical uncertainty detection approach.
Classifies sentences from reasoning traces for uncertainty presence, type,
and confidence level.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

CLASSIFICATION_PROMPT = """You are an expert linguist analyzing uncertainty expression in text.

For the following sentence from an LLM's internal reasoning trace, classify:

1. has_uncertainty (boolean): Does this sentence express ANY form of uncertainty, hedging, or epistemic caution?
2. uncertainty_type (string or null): If uncertain, which category best fits:
   - "epistemic_hedge" — subjective belief markers (I think, perhaps, maybe)
   - "evidential_marker" — source/evidence qualifiers (it seems, apparently)
   - "explicit_uncertainty" — direct admission of not knowing (I'm not sure, I may be wrong)
   - "probability_language" — probabilistic terms (likely, probably, chances are)
   - "modal_hedge" — possibility modals (might, could)
   - "approximator" — imprecision markers (approximately, roughly, about N)
   - null if no uncertainty
3. confidence (string): How confident is the uncertainty signal?
   - "high" — clearly and unambiguously uncertain
   - "medium" — somewhat uncertain, hedging is present but subtle
   - "low" — very slight hedging, borderline case

Respond with ONLY a JSON object, no other text:
{{"has_uncertainty": true, "uncertainty_type": "epistemic_hedge", "confidence": "high"}}

Sentence: "{sentence}"
"""


@dataclass
class ClassifierResult:
    sentence: str
    has_uncertainty: bool
    uncertainty_type: str | None
    confidence: str
    raw_response: str


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output, handling markdown fences and extra text."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def ensure_model(model: str = DEFAULT_MODEL) -> None:
    """Pull the model if it's not already available locally."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        if model in available or any(m.startswith(model.split(":")[0]) for m in available):
            return
    except requests.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it with `ollama serve` or open the Ollama app."
        )

    logger.info("Pulling model %s (this may take a few minutes)...", model)
    subprocess.run(["ollama", "pull", model], check=True)
    logger.info("Model %s ready.", model)


def classify_sentence(sentence: str, model: str = DEFAULT_MODEL) -> ClassifierResult:
    """Classify a single sentence for uncertainty using Ollama."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": CLASSIFICATION_PROMPT.format(sentence=sentence),
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 150,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()

    parsed = _extract_json(raw)
    if parsed is not None:
        return ClassifierResult(
            sentence=sentence,
            has_uncertainty=bool(parsed.get("has_uncertainty", False)),
            uncertainty_type=parsed.get("uncertainty_type"),
            confidence=parsed.get("confidence", "low"),
            raw_response=raw,
        )
    else:
        logger.warning("Failed to parse LLM response: %s", raw[:200])
        return ClassifierResult(
            sentence=sentence,
            has_uncertainty=False,
            uncertainty_type=None,
            confidence="low",
            raw_response=raw,
        )


def classify_batch(
    sentences: list[str],
    model: str = DEFAULT_MODEL,
) -> list[ClassifierResult]:
    """Classify a batch of sentences."""
    from tqdm import tqdm

    ensure_model(model)

    results = []
    parse_failures = 0
    for sent in tqdm(sentences, desc="Classifying sentences"):
        result = classify_sentence(sent, model=model)
        if result.raw_response and _extract_json(result.raw_response) is None:
            parse_failures += 1
        results.append(result)

    if parse_failures > 0:
        logger.warning("JSON parse failures: %d / %d (%.1f%%)",
                       parse_failures, len(sentences),
                       parse_failures / len(sentences) * 100)
    return results


def compute_agreement(lexical_labels: list[bool], llm_labels: list[bool]) -> dict:
    """Compute inter-method agreement statistics."""
    from sklearn.metrics import cohen_kappa_score, classification_report

    kappa = cohen_kappa_score(llm_labels, lexical_labels)
    report = classification_report(
        llm_labels, lexical_labels, target_names=["no_uncertainty", "uncertainty"],
        output_dict=True,
    )

    return {
        "cohens_kappa": float(kappa),
        "classification_report": report,
        "n_samples": len(lexical_labels),
        "llm_positive_rate": sum(llm_labels) / len(llm_labels),
        "lexical_positive_rate": sum(lexical_labels) / len(lexical_labels),
    }
