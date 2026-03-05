"""Uncertainty lexicon: taxonomy and detection of uncertainty markers in text.

Implements a multi-category uncertainty taxonomy grounded in linguistics
literature (Hyland 1998, Holmes 1988). Uses spaCy for sentence tokenization
and POS-aware pattern matching to reduce false positives.
"""

import logging
import re
from dataclasses import dataclass, field

import spacy

logger = logging.getLogger(__name__)

# Load spaCy model lazily
_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


# ──────────────────────────────────────────────────────────────────────
# Uncertainty Taxonomy
# ──────────────────────────────────────────────────────────────────────

UNCERTAINTY_CATEGORIES = {
    "epistemic_hedge": [
        r"\bI think\b",
        r"\bI believe\b",
        r"\bI suppose\b",
        r"\bI suspect\b",
        r"\bI assume\b",
        r"\bI imagine\b",
        r"\bI guess\b",
        r"\bI reckon\b",
        r"\bperhaps\b",
        r"\bmaybe\b",
        r"\bconceivably\b",
        r"\barguably\b",
        r"\bpresumably\b",
    ],
    "evidential_marker": [
        r"\bit seems?\b",
        r"\bit appears?\b",
        r"\bapparently\b",
        r"\bseemingly\b",
        r"\bostensibly\b",
        r"\bfrom what I (?:can |)(?:tell|see|gather|find)\b",
        r"\bas far as I (?:can tell|know|understand)\b",
        r"\bit looks like\b",
        r"\bit would (?:seem|appear)\b",
    ],
    "explicit_uncertainty": [
        r"\bI(?:'m| am) not (?:entirely |completely |fully |totally |100% )?(?:sure|certain|confident)\b",
        r"\bI(?:'m| am) (?:unsure|uncertain)\b",
        r"\bI don(?:'t| not) know\b",
        r"\bI(?:'m| am) not (?:entirely |completely |fully )?aware\b",
        r"\bI may be (?:mistaken|wrong|incorrect|off)\b",
        r"\bI could be (?:mistaken|wrong|incorrect|off)\b",
        r"\bI(?:'m| am) not positive\b",
        r"\bdon(?:'t|'t) quote me\b",
        r"\btake this with a grain of salt\b",
        r"\bI can(?:'t|not) be (?:sure|certain)\b",
        r"\bcorrect me if I(?:'m| am) wrong\b",
    ],
    "probability_language": [
        r"\blikely\b",
        r"\bunlikely\b",
        r"\bprobably\b",
        r"\bpossibly\b",
        r"\bchances are\b",
        r"\bin all likelihood\b",
        r"\bmore likely than not\b",
        r"\bthere(?:'s| is) a (?:good |fair |reasonable )?chance\b",
        r"\bprobability\b",
        r"\bplausib(?:le|ly)\b",
    ],
    "modal_hedge": [
        # These are matched with POS awareness — see detect_uncertainty_spacy
        r"\bmight\b",
        r"\bcould\b",
    ],
    "approximator": [
        # "about" is excluded from simple regex — handled via POS in spaCy
        r"\bapproximately\b",
        r"\broughly\b",
        r"\baround \d",
        r"\bmore or less\b",
        r"\bor so\b",
        r"\b-ish\b",
        r"\bin the (?:ballpark|neighborhood|vicinity)\b",
    ],
}

# Compile all patterns
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in UNCERTAINTY_CATEGORIES.items()
}


@dataclass
class UncertaintyResult:
    """Result of uncertainty detection for a single sentence."""
    text: str
    has_uncertainty: bool
    total_markers: int
    categories: dict[str, int] = field(default_factory=dict)
    matched_phrases: list[str] = field(default_factory=list)


def detect_uncertainty_regex(text: str) -> UncertaintyResult:
    """Detect uncertainty using regex patterns only (no POS awareness).

    Faster but may have more false positives for modal_hedge and approximator.
    """
    categories = {}
    matched = []

    for cat, patterns in _COMPILED_PATTERNS.items():
        count = 0
        for pat in patterns:
            matches = pat.findall(text)
            count += len(matches)
            matched.extend(matches)
        if count > 0:
            categories[cat] = count

    total = sum(categories.values())
    return UncertaintyResult(
        text=text,
        has_uncertainty=total > 0,
        total_markers=total,
        categories=categories,
        matched_phrases=matched,
    )


def detect_uncertainty_spacy(sent_text: str, sent_doc) -> UncertaintyResult:
    """Detect uncertainty with POS-aware matching for ambiguous terms.

    Uses spaCy doc for POS tags to disambiguate:
    - "could"/"might" as modal verbs vs. other uses
    - "about" as approximator vs. preposition
    """
    # Start with non-ambiguous categories
    categories = {}
    matched = []

    for cat in ["epistemic_hedge", "evidential_marker", "explicit_uncertainty", "probability_language"]:
        count = 0
        for pat in _COMPILED_PATTERNS[cat]:
            hits = pat.findall(sent_text)
            count += len(hits)
            matched.extend(hits)
        if count > 0:
            categories[cat] = count

    # POS-aware modal hedge detection
    modal_count = 0
    for token in sent_doc:
        if token.text.lower() in ("might", "could") and token.pos_ == "AUX":
            modal_count += 1
            matched.append(token.text)
    if modal_count > 0:
        categories["modal_hedge"] = modal_count

    # POS-aware approximator detection
    approx_count = 0
    for pat in _COMPILED_PATTERNS["approximator"]:
        hits = pat.findall(sent_text)
        approx_count += len(hits)
        matched.extend(hits)
    # Check "about" as approximator (followed by a number)
    for token in sent_doc:
        if token.text.lower() == "about" and token.i + 1 < len(sent_doc):
            next_token = sent_doc[token.i + 1]
            if next_token.like_num or next_token.pos_ == "NUM":
                approx_count += 1
                matched.append("about")
    if approx_count > 0:
        categories["approximator"] = approx_count

    total = sum(categories.values())
    return UncertaintyResult(
        text=sent_text,
        has_uncertainty=total > 0,
        total_markers=total,
        categories=categories,
        matched_phrases=matched,
    )


def analyze_text(text: str, use_spacy: bool = True) -> list[UncertaintyResult]:
    """Analyze a full text, returning per-sentence uncertainty results.

    Args:
        text: Full text (reasoning trace or response) to analyze.
        use_spacy: If True, use POS-aware detection. If False, regex only.

    Returns:
        List of UncertaintyResult, one per sentence.
    """
    if not text or not text.strip():
        return []

    if use_spacy:
        nlp = get_nlp()
        doc = nlp(text)
        results = []
        for sent in doc.sents:
            result = detect_uncertainty_spacy(sent.text, sent)
            results.append(result)
        return results
    else:
        # Simple sentence splitting fallback
        nlp = get_nlp()
        doc = nlp(text)
        return [detect_uncertainty_regex(sent.text) for sent in doc.sents]


def get_category_names() -> list[str]:
    """Return the list of uncertainty category names."""
    return list(UNCERTAINTY_CATEGORIES.keys())
