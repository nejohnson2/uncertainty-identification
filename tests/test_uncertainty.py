"""Unit tests for uncertainty lexicon detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.uncertainty_lexicon import detect_uncertainty_regex, analyze_text, UncertaintyResult


def test_epistemic_hedge_detected():
    result = detect_uncertainty_regex("I think this is correct.")
    assert result.has_uncertainty
    assert "epistemic_hedge" in result.categories


def test_explicit_uncertainty_detected():
    result = detect_uncertainty_regex("I'm not sure about this answer.")
    assert result.has_uncertainty
    assert "explicit_uncertainty" in result.categories


def test_probability_language_detected():
    result = detect_uncertainty_regex("This is probably the right approach.")
    assert result.has_uncertainty
    assert "probability_language" in result.categories


def test_evidential_marker_detected():
    result = detect_uncertainty_regex("It seems like this could work.")
    assert result.has_uncertainty
    assert "evidential_marker" in result.categories


def test_no_uncertainty():
    result = detect_uncertainty_regex("The capital of France is Paris.")
    assert not result.has_uncertainty
    assert result.total_markers == 0


def test_multiple_categories():
    result = detect_uncertainty_regex("I think this is probably correct, but I'm not sure.")
    assert result.has_uncertainty
    assert result.total_markers >= 3
    assert "epistemic_hedge" in result.categories
    assert "probability_language" in result.categories
    assert "explicit_uncertainty" in result.categories


def test_analyze_text_returns_per_sentence():
    text = "This is certain. I think this might be wrong. The answer is 42."
    results = analyze_text(text, use_spacy=True)
    assert len(results) >= 2  # At least 2 sentences
    # At least one should have uncertainty
    assert any(r.has_uncertainty for r in results)
    # At least one should not
    assert any(not r.has_uncertainty for r in results)


def test_empty_text():
    results = analyze_text("", use_spacy=True)
    assert results == []


def test_modal_hedge():
    result = detect_uncertainty_regex("This might be the issue.")
    assert result.has_uncertainty
    assert "modal_hedge" in result.categories


def test_approximator():
    result = detect_uncertainty_regex("There are approximately 100 items.")
    assert result.has_uncertainty
    assert "approximator" in result.categories


def test_i_may_be_mistaken():
    result = detect_uncertainty_regex("I may be mistaken, but I recall it differently.")
    assert result.has_uncertainty
    assert "explicit_uncertainty" in result.categories


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
