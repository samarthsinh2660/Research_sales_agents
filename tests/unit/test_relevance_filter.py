"""Unit tests for filter_by_relevance - pure filtering logic, no API calls."""
from agents.research.utils import filter_by_relevance


def test_filters_below_threshold():
    results = {"a": {"score": 0.5}, "b": {"score": 0.1}, "c": {"score": 0.9}}
    filtered = filter_by_relevance(results, 0.3)
    assert set(filtered.keys()) == {"a", "c"}


def test_falls_back_to_unfiltered_if_all_dropped():
    results = {"a": {"score": 0.1}, "b": {"score": 0.05}}
    filtered = filter_by_relevance(results, 0.9)
    assert filtered == results


def test_missing_score_defaults_to_kept():
    results = {"a": {}}
    filtered = filter_by_relevance(results, 0.5)
    assert "a" in filtered


def test_regression_g2_style_result_survives_at_0_2_not_0_4():
    # Real evidence from this session: G2 scored 0.329 for a company query -
    # legitimate third-party source, must survive at the actual default (0.2).
    # A second, higher-scoring result is included so filtering at 0.4 doesn't
    # empty the dict entirely and trip the "keep everything" safety fallback.
    results = {"g2_review": {"score": 0.329}, "official_site": {"score": 0.85}}
    assert "g2_review" in filter_by_relevance(results, 0.2)
    assert "g2_review" not in filter_by_relevance(results, 0.4)
