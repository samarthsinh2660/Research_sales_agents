"""Unit tests for pure routing/decision helpers - no LLM calls, no graph."""
from agents.outreach.utils import (
    has_remaining_leads,
    qualification_decision,
    research_sufficiency_decision,
)

DEFAULT_THRESHOLD = 7.0
DEFAULT_MAX_RETRIES = 1


def test_has_remaining_leads_found():
    assert has_remaining_leads({"number_leads": 3}) == "Found leads"


def test_has_remaining_leads_none():
    assert has_remaining_leads({"number_leads": 0}) == "No more leads"


def test_qualification_above_threshold():
    assert qualification_decision("7.5", DEFAULT_THRESHOLD) == "qualified"


def test_qualification_at_threshold():
    assert qualification_decision("7.0", DEFAULT_THRESHOLD) == "qualified"


def test_qualification_below_threshold():
    assert qualification_decision("6.9", DEFAULT_THRESHOLD) == "not qualified"


def test_qualification_unparseable_score_does_not_crash():
    # The scoring model returns free text; a non-numeric response must fail closed
    # rather than raising ValueError mid-pipeline.
    assert qualification_decision("N/A", DEFAULT_THRESHOLD) == "not qualified"


def test_qualification_respects_custom_threshold():
    assert qualification_decision("8.0", 9.0) == "not qualified"
    assert qualification_decision("8.0", 5.0) == "qualified"


def test_research_sufficient():
    assert research_sufficiency_decision(
        {"research_sufficient": True}, DEFAULT_MAX_RETRIES
    ) == "sufficient"


def test_research_insufficient_first_attempt_retries():
    assert research_sufficiency_decision(
        {"research_sufficient": False, "research_retry_count": 0}, DEFAULT_MAX_RETRIES
    ) == "retry"


def test_research_insufficient_after_retry_gives_up():
    assert research_sufficiency_decision(
        {"research_sufficient": False, "research_retry_count": 1}, DEFAULT_MAX_RETRIES
    ) == "insufficient"


def test_research_missing_key_defaults_to_retry_first():
    # If the sufficiency check never ran for some reason, don't silently proceed
    # to score/pitch on missing data - fail closed, retry once before giving up
    assert research_sufficiency_decision({}, DEFAULT_MAX_RETRIES) == "retry"


def test_research_retries_disabled_gives_up_immediately():
    assert research_sufficiency_decision(
        {"research_sufficient": False, "research_retry_count": 0}, 0
    ) == "insufficient"
