"""Unit tests for pure routing/decision functions in nodes.py - no LLM calls."""
from src.nodes import OutReachAutomationNodes


def test_check_if_there_more_leads_found():
    assert OutReachAutomationNodes.check_if_there_more_leads({"number_leads": 3}) == "Found leads"


def test_check_if_there_more_leads_none():
    assert OutReachAutomationNodes.check_if_there_more_leads({"number_leads": 0}) == "No more leads"


def test_check_if_qualified_above_threshold():
    assert OutReachAutomationNodes.check_if_qualified({"lead_score": "7.5"}) == "qualified"


def test_check_if_qualified_at_threshold():
    assert OutReachAutomationNodes.check_if_qualified({"lead_score": "7.0"}) == "qualified"


def test_check_if_qualified_below_threshold():
    assert OutReachAutomationNodes.check_if_qualified({"lead_score": "6.9"}) == "not qualified"


def test_check_if_research_sufficient_true():
    assert OutReachAutomationNodes.check_if_research_sufficient({"research_sufficient": True}) == "sufficient"


def test_check_if_research_sufficient_false_first_attempt_retries():
    result = OutReachAutomationNodes.check_if_research_sufficient(
        {"research_sufficient": False, "research_retry_count": 0}
    )
    assert result == "retry"


def test_check_if_research_sufficient_false_after_retry_gives_up():
    result = OutReachAutomationNodes.check_if_research_sufficient(
        {"research_sufficient": False, "research_retry_count": 1}
    )
    assert result == "insufficient"


def test_check_if_research_sufficient_missing_key_defaults_to_retry_first():
    # If the sufficiency check never ran for some reason, don't silently proceed
    # to score/pitch on missing data - fail closed, retry once before giving up
    assert OutReachAutomationNodes.check_if_research_sufficient({}) == "retry"
