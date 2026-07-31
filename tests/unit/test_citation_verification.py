"""Unit tests for verify_citations - pure text processing, no API calls."""
import logging

from open_deep_research.utils import verify_citations


def test_keeps_citation_with_matching_url():
    report = "### Sources\n[1] Real Source: https://real.example.com/page\n"
    findings = "Some finding mentions https://real.example.com/page as evidence."
    result = verify_citations(report, findings)
    assert "https://real.example.com/page" in result


def test_drops_citation_with_no_matching_url():
    report = (
        "### Sources\n"
        "[1] Real Source: https://real.example.com/page\n"
        "[2] Fake Source: https://hallucinated.example.com/page\n"
    )
    findings = "Mentions https://real.example.com/page only."
    result = verify_citations(report, findings)
    assert "https://real.example.com/page" in result
    assert "https://hallucinated.example.com/page" not in result


def test_non_citation_lines_are_always_kept():
    report = "# Title\n\nSome body text with no citation markers.\n"
    result = verify_citations(report, findings="irrelevant")
    assert result == report


def test_warns_on_mass_strip(caplog):
    report = "### Sources\n[1] A: https://a.example.com/\n[2] B: https://b.example.com/\n"
    findings = "nothing matches here"
    with caplog.at_level(logging.WARNING):
        verify_citations(report, findings)
    assert any("dropped" in record.message for record in caplog.records)


def test_no_warning_when_citations_mostly_survive(caplog):
    report = "### Sources\n[1] A: https://a.example.com/\n[2] B: https://b.example.com/\n"
    findings = "mentions https://a.example.com/ and https://b.example.com/"
    with caplog.at_level(logging.WARNING):
        verify_citations(report, findings)
    assert not any("dropped" in record.message for record in caplog.records)
