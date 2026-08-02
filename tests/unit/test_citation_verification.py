"""Unit tests for verify_citations - pure text processing, no API calls."""
import logging

from agents.research.utils import verify_citations


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


def test_url_inside_a_markdown_link_is_matched():
    # Findings carry URLs inside markdown links, so they arrive with a trailing ")".
    # Comparing raw against a cleaned citation dropped every real source.
    findings = "See [SRHU](https://srhu.edu.in/contact-us/) for the registrar."
    report = "Body [1].\n\n### Sources\n[1] SRHU: https://srhu.edu.in/contact-us/\n"
    assert "srhu.edu.in" in verify_citations(report, findings)


def test_url_followed_by_a_comma_is_matched():
    findings = "See https://graphicera.ac.in, which lists placements."
    report = "Body [1].\n\n### Sources\n[1] Graphic Era: https://graphicera.ac.in\n"
    assert "graphicera.ac.in" in verify_citations(report, findings)


def test_trailing_slash_difference_is_matched():
    findings = "Programs at https://upes.ac.in/"
    report = "Body [1].\n\n### Sources\n[1] UPES: https://upes.ac.in\n"
    assert "upes.ac.in" in verify_citations(report, findings)


def test_real_sources_survive_alongside_a_fabricated_one():
    findings = "See [SRHU](https://srhu.edu.in/) and https://upes.ac.in, both real."
    report = (
        "Body [1] [2] [3].\n\n### Sources\n"
        "[1] SRHU: https://srhu.edu.in/\n"
        "[2] UPES: https://upes.ac.in\n"
        "[3] Invented: https://not-in-findings.example.com\n"
    )
    out = verify_citations(report, findings)
    assert "srhu.edu.in" in out and "upes.ac.in" in out
    assert "not-in-findings" not in out
