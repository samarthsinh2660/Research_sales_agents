"""Unit tests for lead_research.py's pure helpers - no API calls."""
from sales_outreach.tools.lead_research import extract_company_name


def test_extracts_domain_from_email():
    assert extract_company_name("someone@example.com") == "example.com"


def test_malformed_email_returns_not_found():
    assert extract_company_name("not-an-email") == "Company not found"


def test_empty_string_returns_not_found():
    assert extract_company_name("") == "Company not found"
