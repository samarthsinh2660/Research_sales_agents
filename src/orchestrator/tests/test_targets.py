"""Unit tests for target resolution - no network for the pure helpers."""
from orchestrator.targets import extract_sheet_id, parse_inline_list


def test_extract_sheet_id_from_full_url():
    url = "https://docs.google.com/spreadsheets/d/1AbC_dEF-123/edit#gid=0"
    assert extract_sheet_id(url) == "1AbC_dEF-123"


def test_extract_sheet_id_returns_none_for_non_sheet():
    assert extract_sheet_id("https://example.com/about") is None


def test_parse_inline_list_comma_separated():
    assert parse_inline_list("research Acme Corp, Globex, Initech") == ["Acme Corp", "Globex", "Initech"]


def test_parse_inline_list_newline_separated():
    assert parse_inline_list("Acme Corp\nGlobex\nInitech") == ["Acme Corp", "Globex", "Initech"]


def test_parse_inline_list_single_name_is_one_item():
    assert parse_inline_list("Shital Infotech") == ["Shital Infotech"]


def test_parse_inline_list_strips_and_drops_empties():
    assert parse_inline_list("Acme,  , Globex,") == ["Acme", "Globex"]


import pathlib

from orchestrator.targets import parse_listing_content

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "listing_page.txt"


def _parse_fixture():
    return parse_listing_content(FIXTURE.read_text())


def test_listing_extracts_named_people_with_role_and_org():
    names = {t.name for t in _parse_fixture()}
    assert "Rucha Nanavati" in names
    assert "Dheeraj Sinha" in names


def test_listing_person_carries_seed_context():
    rucha = next(t for t in _parse_fixture() if t.name == "Rucha Nanavati")
    assert "CIO" in rucha.context and "Mahindra Group" in rucha.context


def test_listing_extracts_sponsors_that_have_alt_text():
    names = {t.name for t in _parse_fixture()}
    assert {"Adobe", "RedisLabs", "Exotel"} <= names


def test_listing_skips_logos_without_alt_text():
    # "Image 9" is a bare logo with no alt text - it must not become a target.
    assert not any(t.name.startswith("Image") for t in _parse_fixture())


def test_listing_targets_are_page_sourced_and_get_no_crm_row():
    assert all(t.source == "page" and t.crm_row_id is None for t in _parse_fixture())
