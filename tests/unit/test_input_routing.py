"""Guards how a typed line is interpreted.

Naming a URL inside a sentence used to change the behaviour: "find the people from <url>"
matched the discovery verb and ran a single search instead of using the page, silently
answering a different question than the one asked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from pace import looks_like_discovery, url_in  # noqa: E402

URL = "https://cio.economictimes.indiatimes.com/annual-conclave2025"


def test_bare_url_uses_the_page():
    assert not looks_like_discovery(URL)


def test_url_inside_a_sentence_still_uses_the_page():
    assert not looks_like_discovery(f"find the each people details from this website {URL}")
    assert not looks_like_discovery(f"find all people and contacts from {URL}")


def test_url_is_extracted_from_anywhere_in_the_line():
    assert url_in(f"find people from {URL} please") == URL
    assert url_in("Rishabh Software") is None


def test_text_without_a_url_is_a_search():
    assert looks_like_discovery("find colleges in Dehradun")


def test_a_company_name_is_not_a_search():
    assert not looks_like_discovery("Finder Technologies")
    assert not looks_like_discovery("Acme, Globex")
