"""Guards detection of anti-bot interstitials.

A Cloudflare challenge arrives as HTTP 200 with a full body, so every success check
downstream passes and no fallback fires. On a blocked directory that yielded exactly one
extracted "company", named Cloudflare - a wrong answer dressed as a right one.
"""
from orchestrator.targets import looks_blocked

REAL_PAGE = "Speakers " + "Rucha Nanavati CIO Mahindra Group. " * 60


def test_detects_cloudflare_challenge():
    assert looks_blocked("Just a moment...\nEnable JavaScript and cookies to continue")


def test_detects_human_verification():
    assert looks_blocked("Verify you are human by completing the action below." + "x" * 2000)


def test_detects_access_denied():
    assert looks_blocked("Access Denied\nYou do not have permission." + "y" * 2000)


def test_treats_a_stub_page_as_blocked():
    # An empty shell is indistinguishable from a block for our purposes: escalate either way.
    assert looks_blocked("<html><body>Loading...</body></html>")


def test_empty_and_none_are_blocked():
    assert looks_blocked("")
    assert looks_blocked(None)


def test_real_content_is_not_blocked():
    assert not looks_blocked(REAL_PAGE)


def test_marker_deep_in_a_real_page_does_not_trip_detection():
    # "access denied" can appear in ordinary body copy; only the head of the page counts.
    page = REAL_PAGE + " Our policy explains what happens when access denied errors occur."
    assert not looks_blocked(page)
