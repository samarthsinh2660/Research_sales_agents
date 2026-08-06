"""Guards the scroll-to-bottom loop against stopping on a lazy-load pause.

A lazy-loading page plateaus in height while it waits on a network request. Stopping at
the first unchanged reading cut a 123-target page down to 33 - the same page and code,
differing only in how one fetch happened to be timed.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.targets import render_page_text


class FakePage:
    """Page whose height plateaus mid-load before growing again."""

    def __init__(self, heights):
        self.heights = list(heights)
        self.calls = 0

    async def goto(self, *a, **kw):
        return None

    async def wait_for_timeout(self, *a, **kw):
        return None

    async def evaluate(self, script):
        if "scrollHeight" in script and "scrollTo" not in script:
            value = self.heights[min(self.calls, len(self.heights) - 1)]
            self.calls += 1
            return value
        return None

    def locator(self, _sel):
        loc = MagicMock()
        loc.inner_text = AsyncMock(return_value="x" * self.calls)
        return loc


def _render(heights, **kw):
    page = FakePage(heights)
    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=page)
    browser.close = AsyncMock()
    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)
    pw = MagicMock(chromium=chromium)

    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=pw)
    manager.__aexit__ = AsyncMock(return_value=False)

    with patch("playwright.async_api.async_playwright", return_value=manager):
        asyncio.run(render_page_text("http://x", settle_ms=0, **kw))
    return page


def test_keeps_scrolling_through_a_lazy_load_pause():
    # Height stalls at 100 for two checks while a fetch is in flight, then grows to 300.
    page = _render([100, 100, 200, 300, 300, 300, 300], stable_rounds=3)
    # Must have read past the plateau, not stopped at the second 100.
    assert page.calls > 4, f"stopped early at the plateau after {page.calls} checks"


def test_stops_once_height_is_genuinely_stable():
    # A short page must not burn the whole scroll budget.
    page = _render([50] * 40, stable_rounds=3, max_scrolls=40)
    assert page.calls <= 5, f"kept scrolling a static page for {page.calls} checks"


def test_respects_the_scroll_budget_on_an_endless_feed():
    page = _render(list(range(100, 100000, 100)), stable_rounds=3, max_scrolls=6)
    assert page.calls <= 6
