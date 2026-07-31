"""Resolve a prompt into the list of targets to process."""

import logging
import re
from typing import List

from langchain_core.runnables import RunnableConfig

from agent.configuration import AgentConfiguration
from agent.state import Target

##########################
# Input Parsing Utils
##########################

_SHEET_ID_PATTERN = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")
_URL_PATTERN = re.compile(r"https?://\S+")


def extract_sheet_id(text: str) -> str | None:
    """Extract a Google Sheets id from a pasted URL, or None if there isn't one."""
    match = _SHEET_ID_PATTERN.search(text or "")
    return match.group(1) if match else None


def parse_inline_list(text: str) -> List[str]:
    """Split prompt text into candidate names on commas and newlines.

    A single name yields a one-item list, which is what makes the single-target case
    take exactly the same code path as a batch.
    """
    stripped = re.sub(r"^\s*research\s+", "", (text or "").strip(), flags=re.IGNORECASE)
    parts = re.split(r"[,\n]", stripped)
    return [p.strip() for p in parts if p.strip()]


def _first_url(text: str) -> str | None:
    """First http(s) URL in the text, or None."""
    match = _URL_PATTERN.search(text or "")
    return match.group(0) if match else None

##########################
# Target Resolution
##########################

async def resolve_targets(prompt: str, config: RunnableConfig) -> List[Target]:
    """Turn the prompt into targets, from whichever input shape it carries.

    Order matters: a sheet link is checked before a generic URL, and both before inline
    names, because a pasted URL would otherwise be parsed as a company name.

    Args:
        prompt: The user's prompt naming the subject(s)
        config: Runtime configuration, used for the target cap and sheet loading

    Returns:
        List of targets to process

    Raises:
        ValueError: if nothing resolvable is found, or the cap is exceeded
    """
    configurable = AgentConfiguration.from_runnable_config(config)

    sheet_id = extract_sheet_id(prompt)
    page_url = None if sheet_id else _first_url(prompt)

    if sheet_id:
        targets = _targets_from_sheet(sheet_id, config)
    elif page_url:
        targets = await _targets_from_page(page_url, config)
    else:
        targets = [Target(name=n, source="inline") for n in parse_inline_list(prompt)]

    if not targets:
        raise ValueError(
            "No targets could be resolved from the prompt. Name a company, paste a "
            "Google Sheet link, or list several names separated by commas."
        )
    if len(targets) > configurable.max_targets:
        raise ValueError(
            f"Resolved {len(targets)} targets, above the max_targets limit of "
            f"{configurable.max_targets}. Raise the limit or narrow the input."
        )

    logging.info(f"Resolved {len(targets)} target(s) from source '{targets[0].source}'")
    return targets


def _targets_from_sheet(sheet_id: str, config: RunnableConfig) -> List[Target]:
    """Load targets from a Google Sheet, preserving the row id for CRM write-back."""
    from sales_outreach.utils import get_lead_loader

    merged = {**config, "configurable": {**config.get("configurable", {}), "sheet_id": sheet_id}}
    records = get_lead_loader(merged).fetch_records()
    return [
        Target(
            name=(r.get("Company Name") or f'{r.get("First Name", "")} {r.get("Last Name", "")}').strip(),
            website=r.get("Company Website") or None,
            email=r.get("Email") or None,
            source="sheet",
            crm_row_id=r.get("id"),
        )
        for r in records
    ]


##########################
# Listing Page Extraction
##########################

# Matches "Image 12: Adobe" - the alt text is the sponsor name. Bare "Image 9" has no
# alt text and is deliberately not matched: those names are unrecoverable from text.
# The capture is lazy and stops at the next "Image N:" because crawled markup puts
# several logos on one line ("Image 12: RedisLabs Image 13: Exotel"), which a greedy
# match swallows into a single bogus name.
_LOGO_ALT_PATTERN = re.compile(r"Image\s+\d+:\s*(.+?)(?=\s*Image\s+\d+:|$)", re.MULTILINE)
_VALID_SPONSOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9&.\- ]{1,39}$")
# Matches a markdown heading name followed by a "Title, Company" line.
_PERSON_PATTERN = re.compile(
    r"^#{2,4}\s+([A-Z][A-Za-z.\- ]{2,40})\s*$\n+^([^#\n]+,[^#\n]+)$", re.MULTILINE
)


def parse_listing_content(text: str) -> List[Target]:
    """Extract organizations and named people from crawled listing-page content.

    Sponsor names come from logo alt text, which is frequently absent - extraction is
    partial by nature and callers must not assume a complete list. Named people extract
    far more reliably and already carry title and employer, so they skip the separate
    contact-finding step later.

    Args:
        text: Crawled or searched page content

    Returns:
        Targets found, deduplicated by name (case-insensitive)
    """
    targets: List[Target] = []
    seen: set[str] = set()

    for name, role_org in _PERSON_PATTERN.findall(text or ""):
        clean = name.strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            targets.append(Target(name=clean, source="page", context=role_org.strip()))

    for sponsor in _LOGO_ALT_PATTERN.findall(text or ""):
        clean = sponsor.strip()
        if not _VALID_SPONSOR_NAME.match(clean) or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        targets.append(
            Target(name=clean, source="page", context="Sponsor/partner listed on the source page")
        )

    return targets


MAX_PAGE_CHARS = 60000


async def render_page_text(url: str, max_scrolls: int = 25, settle_ms: int = 700) -> str:
    """Load a page in a real browser, scroll to the bottom, and return its visible text.

    Plain HTTP fetching and Tavily's crawl both return the pre-render HTML, which for a
    JavaScript-built page is nearly empty of content - verified on a conclave page whose
    ~120-person speaker grid was completely invisible that way, yielding only 3 names
    from sibling pages. Speaker and sponsor grids are also commonly lazy-loaded, so the
    page has to be scrolled before the full list exists in the DOM.

    Scrolling stops early once the page height stops growing, so a short page costs a
    couple of iterations rather than the full budget.

    Args:
        url: Page to render
        max_scrolls: Upper bound on scroll steps, so an infinite feed cannot hang the run
        settle_ms: Pause after each scroll to let lazy content load

    Returns:
        The rendered page text, truncated to MAX_PAGE_CHARS
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(settle_ms)

            previous_height = 0
            for _ in range(max_scrolls):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(settle_ms)
                height = await page.evaluate("document.body.scrollHeight")
                if height == previous_height:
                    break
                previous_height = height

            text = await page.locator("body").inner_text(timeout=15000)
        finally:
            await browser.close()

    return text[:MAX_PAGE_CHARS]


async def _targets_from_page(url: str, config: RunnableConfig) -> List[Target]:
    """Extract organizations and people from a listing page.

    Rendering plus model-based extraction, rather than per-site regex: every event page
    publishes its speaker grid differently, so patterns tuned to one site silently return
    nothing on the next. A model reads whatever shape the page happens to use.

    Falls back to Tavily crawl if the browser is unavailable, then to regex extraction if
    the model call fails, so a failure at any layer degrades instead of returning nothing.

    Args:
        url: The listing or event page to extract from
        config: Runtime configuration for model settings

    Returns:
        Targets extracted from the page
    """
    text = ""
    try:
        text = await render_page_text(url)
        logging.info(f"Rendered {len(text)} chars from {url}")
    except Exception as e:
        logging.warning(f"Browser render failed for {url}, falling back to crawl: {e}")

    if not text.strip():
        import os

        from tavily import AsyncTavilyClient

        try:
            crawled = await AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY")).crawl(
                url=url, max_depth=1, max_breadth=10,
                instructions="Find all sponsors, partners and their tiers, and all speakers with job titles and companies.",
                extract_depth="advanced",
            )
            text = "\n".join(
                (p.get("raw_content") or p.get("content") or "") for p in crawled.get("results", [])
            )[:MAX_PAGE_CHARS]
        except Exception as e:
            logging.warning(f"Crawl fallback also failed for {url}: {e}")
            return []

    return await _extract_targets(text, url, config)


async def _extract_targets(text: str, url: str, config: RunnableConfig) -> List[Target]:
    """Have a model pull organizations and named people out of rendered page text.

    Args:
        text: Rendered page text
        url: Source page, used as context
        config: Runtime configuration for model settings

    Returns:
        Extracted targets; falls back to regex extraction if the model call fails
    """
    from agent.state import ExtractedTargets
    from sales_outreach.configuration import SalesConfiguration
    from sales_outreach.utils import invoke_llm

    try:
        extracted = await invoke_llm(
            system_prompt=(
                "You extract outreach targets from an event or listing page.\n\n"
                "Return two kinds of entry:\n"
                "- ORGANIZATIONS: sponsors, partners, exhibitors. Use the company name only.\n"
                "- PEOPLE: named speakers, guests, panelists. Give their name, and put their "
                "job title and employer in `context`.\n\n"
                "Ignore navigation labels, section headings ('Special Guests', 'Partner "
                "Speakers'), the hosting site's own brand, archive or edition titles, "
                "generic words, and anything that is not a real organization or person. "
                "Copy names verbatim from the page; never invent or complete a name."
            ),
            user_message=f"Source page: {url}\n\nPage text:\n{text}",
            model_name=SalesConfiguration.from_runnable_config(config).research_sufficiency_model,
            config=config,
            response_format=ExtractedTargets,
        )
    except Exception as e:
        logging.warning(f"Model extraction failed for {url}, falling back to regex: {e}")
        return parse_listing_content(text)

    targets: List[Target] = []
    seen: set[str] = set()
    for item in extracted.targets:
        clean = item.name.strip()
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        targets.append(Target(name=clean, source="page", context=item.context or None))

    logging.info(f"Extracted {len(targets)} targets from {url}")
    return targets
