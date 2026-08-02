"""Resolve a prompt into the list of targets to process."""

import logging
import re
from typing import List

from langchain_core.runnables import RunnableConfig

from orchestrator.configuration import AgentConfiguration
from orchestrator.state import Target

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
    from agents.outreach.utils import get_lead_loader

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


async def render_page_text(
    url: str, max_scrolls: int = 40, settle_ms: int = 1500, stable_rounds: int = 3
) -> str:
    """Load a page in a real browser, scroll to the bottom, and return its visible text.

    Plain HTTP fetching and Tavily's crawl both return the pre-render HTML, which for a
    JavaScript-built page is nearly empty of content - verified on a conclave page whose
    ~120-person speaker grid was completely invisible that way, yielding only 3 names
    from sibling pages. Speaker and sponsor grids are also commonly lazy-loaded, so the
    page has to be scrolled before the full list exists in the DOM.

    Scrolling stops only after the height holds steady for `stable_rounds` consecutive
    checks. Stopping at the first unchanged reading looks equivalent but is not: a
    lazy-loading page plateaus while it waits on a network request, so a single slow
    response ends the scroll early. That produced 33 targets from 16.8k chars on a page
    that yields 123 from 22.4k when scrolled to true completion - the same page, the same
    code, differing only by how one fetch happened to be timed.

    Args:
        url: Page to render
        max_scrolls: Upper bound on scroll steps, so an infinite feed cannot hang the run
        settle_ms: Pause after each scroll to let lazy content load
        stable_rounds: Consecutive unchanged height readings required before stopping

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

            previous_height, stable = 0, 0
            for _ in range(max_scrolls):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(settle_ms)
                height = await page.evaluate("document.body.scrollHeight")
                stable = stable + 1 if height == previous_height else 0
                if stable >= stable_rounds:
                    break
                previous_height = height

            text = await page.locator("body").inner_text(timeout=15000)
        finally:
            await browser.close()

    logging.info(f"Rendered {len(text)} chars from {url}")
    return text[:MAX_PAGE_CHARS]


# Phrases an anti-bot interstitial serves in place of the real page. They arrive with
# HTTP 200 and a full HTML body, so nothing downstream looks like an error.
_BLOCK_MARKERS = (
    "just a moment",
    "verify you are human",
    "checking your browser",
    "enable javascript and cookies to continue",
    "attention required!",
    "access denied",
    "ddos protection by",
    "please turn javascript on",
)
# Below this, a "successful" render is really a stub - a challenge page or an empty shell.
_MIN_USEFUL_CHARS = 800


def looks_blocked(text: str) -> bool:
    """Whether a fetched page is an anti-bot interstitial rather than the real content.

    Needed because a Cloudflare challenge returns HTTP 200 with a populated body, so the
    browser render reports success and no fallback ever fires. On a blocked directory
    that produced exactly one extracted "company", named Cloudflare - a wrong answer
    presented as a right one, which is worse than a visible failure.
    """
    # Only the very start counts. An interstitial announces itself in the title and first
    # paragraph, whereas a real page can mention "access denied" anywhere in its body copy
    # - a policy page explaining error messages must not read as a block.
    head = (text or "")[:1200].lower()
    if any(marker in head for marker in _BLOCK_MARKERS):
        return True
    return len((text or "").strip()) < _MIN_USEFUL_CHARS


async def _fetch_via_tavily(url: str) -> str:
    """Fetch page content through Tavily, which retrieves server-side.

    Reaches pages our own browser cannot: Tavily returned 43k characters from a
    Cloudflare-protected directory that blocked local rendering outright.
    """
    import os

    from tavily import AsyncTavilyClient

    client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    try:
        extracted = await client.extract(urls=[url])
        text = "\n".join(r.get("raw_content") or "" for r in extracted.get("results", []))
        if text.strip():
            return text[:MAX_PAGE_CHARS]
    except Exception as e:
        logging.warning(f"Tavily extract failed for {url}: {e}")

    try:
        crawled = await client.crawl(
            url=url, max_depth=1, max_breadth=10,
            instructions="Find all organizations, sponsors, partners and named people with job titles.",
            extract_depth="advanced",
        )
        return "\n".join(
            (p.get("raw_content") or p.get("content") or "") for p in crawled.get("results", [])
        )[:MAX_PAGE_CHARS]
    except Exception as e:
        logging.warning(f"Tavily crawl failed for {url}: {e}")
        return ""


async def _targets_from_page(
    url: str, config: RunnableConfig, request: str = ""
) -> List[Target]:
    """Extract organizations and people from a listing page.

    Rendering plus model-based extraction, rather than per-site regex: every event page
    publishes its speaker grid differently, so patterns tuned to one site silently return
    nothing on the next. A model reads whatever shape the page happens to use.

    Escalates to Tavily when the browser is blocked or returns a stub, then falls back to
    regex extraction if the model call fails, so each layer degrades instead of returning
    a confidently wrong answer.

    Args:
        url: The listing or event page to extract from
        config: Runtime configuration for model settings
        request: What the caller asked for, e.g. "engineering colleges in Dehradun".
            Passed to the model so it returns only matching entries. A "top 10 colleges"
            page is usually one college's marketing page carrying its recruiter logos,
            and returning Google and Adobe alongside the colleges is not useful.

    Returns:
        Targets extracted from the page
    """
    text = ""
    try:
        text = await render_page_text(url)
    except Exception as e:
        logging.warning(f"Browser render failed for {url}: {e}")

    if looks_blocked(text):
        logging.info(f"Browser render blocked or empty for {url}; fetching via Tavily")
        text = await _fetch_via_tavily(url) or text

    if not text.strip():
        return []

    return await _extract_targets(text, url, config, request)


async def _extract_targets(
    text: str, url: str, config: RunnableConfig, request: str = ""
) -> List[Target]:
    """Have a model pull organizations and named people out of rendered page text.

    Args:
        text: Rendered page text
        url: Source page, used as context
        config: Runtime configuration for model settings
        request: What the caller asked for, so the model returns only matching entries

    Returns:
        Extracted targets; falls back to regex extraction if the model call fails
    """
    from agents.outreach.configuration import SalesConfiguration
    from agents.outreach.utils import invoke_llm
    from orchestrator.state import ExtractedTargets

    # Filtering belongs here, where a model is already reading the page, rather than in
    # the caller: a keyword list downstream cannot tell a college from its recruiters
    # without re-deriving what the page already says, and gets it wrong on the next site.
    wanted = (
        f"\n\n# What the user asked for\n{request}\n"
        "Return ONLY entries matching that request. A listing page carries other "
        "organizations - recruiters, sponsors, parent bodies, the site's own brand - "
        "and those are not what was asked for even though they appear on the page. "
        "If nothing on the page matches, return an empty list rather than substituting "
        "whatever else is there."
        if request.strip() else ""
    )

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
                + wanted
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
