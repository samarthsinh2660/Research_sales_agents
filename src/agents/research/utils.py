"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import logging
import os
import re
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from agents.research.configuration import Configuration, SearchAPI
from agents.research.prompts import summarize_webpage_prompt
from agents.research.state import ResearchComplete, Summary

##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events. "
    "For recent news, announcements, or press specifically, set topic='news' instead of "
    "the default 'general' - it returns far more relevant, recent results for that kind of query."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    
    # Step 2: Deduplicate results by URL to avoid processing the same content multiple times
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    # Step 3: Set up the summarization model with configuration
    configurable = Configuration.from_runnable_config(config)

    # Step 3b: Drop low-relevance results before spending model calls summarizing them
    unique_results = filter_by_relevance(unique_results, configurable.relevance_threshold)

    # Character limit to stay within model token limits (configurable)
    max_char_to_include = configurable.max_content_length
    
    # max_retries=0: retrying the primary burns the caller's 60s timeout before the
    # fallback is ever reached.
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    fallback_summarization_model = init_chat_model(
        **get_fallback_model_config(configurable, config, configurable.summarization_model_max_tokens),
        max_retries=2,
    ).with_structured_output(Summary)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"],
        max_retries=0,
    ).with_structured_output(Summary).with_fallbacks([fallback_summarization_model])
    
    # Step 4 & 5: Summarize results one at a time (not in parallel) so a single search
    # step (which can have 7-8 results) doesn't burst past Gemini free tier's
    # 5-requests-per-minute limit on both the primary and fallback model at once.
    summaries = []
    for result in unique_results.values():
        if not result.get("raw_content"):
            summaries.append(None)
        else:
            summary = await summarize_webpage(
                summarization_model,
                result['raw_content'][:max_char_to_include]
            )
            summaries.append(summary)

    # Step 6: Combine results with their summaries
    summarized_results = {
        url: {
            'title': result['title'], 
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(), 
            unique_results.values(), 
            summaries
        )
    }
    
    # Step 7: Format the final output
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."
    
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output

def filter_by_relevance(results: dict, threshold: float) -> dict:
    """Drop results below a Tavily relevance score threshold, keyed by URL.

    Falls back to the unfiltered set if filtering would remove everything, so a
    narrow/niche query never ends up with zero results.

    Args:
        results: Dict of {url: result_dict}, where each result_dict may have a 'score' key
        threshold: Minimum score to keep a result (results with no 'score' key are kept)

    Returns:
        Filtered dict, or the original dict unchanged if filtering would empty it
    """
    relevant_results = {
        url: result for url, result in results.items()
        if result.get("score", 1.0) >= threshold
    }
    return relevant_results or results

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """Execute multiple Tavily search queries asynchronously.
    
    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access
        
    Returns:
        List of search result dictionaries from Tavily API
    """
    # Initialize the Tavily client with API key from config
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]
    
    # Execute all search queries in parallel and return results
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content using AI model with timeout protection.
    
    Args:
        model: The chat model configured for summarization
        webpage_content: Raw webpage content to be summarized
        
    Returns:
        Formatted summary with key excerpts, or original content if summarization fails
    """
    try:
        # Create prompt with current date context
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        # Execute summarization with timeout to prevent hanging
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 60 second timeout for summarization
        )
        
        # Format the summary with structured sections
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except asyncio.TimeoutError:
        # Timeout during summarization - return original content
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # Other errors during summarization - log and return original content
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

##########################
# Website Contact Finder Tool Utils
##########################
WEBSITE_CONTACT_FINDER_DESCRIPTION = (
    "Crawls a specific organization's own website (not a generic search) looking for named "
    "people and contact information - e.g. About Us, Team, Leadership, or Contact pages - and "
    "also their blog content and linked social media profiles (Twitter/X, LinkedIn, YouTube, "
    "Facebook, Instagram). Use this when you already know the organization's official website "
    "URL and need real contacts, recent blog activity, or social presence rather than generic "
    "company facts. Do not use this for generic research questions - use tavily_search for those."
)
@tool(description=WEBSITE_CONTACT_FINDER_DESCRIPTION)
async def website_contact_finder(
    website_url: str,
    config: RunnableConfig = None
) -> str:
    """Crawl an organization's own website for contacts, blog content, and social links.

    Unlike tavily_search's generic query-based search, this targets a known URL directly and
    crawls it, which surfaces far more from About/Team/Contact/Blog pages than a generic web
    search. Raw crawled content is returned without a summarization pass, so exact names/
    emails/URLs aren't lost to lossy summarization.

    Args:
        website_url: The organization's official website URL (e.g. https://example.com)
        config: Runtime configuration for API keys

    Returns:
        Formatted string containing crawled content from contact/blog-relevant pages, with source URLs
    """
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    try:
        crawl_result = await tavily_client.crawl(
            url=website_url,
            max_depth=2,
            max_breadth=10,
            instructions=(
                "Find pages about the team, leadership, founders, management, or contact "
                "information - including named people, their roles, and email addresses or "
                "contact forms. Also find the company's blog (recent post titles/topics) and "
                "any linked social media profiles (Twitter/X, LinkedIn, YouTube, Facebook, "
                "Instagram) - usually linked in the site header, footer, or an About/Contact page."
            ),
            extract_depth="advanced",
        )
    except Exception as e:
        crawl_result = {}
        logging.warning(f"Crawl failed for {website_url}, falling back to page source: {e}")

    results = crawl_result.get("results", [])
    published = await _emails_in_page_source(website_url)

    if not results and not published:
        return f"No contact-relevant pages found by crawling {website_url}."

    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = configurable.max_content_length

    formatted_output = f"Crawled pages from {website_url}:\n\n"
    if published:
        # Listed first, and separately, because the crawl demonstrably loses addresses that
        # are present in the raw page: on one site it returned ~6k characters and none of
        # them were the address sitting in the homepage source.
        formatted_output += (
            "--- ADDRESSES READ DIRECTLY FROM THE PAGE SOURCE ---\n"
            "Published verbatim on the site, safe to quote exactly:\n"
            + "\n".join(f"- {e}" for e in published)
            + "\n\n" + "-" * 80 + "\n"
        )
    for i, result in enumerate(results):
        page_url = result.get("url", "")
        content = result.get("raw_content") or result.get("content", "")
        formatted_output += f"\n\n--- PAGE {i+1}: {page_url} ---\n\n{content[:max_char_to_include]}\n\n" + "-" * 80 + "\n"

    return formatted_output


# Asset filenames match the email shape ("logo@2x.png"), and template placeholders ship on
# real sites - one page carries "you@example.com" in its contact form beside a real address.
_EMAIL_IN_SOURCE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_EMAIL_REJECT = re.compile(
    r"(\.(png|jpe?g|gif|svg|webp|css|js)$|^[0-9.]+@"
    r"|@(example|sentry|test|domain|email|yourdomain)\.)",
    re.IGNORECASE,
)


async def _emails_in_page_source(url: str) -> list[str]:
    """Read published addresses straight out of a page's HTML. No model, no crawler.

    Complements the crawl rather than replacing it: the crawler is better at finding which
    pages matter, but its extracted text drops addresses that sit in markup. Parsing can
    only copy an address or miss it, never invent one, so this adds no fabrication risk.
    """
    try:
        async with httpx.AsyncClient(
            timeout=12.0, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PACE-outreach-research/1.0)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except Exception as e:
        logging.debug(f"Could not read page source for {url}: {e}")
        return []

    found: dict[str, None] = {}
    for raw in _EMAIL_IN_SOURCE.findall(html):
        email = raw.strip().strip(".,;:").lower()
        if not _EMAIL_REJECT.search(email):
            found.setdefault(email, None)
    if found:
        logging.info(f"Page source for {url} published {len(found)} address(es)")
    return list(found)

##########################
# LinkedIn Search Tool Utils
##########################
LINKEDIN_SESSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".linkedin_session.json")

LINKEDIN_SEARCH_DESCRIPTION = (
    "Looks up a specific LinkedIn profile or company page URL and returns structured data "
    "(name, headline, experience, education for a person; overview, industry, size for a "
    "company). Requires a real LinkedIn URL - use tavily_search first to find the URL if you "
    "don't already have one, then pass it here. Each call is a real LinkedIn page visit under "
    "a dedicated scraping account, so use it deliberately for a specific known URL, not as a "
    "general search tool."
)

async def _linkedin_login(page, email: str, password: str, timeout: int = 30000) -> None:
    """Log into LinkedIn directly, bypassing linkedin_scraper's login_with_credentials().

    That function's '#username' selector is stale against LinkedIn's current login page
    markup. Worse: LinkedIn renders the login form TWICE in the DOM (a plain variant with
    autocomplete="username", and a hidden WebAuthn/passkey variant with
    autocomplete="username webauthn") - an exact-match selector can silently lock onto
    whichever copy isn't actually visible. Using Playwright's :visible filter to always
    target whichever copy is really on screen. Everything else in the library
    (is_logged_in, PersonScraper, CompanyScraper, session save/load) still works fine -
    only the login form selector needed a workaround.
    """
    from linkedin_scraper import AuthenticationError, is_logged_in

    await page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")

    email_field = page.locator('input[autocomplete*="username"]:visible').first
    password_field = page.locator('input[autocomplete="current-password"]:visible').first

    try:
        await email_field.wait_for(state="visible", timeout=timeout)
    except Exception:
        raise AuthenticationError("Login form not found (LinkedIn may have changed their page structure again).")

    await email_field.fill(email)
    await password_field.fill(password)

    # get_by_role(exact=True) still matches duplicate hidden buttons (same DOM-duplication
    # issue as the input fields) - "Sign in with Apple" would also match a plain substring
    # filter, so check visibility explicitly rather than just taking .first
    sign_in_buttons = page.get_by_role("button", name="Sign in", exact=True)
    clicked = False
    for i in range(await sign_in_buttons.count()):
        btn = sign_in_buttons.nth(i)
        if await btn.is_visible():
            await btn.click()
            clicked = True
            break
    if not clicked:
        raise AuthenticationError("Sign in button not found or not visible.")

    try:
        await page.wait_for_url(
            lambda url: "feed" in url or "checkpoint" in url or "authwall" in url,
            timeout=timeout
        )
    except Exception:
        if "login" in page.url:
            raise AuthenticationError("Login failed - page did not navigate after clicking sign in. Check credentials.")

    current_url = page.url
    if "checkpoint" in current_url or "challenge" in current_url:
        raise AuthenticationError(
            f"LinkedIn security checkpoint detected - manual verification needed once, "
            f"then session persistence avoids repeating this. Current URL: {current_url}"
        )
    if "authwall" in current_url:
        raise AuthenticationError(f"Authentication wall encountered. Current URL: {current_url}")

    for _ in range(10):
        if await is_logged_in(page):
            return
        await page.wait_for_timeout(500)

_navigate_and_wait_patched = False

def _patch_navigate_and_wait() -> None:
    """Patch BaseScraper.navigate_and_wait to settle before checking for rate limits.

    The library calls check_rate_limit() immediately after page.goto(..., wait_until=
    "domcontentloaded") - which fires before the SPA has actually hydrated real profile
    content. Verified directly: this causes a false-positive "Rate limit message detected
    on page" on the very first navigation, while a manual re-check with a short settle
    delay succeeds cleanly every time. Idempotent - safe to call more than once.
    """
    global _navigate_and_wait_patched
    if _navigate_and_wait_patched:
        return

    from linkedin_scraper.scrapers.base import BaseScraper

    async def patched_navigate_and_wait(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 60000) -> None:
        await self.page.goto(url, wait_until=wait_until, timeout=timeout)
        await self.page.wait_for_timeout(2500)  # let the SPA hydrate before checking content
        await self.check_rate_limit()

    BaseScraper.navigate_and_wait = patched_navigate_and_wait
    _navigate_and_wait_patched = True

_person_extraction_patched = False

async def _get_main_content_text(page, timeout: int = 10000) -> str:
    """Get the main content column's text, however LinkedIn happens to render it.

    data-testid="lazy-column" is present on the main profile page but was observed absent
    on some /details/* sub-pages (verified directly) - falls back to the full page body,
    which contains the same structured text either way.
    """
    column = page.locator('[data-testid="lazy-column"]').first
    if await column.count() > 0:
        return await column.inner_text(timeout=timeout)
    return await page.locator("body").inner_text(timeout=timeout)

_DATE_RANGE_PATTERN = re.compile(r"\b(19|20)\d{2}\b.*(present|\b(19|20)\d{2}\b)", re.IGNORECASE)

def _patch_person_extraction() -> None:
    """Patch PersonScraper's field extraction (name, location, about, experience, education).

    linkedin_scraper's selectors for these are stale against LinkedIn's current markup -
    verified directly: LinkedIn no longer uses <h1> for the name or <h2> "Experience"/
    "Education" headings inline on the main profile page (moved to dedicated /details/*
    sub-pages), and profile-card data attributes it relies on for About no longer exist.
    LinkedIn's visual CSS classes are build-hashed and churn every deploy (e.g. "_036a6bd2
    ffb20cf8..."), so matching those directly is a losing battle - instead this extracts
    from data-testid="lazy-column" (a stable test-automation hook) and parses fields
    positionally/structurally, which is far more resilient to LinkedIn's routine CSS-in-JS
    class churn than exact class selectors.

    Experience entries are anchored on date-range lines (e.g. "Feb 2014 - Present · 12 yrs
    6 mos") via regex, since that's the one line every entry reliably has - title/company
    are the two lines before it, and any line(s) before the next entry's title are treated
    as location. Education entries don't reliably have dates (some show a degree line
    instead - verified directly on a real profile), so those are parsed as simple pairs of
    (institution, degree-or-dates) lines instead.

    Interests/accomplishments/contacts extraction remains unfixed and will keep returning
    empty - lower value for outreach personalization than who someone is, their role,
    location, and work history, and each would need its own reverse-engineering pass.
    """
    global _person_extraction_patched
    if _person_extraction_patched:
        return

    from linkedin_scraper import PersonScraper
    from linkedin_scraper.models.person import Education, Experience

    async def patched_get_name_and_location(self):
        try:
            text = await _get_main_content_text(self.page)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            name = lines[0] if lines else "Unknown"
            location = lines[2] if len(lines) > 2 and "," in lines[2] else None
            return name, location
        except Exception as e:
            logging.warning(f"Error getting name/location: {e}")
            return "Unknown", None

    async def patched_get_about(self):
        try:
            text = await _get_main_content_text(self.page)
            if "\nAbout\n" not in text:
                return None
            after_about = text.split("\nAbout\n", 1)[1]
            # About runs until the next major section heading
            for stop_marker in ["\nFeatured\n", "\nActivity\n", "\nExperience\n"]:
                if stop_marker in after_about:
                    after_about = after_about.split(stop_marker, 1)[0]
            return after_about.strip() or None
        except Exception as e:
            logging.debug(f"Error getting about section: {e}")
            return None

    async def patched_get_experiences(self, base_url: str):
        try:
            exp_url = base_url.rstrip("/") + "/details/experience/"
            await self.navigate_and_wait(exp_url)
            text = await _get_main_content_text(self.page)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if "Experience" in lines:
                lines = lines[lines.index("Experience") + 1:]

            date_indices = [i for i, line in enumerate(lines) if _DATE_RANGE_PATTERN.search(line)]
            experiences = []
            for idx, date_idx in enumerate(date_indices):
                if date_idx < 2:
                    continue
                title = lines[date_idx - 2]
                company = lines[date_idx - 1]
                duration = lines[date_idx]
                next_entry_start = date_indices[idx + 1] - 2 if idx + 1 < len(date_indices) else min(date_idx + 3, len(lines))
                location_lines = lines[date_idx + 1:next_entry_start]
                location = location_lines[0] if location_lines else None
                experiences.append(Experience(
                    position_title=title,
                    institution_name=company,
                    duration=duration,
                    location=location,
                    from_date="",
                    to_date="",
                    description="",
                ))
            return experiences
        except Exception as e:
            logging.warning(f"Error getting experiences: {e}")
            return []

    async def patched_get_educations(self, base_url: str):
        try:
            edu_url = base_url.rstrip("/") + "/details/education/"
            await self.navigate_and_wait(edu_url)
            text = await _get_main_content_text(self.page)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if "Education" in lines:
                lines = lines[lines.index("Education") + 1:]
            # Stop at the next unrelated section (sidebar recommendations, footer, etc.)
            for stop_marker in ["More profiles for you", "About", "Accessibility"]:
                if stop_marker in lines:
                    lines = lines[:lines.index(stop_marker)]

            educations = []
            for i in range(0, len(lines) - 1, 2):
                educations.append(Education(
                    institution_name=lines[i],
                    degree=lines[i + 1],
                    from_date="",
                    to_date="",
                    description="",
                ))
            return educations
        except Exception as e:
            logging.warning(f"Error getting educations: {e}")
            return []

    PersonScraper._get_name_and_location = patched_get_name_and_location
    PersonScraper._get_about = patched_get_about
    PersonScraper._get_experiences = patched_get_experiences
    PersonScraper._get_educations = patched_get_educations
    _person_extraction_patched = True

@tool(description=LINKEDIN_SEARCH_DESCRIPTION)
async def linkedin_search(linkedin_url: str) -> str:
    """Scrape a LinkedIn profile or company page for structured data.

    Reuses a saved login session across calls (session.json) so most calls don't need a fresh
    login - repeated logins are themselves a signal LinkedIn's detection systems watch for.

    Args:
        linkedin_url: Full LinkedIn URL, e.g. https://linkedin.com/in/username or
            https://linkedin.com/company/company-name

    Returns:
        Formatted string of the scraped data, or a clear error message if scraping failed
    """
    from linkedin_scraper import (
        AuthenticationError,
        BrowserManager,
        CompanyScraper,
        LinkedInScraperException,
        PersonScraper,
        ProfileNotFoundError,
        RateLimitError,
        load_credentials_from_env,
    )

    email, password = load_credentials_from_env()
    if not email or not password:
        return (
            "LinkedIn credentials not configured (LINKEDIN_EMAIL/LINKEDIN_PASSWORD env vars "
            "missing) - skipping LinkedIn lookup. Use tavily_search or website_contact_finder instead."
        )

    try:
        async with BrowserManager(headless=True) as browser:
            session_valid = False
            if os.path.exists(LINKEDIN_SESSION_PATH):
                try:
                    # load_session() closes the current page/context and creates new ones,
                    # so browser.page must be re-read after this call, not cached beforehand
                    await browser.load_session(LINKEDIN_SESSION_PATH)
                    session_valid = browser.is_authenticated
                except Exception:
                    session_valid = False

            page = browser.page

            if not session_valid:
                await _linkedin_login(page, email, password)
                await browser.save_session(LINKEDIN_SESSION_PATH)

            _patch_navigate_and_wait()
            _patch_person_extraction()
            scraper = CompanyScraper(page) if "/company/" in linkedin_url else PersonScraper(page)
            try:
                result = await scraper.scrape(linkedin_url)
            except RateLimitError as e:
                # The library checks for rate-limit phrases immediately after
                # domcontentloaded, before the page has actually hydrated real content -
                # a transient loading-state false positive, verified directly (manual
                # re-checks with a short settle delay succeed cleanly). Only retry this
                # specific false-positive-prone check, not a real checkpoint/CAPTCHA block.
                if "Rate limit message detected on page" not in str(e):
                    raise
                await page.wait_for_timeout(3000)
                result = await scraper.scrape(linkedin_url)
            return f"LinkedIn data for {linkedin_url}:\n\n{result.model_dump_json(indent=2)}"

    except AuthenticationError as e:
        return f"LinkedIn login failed: {e}"
    except RateLimitError as e:
        return f"LinkedIn rate limit hit, back off before retrying: {e}"
    except ProfileNotFoundError as e:
        return f"LinkedIn profile/company not found at {linkedin_url}: {e}"
    except LinkedInScraperException as e:
        return f"LinkedIn scraping error: {e}"

##########################
# YouTube Search Tool Utils
##########################
YOUTUBE_SEARCH_DESCRIPTION = (
    "Searches YouTube for a person's or organization's channel and returns the channel's "
    "description, subscriber count, and their most recent videos (titles, descriptions, "
    "publish dates). Uses the official YouTube Data API - no scraping. Useful for learning "
    "what someone or some organization publicly talks about and is currently focused on, to "
    "personalize outreach with specific, real, recent detail rather than generic claims."
)
@tool(description=YOUTUBE_SEARCH_DESCRIPTION)
async def youtube_search(query: str, max_videos: int = 5) -> str:
    """Search YouTube for a channel and return its info plus recent videos.

    Args:
        query: Name of the person, company, or channel to search for
        max_videos: Max number of recent videos to include (default 5)

    Returns:
        Formatted string with channel info and recent video titles/descriptions
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return "YouTube API key not configured (YOUTUBE_API_KEY env var missing) - skipping YouTube lookup."

    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: find candidate channels. YouTube's default search relevance ranking
            # does NOT prioritize the most legitimate/largest channel - a small unrelated
            # channel with matching keywords in its title can outrank the real one
            # (verified directly: searching "Satya Nadella" returned a 1-subscriber student
            # project channel first). Fetch several candidates and pick by subscriber count
            # instead of trusting result order.
            async with session.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={"part": "snippet", "q": query, "type": "channel", "maxResults": 5, "key": api_key}
            ) as resp:
                search_data = await resp.json()

            candidates = search_data.get("items", [])
            if not candidates:
                return f"No YouTube channel found for '{query}'."

            candidate_ids = [c["id"]["channelId"] for c in candidates]

            # Step 2: get stats for all candidates in one call, pick the most-subscribed
            async with session.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={"part": "snippet,statistics", "id": ",".join(candidate_ids), "key": api_key}
            ) as resp:
                channels_data = await resp.json()

            channels = channels_data.get("items", [])
            if not channels:
                return f"Channel(s) found for '{query}' but details unavailable."

            def _subscriber_count(ch):
                try:
                    return int(ch.get("statistics", {}).get("subscriberCount", 0))
                except (ValueError, TypeError):
                    return 0

            channel = max(channels, key=_subscriber_count)
            channel_id = channel["id"]
            channel_title = channel.get("snippet", {}).get("title", query)
            description = channel.get("snippet", {}).get("description", "")
            stats = channel.get("statistics", {})
            subscriber_count = stats.get("subscriberCount", "hidden")
            video_count = stats.get("videoCount", "0")

            # Step 3: get recent videos, newest first
            async with session.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet", "channelId": channel_id, "type": "video",
                    "order": "date", "maxResults": max_videos, "key": api_key
                }
            ) as resp:
                videos_data = await resp.json()

            videos = videos_data.get("items", [])

        formatted_output = f"YouTube channel: {channel_title}\n"
        formatted_output += f"Subscribers: {subscriber_count} | Total videos: {video_count}\n"
        formatted_output += f"Channel description: {description}\n\n"
        formatted_output += "Recent videos:\n"
        if not videos:
            formatted_output += "(none found)\n"
        for video in videos:
            title = video["snippet"]["title"]
            published = video["snippet"]["publishedAt"]
            video_desc = video["snippet"].get("description", "")[:200]
            formatted_output += f"- {title} ({published}): {video_desc}\n"

        return formatted_output

    except Exception as e:
        return f"YouTube lookup failed: {e}"

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP Utils
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """Exchange Supabase token for MCP access token using OAuth token exchange.
    
    Args:
        supabase_token: Valid Supabase authentication token
        base_mcp_url: Base URL of the MCP server
        
    Returns:
        Token data dictionary if successful, None if failed
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
                    
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """Retrieve stored authentication tokens with expiration validation.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        
    Returns:
        Token dictionary if valid and not expired, None otherwise
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # Retrieve stored tokens
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # Check token expiration
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # Token expired, clean up and return None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """Store authentication tokens in the configuration store.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        tokens: Token dictionary to store
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # Store the tokens
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """Fetch and refresh MCP tokens, obtaining new ones if needed.
    
    Args:
        config: Runtime configuration with authentication details
        
    Returns:
        Valid token dictionary, or None if unable to obtain tokens
    """
    # Try to get existing valid tokens first
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # Extract Supabase token for new token exchange
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # Extract MCP configuration
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # Exchange Supabase token for MCP tokens
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # Store the new tokens and return them
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap MCP tool with comprehensive authentication and error handling.
    
    Args:
        tool: The MCP structured tool to wrap
        
    Returns:
        Enhanced tool with authentication error handling
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """Enhanced coroutine with MCP error handling and user-friendly messages."""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """Recursively search for MCP errors in exception chains."""
            if isinstance(exc, McpError):
                return exc
            
            # Handle ExceptionGroup (Python 3.11+) by checking attributes
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # Execute the original tool functionality
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # Search for MCP-specific errors in the exception chain
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # Not an MCP error, re-raise the original exception
                raise original_error
            
            # Handle MCP-specific error cases
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # Check for authentication/interaction required error
            if error_code == -32003:  # Interaction required error code
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # Extract user-friendly message if available
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # Append URL if provided for user reference
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # For other MCP errors, re-raise the original
            raise original_error
    
    # Replace the tool's coroutine with our enhanced version
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load and configure MCP (Model Context Protocol) tools with authentication.
    
    Args:
        config: Runtime configuration containing MCP server details
        existing_tool_names: Set of tool names already in use to avoid conflicts
        
    Returns:
        List of configured MCP tools ready for use
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: Handle authentication if required
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # Step 2: Validate configuration requirements
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # Step 3: Set up MCP server connection
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # Configure authentication headers if tokens are available
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: When Multi-MCP Server support is merged in OAP, update this code
    
    # Step 4: Load tools from MCP server
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # If MCP server connection fails, return empty list
        return []
    
    # Step 5: Filter and configure tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # Skip tools with conflicting names
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # Only include tools specified in configuration
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # Wrap tool with authentication handling and add to list
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


##########################
# Tool Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.
    
    Args:
        search_api: The search API provider to use (Anthropic, OpenAI, Tavily, or None)
        
    Returns:
        List of configured search tool objects for the specified provider
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic's native web search with usage limits
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI's web search preview functionality
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # Configure Tavily search tool with metadata
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]
        
    elif search_api == SearchAPI.NONE:
        # No search functionality configured
        return []
        
    # Default fallback for unknown search API types
    return []
    
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research, search, and MCP tools.
    
    Args:
        config: Runtime configuration specifying search API and MCP settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Add configured search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)

    # Website contact finder needs Tavily's crawl() specifically, so only add it
    # alongside Tavily search (not for OpenAI/Anthropic native web search or no-search config)
    if search_api == SearchAPI.TAVILY:
        tools.append(website_contact_finder)

    # Only expose LinkedIn search once a dedicated scraping account is actually configured -
    # otherwise it's a tool call guaranteed to fail before an account even exists
    if os.getenv("LINKEDIN_EMAIL") and os.getenv("LINKEDIN_PASSWORD"):
        tools.append(linkedin_search)

    if os.getenv("YOUTUBE_API_KEY"):
        tools.append(youtube_search)

    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # Add MCP tools if configured
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# Model Provider Native Websearch Utils
##########################

def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used in the response.
    
    Args:
        response: The response object from Anthropic's API
        
    Returns:
        True if web search was called, False otherwise
    """
    try:
        # Navigate through the response metadata structure
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # Check for server-side tool usage information
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # Look for web search request count
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # Return True if any web search requests were made
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # Handle cases where response structure is unexpected
        return False

def openai_websearch_called(response):
    """Detect if OpenAI's web search functionality was used in the response.
    
    Args:
        response: The response object from OpenAI's API
        
    Returns:
        True if web search was called, False otherwise
    """
    # Check for tool outputs in the response metadata
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # Look for web search calls in the tool outputs
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token Limit Exceeded Utils
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.
    
    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection
        
    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()
    
    # Step 1: Determine provider from model name if available
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # Step 2: Check provider-specific token limit patterns
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    # Step 3: If provider unknown, check all providers
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an OpenAI exception
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # Check for typical OpenAI token limit error types
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # Look for token-related keywords in error message
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # Check for specific OpenAI error codes
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an Anthropic exception
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # Check for Anthropic-specific error patterns
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic uses specific error messages for token limits
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is a Google/Gemini exception
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # Check for Google-specific resource exhaustion errors
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # Check for specific Google API resource exhaustion patterns
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """Look up the token limit for a specific model.
    
    Args:
        model_string: The model identifier string to look up
        
    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # Model not found in lookup table
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.
    
    This is useful for handling token limit exceeded errors by removing recent context.
    
    Args:
        messages: List of message objects to truncate
        
    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]
    
    # No AI messages found, return original list
    return messages

##########################
# Citation Verification Utils
##########################

def extract_answer_text(content) -> str:
    """Extract the final answer text from a model response's content.

    Some providers (e.g. Gemini in thinking mode) return content as a list of typed
    blocks (a 'thinking' block plus a 'text' block) instead of a plain string. This
    pulls out just the actual answer, skipping any reasoning/thinking blocks.

    Args:
        content: A message's .content, either a plain string or a list of blocks

    Returns:
        The plain-text answer
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_blocks = [
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if text_blocks:
            return "".join(text_blocks)
        # No explicit 'text' blocks found (e.g. a provider without typed blocks) -
        # join whatever text content exists rather than falling through to the raw list
        return "".join(
            block if isinstance(block, str) else str(block.get("text", block))
            for block in content
        )
    return str(content)

def _normalize_url(url: str) -> str:
    """Reduce a URL to a comparable form.

    Both sides must be normalized identically. Findings carry URLs inside markdown links
    and sentences, so they arrive with trailing `)`, `,` or `.` attached, while a cited
    URL does not - comparing them raw drops every real citation.
    """
    return url.rstrip(').,;:\'"]>').rstrip('/').lower()


def verify_citations(report_content: str, findings: str) -> str:
    """Strip citation lines from a report's Sources section whose URL doesn't
    actually appear in the research findings, catching writer-model hallucinated
    citations without a second model call.

    Args:
        report_content: The generated final report, including its Sources section
        findings: The raw research findings the report was generated from

    Returns:
        The report with any unverifiable citation lines removed
    """
    findings_urls = {_normalize_url(u) for u in re.findall(r'https?://\S+', findings)}
    lines = report_content.split("\n")
    verified_lines = []
    total_citations = 0
    dropped_citations = 0
    for line in lines:
        match = re.match(r"^\s*\[\d+\].*?(https?://\S+)", line)
        if match:
            total_citations += 1
            if _normalize_url(match.group(1)) not in findings_urls:
                dropped_citations += 1
                continue
        verified_lines.append(line)

    # A mass-strip means the comparison is broken, not that the writer invented every
    # source - a report cannot cite ten hallucinated URLs and no real one.
    if total_citations > 0 and dropped_citations / total_citations > 0.5:
        logging.warning(
            f"verify_citations dropped {dropped_citations}/{total_citations} citations "
            "(>50%) - suspect URL matching rather than hallucination; the report will "
            "ship with most of its Sources section missing."
        )

    return "\n".join(verified_lines)

##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_fallback_model_config(configurable, config: RunnableConfig, max_tokens: int):
    """Build the .with_config() dict for the fallback model, reusing the primary role's token limit."""
    return {
        "model": configurable.fallback_model,
        "max_tokens": max_tokens,
        "api_key": get_api_key_for_model(configurable.fallback_model, config),
        "tags": ["langsmith:nostream"]
    }

def get_tavily_api_key(config: RunnableConfig):
    """Get Tavily API key from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
