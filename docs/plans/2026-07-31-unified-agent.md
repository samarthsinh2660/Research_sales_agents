# Unified Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One graph and one entry point where the user names a subject in the prompt and picks how deep to go (research → qualify → draft → send) in config, working over one target or many from four input sources.

**Architecture:** A thin `src/agent/` orchestrator package holds target resolution, intent gates, and the per-target loop. It nests `deep_researcher` as a **native subgraph node** (verified working) and imports the existing sales nodes as a library. Research and sales packages are unchanged and stay independently runnable.

**Tech Stack:** Python 3.12, LangGraph, LangChain, Pydantic v2, pytest, ruff.

**Spec:** `docs/unified-agent-design.md`

## Global Constraints

Copied from the existing codebase conventions — every task must follow all of these.

- Nodes are module-level `async def name(state, config: RunnableConfig) -> Command[Literal[...]]`. No node classes.
- Routing is `return Command(goto=..., update={...})`. **Zero `add_conditional_edges`.** Only `START` is a declared edge.
- Config is read as the first statement of a node: `configurable = AgentConfiguration.from_runnable_config(config)`.
- `logging.info/warning/debug` with f-strings. **No `print()`, no colorama.**
- Google-style docstrings with `Args:`/`Returns:` and a rationale paragraph explaining *why* the node exists where it does.
- Absolute imports only (`from agent.x import y`). `##########################` (26 hashes) section fences in utils modules, `###################` (19) in state modules.
- Pydantic config fields use `Field(default=..., metadata={"x_oap_ui_config": {...}})` so they render in Studio.
- New config field names must not collide with `Configuration` or `SalesConfiguration` field names — `from_runnable_config` resolves every field from `os.environ[FIELD.upper()]`, so a shared name means a shared env var.
- Verify command: `python -m pytest tests/unit/ src/sales_outreach/tests/ src/agent/tests/ -q`
- Lint command: `python -m ruff check src/agent/`
- **Never enable real email sending in tests.** Gmail drafts only.

## Verified Technical Findings

These were tested before planning; build on them rather than re-deriving.

1. **Native subgraph nesting works with partial key overlap.** A parent declaring only `messages`/`final_report`/`entity_type` can do `add_node("research", deep_researcher)`. The subgraph's undeclared keys (`notes`, `research_brief`) are dropped cleanly; parent-only keys survive.
2. **The subgraph echoes `messages` back to the parent.** With an append reducer the parent accumulates duplicates. The parent's `messages` reducer must **replace**, not append.
3. **`entity_type` is returned by research** (written in `classify_entity_type`, `deep_researcher.py:~180`). Read it from the result; do not re-classify.
4. **Tavily crawl on a listing page** yields named people with titles/companies and event contact emails reliably; sponsor company names only where logo alt text exists.

## File Structure

```
src/agent/
  __init__.py          (empty)
  configuration.py     AgentConfiguration: intent, require_send_approval, max_targets
  state.py             Target, AgentState, replace_reducer
  targets.py           resolve_targets across 4 sources + page extraction
  graph.py             the unified graph, intent gates, approval interrupt
  tests/
    test_configuration.py
    test_targets.py
    test_graph_shape.py
    test_per_target_reset.py
```

`configuration.py` / `state.py` / `targets.py` / `graph.py` split by responsibility: config schema, data shapes, input resolution, orchestration. `targets.py` is separate because page extraction is the most complex and most likely to change.

---

### Task 1: Package skeleton and AgentConfiguration

**Files:**
- Create: `src/agent/__init__.py`, `src/agent/configuration.py`, `src/agent/tests/test_configuration.py`
- Modify: `pyproject.toml` (`[tool.setuptools] packages`, `[tool.setuptools.package-dir]`)

**Interfaces:**
- Produces: `OutreachIntent` enum with members `RESEARCH`, `QUALIFY`, `DRAFT`, `SEND` (values `"research"`, `"qualify"`, `"draft"`, `"send"`); `AgentConfiguration` with fields `intent: OutreachIntent`, `require_send_approval: bool`, `max_targets: int`, and classmethod `from_runnable_config(config) -> AgentConfiguration`; module constant `INTENT_DEPTH: dict[OutreachIntent, int]` mapping RESEARCH→0, QUALIFY→1, DRAFT→2, SEND→3.

- [ ] **Step 1: Write the failing test**

```python
# src/agent/tests/test_configuration.py
"""Unit tests for AgentConfiguration - no API calls."""
from agent.configuration import INTENT_DEPTH, AgentConfiguration, OutreachIntent


def test_default_intent_is_research():
    # research is the only intent with no outward-facing side effect
    assert AgentConfiguration.from_runnable_config({}).intent == OutreachIntent.RESEARCH


def test_approval_required_by_default():
    assert AgentConfiguration.from_runnable_config({}).require_send_approval is True


def test_runtime_override():
    cfg = AgentConfiguration.from_runnable_config({"configurable": {"intent": "draft"}})
    assert cfg.intent == OutreachIntent.DRAFT


def test_intent_depth_ordering():
    assert (INTENT_DEPTH[OutreachIntent.RESEARCH]
            < INTENT_DEPTH[OutreachIntent.QUALIFY]
            < INTENT_DEPTH[OutreachIntent.DRAFT]
            < INTENT_DEPTH[OutreachIntent.SEND])


def test_no_env_var_collision_with_other_configs():
    from open_deep_research.configuration import Configuration
    from sales_outreach.configuration import SalesConfiguration
    mine = set(AgentConfiguration.model_fields)
    assert not (mine & set(Configuration.model_fields))
    assert not (mine & set(SalesConfiguration.model_fields))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/agent/tests/test_configuration.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Register the package**

In `pyproject.toml`, add `"agent"` to `[tool.setuptools] packages` and `"agent" = "src/agent"` to `[tool.setuptools.package-dir]`. Then run `pip install -e . --no-deps -q`.

- [ ] **Step 4: Write the implementation**

```python
# src/agent/configuration.py
"""Configuration management for the unified agent orchestrator."""

import os
from enum import Enum
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class OutreachIntent(Enum):
    """How deep through the pipeline a run should go."""

    RESEARCH = "research"
    QUALIFY = "qualify"
    DRAFT = "draft"
    SEND = "send"

# Intents nest: each runs the same pipeline, stopping deeper. Gates compare depth
# rather than equality so adding an intent does not mean editing every gate.
INTENT_DEPTH: dict[OutreachIntent, int] = {
    OutreachIntent.RESEARCH: 0,
    OutreachIntent.QUALIFY: 1,
    OutreachIntent.DRAFT: 2,
    OutreachIntent.SEND: 3,
}

class AgentConfiguration(BaseModel):
    """Orchestration settings: how deep to go, and how many targets to allow.

    Field names are checked against Configuration and SalesConfiguration by test,
    because from_runnable_config resolves every field from os.environ[FIELD.upper()]
    and a shared name would mean two configs fighting over one environment variable.
    """

    intent: OutreachIntent = Field(
        default=OutreachIntent.RESEARCH,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "research",
                "description": "How far to take each target. Intents nest: qualify includes research, draft includes qualify, send includes draft.",
                "options": [
                    {"label": "Research only", "value": OutreachIntent.RESEARCH.value},
                    {"label": "Research + score", "value": OutreachIntent.QUALIFY.value},
                    {"label": "Research + score + draft email", "value": OutreachIntent.DRAFT.value},
                    {"label": "Research + score + draft + send", "value": OutreachIntent.SEND.value},
                ]
            }
        }
    )
    require_send_approval: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Pause for explicit approval before each send. Selecting the send intent is authorization, but in batch one toggle would otherwise fire N real emails unattended."
            }
        }
    )
    max_targets: int = Field(
        default=25,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 25,
                "min": 1,
                "max": 200,
                "description": "Refuse runs with more targets than this, rather than discovering the cost after burning quota."
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "AgentConfiguration":
        """Create an AgentConfiguration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
```

Also create an empty `src/agent/__init__.py` and `src/agent/tests/__init__.py` is **not** needed (tests are collected by path).

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest src/agent/tests/test_configuration.py -q`
Expected: 5 passed

- [ ] **Step 6: Lint**

Run: `python -m ruff check src/agent/`
Expected: All checks passed (fix any `I001`/`UP045` with `--fix`)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/agent/__init__.py src/agent/configuration.py src/agent/tests/test_configuration.py
git commit -m "feat(agent): add orchestrator package with intent configuration"
```

---

### Task 2: Target model and state

**Files:**
- Create: `src/agent/state.py`
- Test: `src/agent/tests/test_per_target_reset.py` (created here, extended in Task 6)

**Interfaces:**
- Consumes: nothing.
- Produces: `Target` (pydantic model, fields `name: str`, `website: str | None`, `email: str | None`, `context: str | None`, `source: Literal["prompt","sheet","inline","page"]`, `crm_row_id: str | None`); `replace_reducer(current, new)`; `AgentState` TypedDict; module constant `PER_TARGET_FIELDS: set[str]`.

- [ ] **Step 1: Write the failing test**

```python
# src/agent/tests/test_per_target_reset.py
"""Guards that per-target state is fully reset between targets."""
from agent.state import PER_TARGET_FIELDS, AgentState, Target, replace_reducer

# Fields that legitimately persist across the loop rather than being reset.
CROSS_TARGET_FIELDS = {
    "messages", "targets", "current_target", "targets_remaining", "failures",
}


def test_target_defaults():
    t = Target(name="Acme", source="prompt")
    assert t.website is None and t.email is None and t.crm_row_id is None


def test_replace_reducer_replaces_rather_than_appends():
    # The research subgraph echoes messages back; appending would duplicate them.
    assert replace_reducer(["old"], ["new"]) == ["new"]


def test_per_target_fields_cover_state_minus_cross_target():
    assert PER_TARGET_FIELDS == set(AgentState.__annotations__) - CROSS_TARGET_FIELDS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/agent/tests/test_per_target_reset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.state'`

- [ ] **Step 3: Write the implementation**

```python
# src/agent/state.py
"""State definitions for the unified agent orchestrator."""

from typing import Annotated, List, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from sales_outreach.state import Report, override_reducer


###################
# Structured Outputs
###################

class Target(BaseModel):
    """One subject to process.

    Orchestration only - deliberately carries no kind/type field. The research core
    already classifies subjects via classify_entity_type and returns entity_type;
    duplicating that here would create two sources of truth that can disagree.
    """

    name: str = Field(description="What to research, e.g. a company or person name")
    website: Optional[str] = Field(default=None, description="Known website, used as a research seed")
    email: Optional[str] = Field(default=None, description="Known contact email, if any")
    context: Optional[str] = Field(
        default=None,
        description="Free-form seed facts from the source, e.g. 'CIO, Mahindra Group - speaker, ETCIO 2025'",
    )
    source: Literal["prompt", "sheet", "inline", "page"] = Field(
        description="Where this target came from. Only sheet-sourced targets get CRM write-back."
    )
    crm_row_id: Optional[str] = Field(default=None, description="Sheet row to update; None for every other source")


###################
# State Definitions
###################

def replace_reducer(current_value, new_value):
    """Replace rather than append.

    Used for `messages`: the research subgraph echoes its messages back to the parent,
    so an appending reducer accumulates duplicates on every target.
    """
    return new_value

class AgentInputState(TypedDict):
    """Input is just the prompt naming the subject."""

    messages: Annotated[list[MessageLikeRepresentation], replace_reducer]

class AgentState(TypedDict):
    """Orchestrator state, carried across the per-target loop."""

    messages: Annotated[list[MessageLikeRepresentation], replace_reducer]
    targets: List[Target]
    targets_remaining: int
    current_target: Optional[Target]
    failures: Annotated[list[str], override_reducer]
    # --- per-target, reset by finish_target ---
    final_report: str
    entity_type: Optional[str]
    research_sufficient: bool
    research_gaps: str
    research_retry_count: int
    lead_score: str
    lead_qualified: bool
    reports: Annotated[list[Report], override_reducer]
    outreach_report_link: str
    send_approved: Optional[bool]

# Every field below the marker above must be reset between targets. Enforced by test.
PER_TARGET_FIELDS = {
    "final_report", "entity_type", "research_sufficient", "research_gaps",
    "research_retry_count", "lead_score", "lead_qualified", "reports",
    "outreach_report_link", "send_approved",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/agent/tests/test_per_target_reset.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/state.py src/agent/tests/test_per_target_reset.py
git commit -m "feat(agent): add Target model and orchestrator state"
```

---

### Task 3: Target resolution — prompt, inline, sheet

**Files:**
- Create: `src/agent/targets.py`, `src/agent/tests/test_targets.py`

**Interfaces:**
- Consumes: `Target` from `agent.state`; `AgentConfiguration` from `agent.configuration`.
- Produces: `extract_sheet_id(text) -> str | None`; `parse_inline_list(text) -> list[str]`; `async resolve_targets(prompt: str, config: RunnableConfig) -> list[Target]`. Page extraction is added in Task 4.

- [ ] **Step 1: Write the failing test**

```python
# src/agent/tests/test_targets.py
"""Unit tests for target resolution - no network for the pure helpers."""
import pytest

from agent.targets import extract_sheet_id, parse_inline_list


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/agent/tests/test_targets.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.targets'`

- [ ] **Step 3: Write the implementation**

```python
# src/agent/targets.py
"""Resolve a prompt into the list of targets to process."""

import logging
import re
from typing import List, Optional

from langchain_core.runnables import RunnableConfig

from agent.configuration import AgentConfiguration
from agent.state import Target

##########################
# Input Parsing Utils
##########################

_SHEET_ID_PATTERN = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")
_URL_PATTERN = re.compile(r"https?://\S+")

def extract_sheet_id(text: str) -> Optional[str]:
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

##########################
# Target Resolution
##########################

async def resolve_targets(prompt: str, config: RunnableConfig) -> List[Target]:
    """Turn the prompt into targets, from whichever input shape it carries.

    Order matters: a sheet link is checked before inline names, because a pasted URL
    would otherwise be parsed as a company name.

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
    if sheet_id:
        targets = _targets_from_sheet(sheet_id, config)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/agent/tests/test_targets.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/targets.py src/agent/tests/test_targets.py
git commit -m "feat(agent): resolve targets from prompt, inline list, and sheet link"
```

---

### Task 4: Target resolution — listing page extraction

**Files:**
- Modify: `src/agent/targets.py` (add page branch)
- Modify: `src/agent/tests/test_targets.py` (add extraction tests)
- Create: `src/agent/tests/fixtures/listing_page.txt`

**Interfaces:**
- Consumes: `Target`, `resolve_targets` from Task 3.
- Produces: `parse_listing_content(text) -> list[Target]`; `resolve_targets` gains a page branch triggered when the prompt contains a non-sheet URL.

- [ ] **Step 1: Create the fixture**

Save this to `src/agent/tests/fixtures/listing_page.txt` — it is a trimmed excerpt of a real Tavily crawl of an ETCIO conclave page, including the alt-text gap.

```text
### Rucha Nanavati

CIO, Mahindra Group

### Dheeraj Sinha

CIO, JSW Steel Limited

#### Presenting Partner

Image 9

#### Powered By

Image 11: Adobe

#### Co-Powered By

Image 12: RedisLabs Image 13: Exotel

## Contact Us

For Partnership

Ashish Kumar

ashish.kumar3@timesinternet.in
```

- [ ] **Step 2: Write the failing test**

```python
# append to src/agent/tests/test_targets.py
import pathlib

from agent.targets import parse_listing_content

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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest src/agent/tests/test_targets.py -q -k listing`
Expected: FAIL with `ImportError: cannot import name 'parse_listing_content'`

- [ ] **Step 4: Write the implementation**

Add to `src/agent/targets.py`:

```python
# Matches "Image 12: Adobe" - the alt text is the sponsor name. Bare "Image 9" has no
# alt text and is deliberately not matched: those names are unrecoverable from text.
_LOGO_ALT_PATTERN = re.compile(r"Image\s+\d+:\s*([A-Za-z0-9&.\- ]{2,40})")
# Matches a markdown heading name followed by a "Title, Company" line.
_PERSON_PATTERN = re.compile(r"^#{2,4}\s+([A-Z][A-Za-z.\- ]{2,40})\s*$\n+^([^#\n]+,[^#\n]+)$", re.MULTILINE)

def parse_listing_content(text: str) -> List[Target]:
    """Extract organizations and named people from crawled listing-page content.

    Sponsor names come from logo alt text, which is frequently absent - extraction is
    partial by nature and callers must not assume a complete list. Named people extract
    far more reliably and already carry title and employer, so they skip the separate
    contact-finding step later.
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
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            targets.append(Target(name=clean, source="page", context="Sponsor/partner listed on the source page"))

    return targets

async def _targets_from_page(url: str, config: RunnableConfig) -> List[Target]:
    """Crawl a listing page, then search as a fallback for logo-only sponsor names."""
    import os

    from tavily import AsyncTavilyClient

    client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    text = ""
    try:
        crawled = await client.crawl(
            url=url, max_depth=1, max_breadth=10,
            instructions="Find all sponsors, partners and their tiers, and all speakers with job titles and companies.",
            extract_depth="advanced",
        )
        text = "\n".join(
            (p.get("raw_content") or p.get("content") or "") for p in crawled.get("results", [])
        )
    except Exception as e:
        logging.warning(f"Listing-page crawl failed for {url}: {e}")

    targets = parse_listing_content(text)

    # Logos without alt text are unrecoverable from the crawl; press coverage often
    # lists the same sponsors as plain text, so search recovers some of them.
    try:
        search = await client.search(f"{url} sponsors partners list", max_results=5)
        combined = "\n".join(r.get("content", "") for r in search.get("results", []))
        known = {t.name.lower() for t in targets}
        targets.extend(t for t in parse_listing_content(combined) if t.name.lower() not in known)
    except Exception as e:
        logging.warning(f"Sponsor search fallback failed for {url}: {e}")

    return targets
```

Then wire the branch into `resolve_targets`, replacing the `if sheet_id:` block:

```python
    sheet_id = extract_sheet_id(prompt)
    page_url = None if sheet_id else _first_url(prompt)

    if sheet_id:
        targets = _targets_from_sheet(sheet_id, config)
    elif page_url:
        targets = await _targets_from_page(page_url, config)
    else:
        targets = [Target(name=n, source="inline") for n in parse_inline_list(prompt)]
```

And add the helper:

```python
def _first_url(text: str) -> Optional[str]:
    """First http(s) URL in the text, or None."""
    match = _URL_PATTERN.search(text or "")
    return match.group(0) if match else None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest src/agent/tests/test_targets.py -q`
Expected: 11 passed

- [ ] **Step 6: Commit**

```bash
git add src/agent/targets.py src/agent/tests/test_targets.py src/agent/tests/fixtures/listing_page.txt
git commit -m "feat(agent): extract targets from listing pages with search fallback"
```

---

### Task 5: The unified graph

**Files:**
- Create: `src/agent/graph.py`, `src/agent/tests/test_graph_shape.py`
- Modify: `langgraph.json`

**Interfaces:**
- Consumes: everything from Tasks 1-4; `deep_researcher` from `open_deep_research.deep_researcher`; from `sales_outreach.outreach_automation` the node bodies are **not** reused directly (they take `GraphState`); instead reuse `sales_outreach.utils.invoke_llm`, `get_report`, `save_reports_locally`, `qualification_decision`, `research_sufficiency_decision`, and the prompts.
- Produces: module-level compiled graph `unified_agent`.

- [ ] **Step 1: Write the failing test**

```python
# src/agent/tests/test_graph_shape.py
"""Structural tests for the unified graph - no API calls, no credentials."""
from agent.graph import unified_agent

EDGES = {(e.source, e.target) for e in unified_agent.get_graph().edges}
NODES = set(unified_agent.get_graph().nodes)


def test_graph_compiles_without_credentials():
    # Proves no OAuth or client construction happens at import time.
    assert unified_agent is not None


def test_research_is_a_nested_subgraph_node():
    assert "research" in NODES


def test_all_four_intent_depths_are_reachable():
    assert ("prepare_research", "research") in EDGES
    assert ("check_research_sufficiency", "score_target") in EDGES
    assert ("score_target", "generate_materials") in EDGES
    assert ("generate_materials", "approve_send") in EDGES


def test_send_is_gated_by_approval():
    # There must be no path from material generation straight to sending.
    assert ("generate_materials", "send_email") not in EDGES
    assert ("approve_send", "send_email") in EDGES


def test_every_branch_converges_on_finish_target():
    for source in ("check_research_sufficiency", "score_target", "generate_materials", "approve_send", "send_email"):
        assert any(s == source and t == "finish_target" for s, t in EDGES), source


def test_per_target_loop_closes_and_terminates():
    assert ("finish_target", "next_target") in EDGES
    assert ("next_target", "__end__") in EDGES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/agent/tests/test_graph_shape.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.graph'`

- [ ] **Step 3: Write the implementation**

```python
# src/agent/graph.py
"""Main LangGraph implementation for the unified research + outreach agent.

One entry point: the prompt names the subject, config names how deep to go. Research is
nested as a native subgraph so its internals stay visible in traces and Studio, and the
sales work is reused from sales_outreach rather than reimplemented.
"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from open_deep_research.deep_researcher import deep_researcher

from agent.configuration import INTENT_DEPTH, AgentConfiguration, OutreachIntent
from agent.state import AgentInputState, AgentState, Target
from agent.targets import resolve_targets
from sales_outreach.configuration import SalesConfiguration
from sales_outreach.prompts import (
    CHECK_RESEARCH_SUFFICIENCY_PROMPT,
    GENERATE_OUTREACH_REPORT_PROMPT,
    PERSONALIZE_EMAIL_PROMPT,
    PROOF_READER_PROMPT,
    SCORE_LEAD_PROMPT,
)
from sales_outreach.state import EmailResponse, Report, ResearchSufficiency
from sales_outreach.tools.base.gmail_tools import GmailTools
from sales_outreach.utils import (
    invoke_llm,
    qualification_decision,
    research_sufficiency_decision,
    save_reports_locally,
)

RESEARCH_REPORT_TITLE = "General Lead Research Report"


def _at_least(configurable: AgentConfiguration, intent: OutreachIntent) -> bool:
    """Whether the configured intent goes at least as deep as the given one."""
    return INTENT_DEPTH[configurable.intent] >= INTENT_DEPTH[intent]


async def start_run(state: AgentState, config: RunnableConfig) -> Command[Literal["next_target"]]:
    """Resolve the prompt into a target list.

    Fails loudly rather than processing zero targets silently, which would look like a
    successful no-op run.

    Args:
        state: Input state carrying the user's prompt
        config: Runtime configuration for target cap and sheet loading

    Returns:
        Command to enter the per-target loop
    """
    prompt = state["messages"][-1].content if state.get("messages") else ""
    targets = await resolve_targets(str(prompt), config)
    return Command(
        goto="next_target",
        update={"targets": targets, "targets_remaining": len(targets), "failures": []},
    )


async def next_target(state: AgentState, config: RunnableConfig) -> Command[Literal["prepare_research", "__end__"]]:
    """Pop the next target, or end when the queue is empty.

    Args:
        state: Current state holding the remaining targets
        config: Runtime configuration (unused, kept for node signature consistency)

    Returns:
        Command to research the next target, or to end
    """
    remaining = list(state.get("targets", []))
    if not remaining:
        failures = state.get("failures", [])
        if failures:
            logging.warning(f"Finished with {len(failures)} failed target(s): {failures}")
        logging.info("Finished - no targets remaining")
        return Command(goto=END)

    current = remaining.pop(0)
    logging.info(f"Processing target: {current.name} ({len(remaining)} remaining)")
    return Command(
        goto="prepare_research",
        update={"current_target": current, "targets": remaining},
    )


async def prepare_research(state: AgentState, config: RunnableConfig) -> Command[Literal["research"]]:
    """Build the research query for the current target and hand off to the subgraph.

    Exists as its own node because the research subgraph reads `messages` from parent
    state; this is where that gets set, and where a retry adds gap focus.

    Args:
        state: Current state holding the target and any prior research gap
        config: Runtime configuration (unused, kept for node signature consistency)

    Returns:
        Command into the research subgraph
    """
    target: Target = state["current_target"]

    query = f"Research {target.name}"
    if target.website:
        query += f" ({target.website})"
    if target.context:
        query += f". Known context: {target.context}"
    query += (
        " - what they do, services/products offered, size, and any named contacts "
        "or contact information."
    )

    gaps = state.get("research_gaps", "")
    if gaps:
        query += f" A previous pass found this gap - focus on filling it: {gaps}"

    return Command(goto="research", update={"messages": [HumanMessage(content=query)]})


async def check_research_sufficiency(state: AgentState, config: RunnableConfig) -> Command[Literal["score_target", "prepare_research", "finish_target"]]:
    """Gate on whether research has enough substance to act on.

    Args:
        state: Current state holding the research report
        config: Runtime configuration with model settings and retry budget

    Returns:
        Command to score, retry research, or stop with this target
    """
    sales_cfg = SalesConfiguration.from_runnable_config(config)
    agent_cfg = AgentConfiguration.from_runnable_config(config)
    report = state.get("final_report", "")

    result = await invoke_llm(
        system_prompt=CHECK_RESEARCH_SUFFICIENCY_PROMPT,
        user_message=report,
        model_name=sales_cfg.research_sufficiency_model,
        config=config,
        response_format=ResearchSufficiency,
    )

    update = {
        "research_sufficient": result.sufficient,
        "research_gaps": result.gaps,
        "reports": [Report(title=RESEARCH_REPORT_TITLE, content=report, is_markdown=True)],
    }
    decision = research_sufficiency_decision({**state, **update}, sales_cfg.max_research_retries)

    if decision == "retry":
        logging.warning(f"Research insufficient, retrying with gap focus: {result.gaps}")
        update["research_retry_count"] = state.get("research_retry_count", 0) + 1
        return Command(goto="prepare_research", update=update)

    if decision == "insufficient" or not _at_least(agent_cfg, OutreachIntent.QUALIFY):
        return Command(goto="finish_target", update=update)

    return Command(goto="score_target", update=update)


async def score_target(state: AgentState, config: RunnableConfig) -> Command[Literal["generate_materials", "finish_target"]]:
    """Score partnership fit, then route on qualification and intent depth.

    For a person target, scoring evaluates their employer: SCORE_LEAD_PROMPT assesses
    company partner-track fit and is meaningless applied to an individual.

    Args:
        state: Current state holding the research report and entity type
        config: Runtime configuration with model settings and score threshold

    Returns:
        Command to generate materials, or to stop with this target
    """
    sales_cfg = SalesConfiguration.from_runnable_config(config)
    agent_cfg = AgentConfiguration.from_runnable_config(config)
    target: Target = state["current_target"]

    subject = state.get("final_report", "")
    if state.get("entity_type") == "person":
        subject = (
            f"Score the partnership fit of the EMPLOYER of this person, not the individual.\n\n{subject}"
        )

    score = (await invoke_llm(
        system_prompt=SCORE_LEAD_PROMPT,
        user_message=subject,
        model_name=sales_cfg.lead_scoring_model,
        config=config,
    )).strip()

    qualified = qualification_decision(score, sales_cfg.lead_score_threshold) == "qualified"
    logging.info(f"{target.name} scored {score} ({'qualified' if qualified else 'not qualified'})")
    update = {"lead_score": score, "lead_qualified": qualified}

    if not qualified or not _at_least(agent_cfg, OutreachIntent.DRAFT):
        return Command(goto="finish_target", update=update)
    return Command(goto="generate_materials", update=update)


async def generate_materials(state: AgentState, config: RunnableConfig) -> Command[Literal["approve_send", "finish_target"]]:
    """Write the outreach report and email, and create a Gmail draft.

    A draft is always created; sending is a separate, gated step so a bad generation can
    never reach a real prospect without an explicit decision.

    Args:
        state: Current state holding the research report and target
        config: Runtime configuration with model settings

    Returns:
        Command to the approval gate when sending, else to finish this target
    """
    sales_cfg = SalesConfiguration.from_runnable_config(config)
    agent_cfg = AgentConfiguration.from_runnable_config(config)
    target: Target = state["current_target"]
    report = state.get("final_report", "")

    outreach = await invoke_llm(
        system_prompt=GENERATE_OUTREACH_REPORT_PROMPT,
        user_message=f"**Research Report:**\n\n{report}",
        model_name=sales_cfg.outreach_report_model,
        config=config,
    )
    outreach = await invoke_llm(
        system_prompt=PROOF_READER_PROMPT,
        user_message=outreach,
        model_name=sales_cfg.outreach_report_model,
        config=config,
    )

    email = await invoke_llm(
        system_prompt=PERSONALIZE_EMAIL_PROMPT,
        user_message=(
            f"# **Lead & company Information:**\n\n{report}\n\n"
            "# Outreach report Link:\n\nNo hosted report link is available - omit that "
            "line from the email entirely rather than including a placeholder."
        ),
        model_name=sales_cfg.email_model,
        config=config,
        response_format=EmailResponse,
    )

    update = {
        "reports": [
            Report(title="Outreach Report", content=outreach, is_markdown=True),
            Report(title="Personalized Email", content=email.email, is_markdown=False),
        ]
    }

    recipient = target.email
    if recipient:
        GmailTools().create_draft_email(
            recipient=recipient, subject=email.subject, email_content=email.email
        )
    else:
        logging.warning(f"No email address known for {target.name}; draft not created")

    if _at_least(agent_cfg, OutreachIntent.SEND) and recipient:
        return Command(goto="approve_send", update=update)
    return Command(goto="finish_target", update=update)


async def approve_send(state: AgentState, config: RunnableConfig) -> Command[Literal["send_email", "finish_target"]]:
    """Pause for explicit human approval before sending.

    Selecting the send intent is authorization, but in batch mode one toggle would
    otherwise fire N real emails unattended, so the default is to confirm each one.

    Args:
        state: Current state holding the drafted email and target
        config: Runtime configuration with the approval toggle

    Returns:
        Command to send, or to skip sending for this target
    """
    from sales_outreach.utils import get_report

    agent_cfg = AgentConfiguration.from_runnable_config(config)
    target: Target = state["current_target"]

    if not agent_cfg.require_send_approval:
        return Command(goto="send_email", update={"send_approved": True})

    decision = interrupt({
        "action": "confirm_send",
        "recipient": target.email,
        "target": target.name,
        "email": get_report(state.get("reports", []), "Personalized Email"),
    })
    approved = decision is True or (isinstance(decision, dict) and decision.get("approved") is True)

    if not approved:
        logging.warning(f"Send rejected for {target.name}; keeping the draft")
        return Command(goto="finish_target", update={"send_approved": False})
    return Command(goto="send_email", update={"send_approved": True})


async def send_email(state: AgentState, config: RunnableConfig) -> Command[Literal["finish_target"]]:
    """Send the approved email.

    Args:
        state: Current state holding the approved email and target
        config: Runtime configuration (unused, kept for node signature consistency)

    Returns:
        Command to finish this target
    """
    from sales_outreach.utils import get_report

    target: Target = state["current_target"]
    body = get_report(state.get("reports", []), "Personalized Email")
    GmailTools().send_email(
        recipient=target.email, subject=f"PACE Uttarakhand - {target.name}", email_content=body
    )
    logging.info(f"Email sent to {target.email}")
    return Command(goto="finish_target")


async def finish_target(state: AgentState, config: RunnableConfig) -> Command[Literal["next_target"]]:
    """Persist this target's output, write back to CRM if applicable, and reset state.

    CRM write-back only happens for sheet-sourced targets, since an inline or page-derived
    name has no row to update. Every per-target field is reset here: anything left behind
    carries into the next target, which previously leaked one lead's report link into the
    next lead's email.

    Args:
        state: Current state holding this target's results
        config: Runtime configuration for the lead loader

    Returns:
        Command back to the loop head
    """
    from sales_outreach.utils import get_current_date, get_lead_loader

    target: Target = state["current_target"]
    reports = state.get("reports", [])
    if reports:
        save_reports_locally(reports)

    if target.source == "sheet" and target.crm_row_id:
        if not state.get("research_sufficient", True):
            status = "NEEDS_MORE_RESEARCH"
        elif state.get("lead_qualified"):
            status = "ATTEMPTED_TO_CONTACT"
        else:
            status = "NOT_QUALIFIED"
        get_lead_loader(config).update_record(target.crm_row_id, {
            "Status": status,
            "Score": state.get("lead_score", "N/A"),
            "Last Contacted": get_current_date(),
        })

    return Command(
        goto="next_target",
        update={
            "targets_remaining": max(0, state.get("targets_remaining", 1) - 1),
            "reports": {"type": "override", "value": []},
            "final_report": "",
            "entity_type": None,
            "research_sufficient": False,
            "research_gaps": "",
            "research_retry_count": 0,
            "lead_score": "",
            "lead_qualified": False,
            "outreach_report_link": "",
            "send_approved": None,
        },
    )


# Unified Agent Graph Construction
# One prompt in, one target list, then research -> qualify -> draft -> send as deep as
# the configured intent allows.
unified_agent_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=AgentConfiguration,
)

unified_agent_builder.add_node("start_run", start_run)
unified_agent_builder.add_node("next_target", next_target)                          # Per-target loop head
unified_agent_builder.add_node("prepare_research", prepare_research)
unified_agent_builder.add_node("research", deep_researcher)                         # Nested research subgraph
unified_agent_builder.add_node("check_research_sufficiency", check_research_sufficiency)
unified_agent_builder.add_node("score_target", score_target)
unified_agent_builder.add_node("generate_materials", generate_materials)
unified_agent_builder.add_node("approve_send", approve_send)                        # Human-in-the-loop gate
unified_agent_builder.add_node("send_email", send_email)
unified_agent_builder.add_node("finish_target", finish_target)

# The subgraph node cannot return a Command, so its one outgoing edge is declared here.
unified_agent_builder.add_edge(START, "start_run")
unified_agent_builder.add_edge("research", "check_research_sufficiency")

unified_agent = unified_agent_builder.compile()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/agent/tests/test_graph_shape.py -q`
Expected: 6 passed

- [ ] **Step 5: Register the graph in Studio**

In `langgraph.json`, add to `graphs`:

```json
"Unified Agent": "./src/agent/graph.py:unified_agent"
```

- [ ] **Step 6: Verify all three graphs load**

Run: `langgraph dev --no-browser` and confirm the log contains `graph_id='Unified Agent'` alongside `'Deep Researcher'` and `'Sales Outreach'`. Stop the server.

- [ ] **Step 7: Commit**

```bash
git add src/agent/graph.py src/agent/tests/test_graph_shape.py langgraph.json
git commit -m "feat(agent): add unified graph with intent gates and nested research"
```

---

### Task 6: Per-target reset guard

**Files:**
- Modify: `src/agent/tests/test_per_target_reset.py`

**Interfaces:**
- Consumes: `finish_target` from `agent.graph`; `PER_TARGET_FIELDS` from `agent.state`.

- [ ] **Step 1: Write the failing test**

```python
# append to src/agent/tests/test_per_target_reset.py
import asyncio
from unittest.mock import patch

from agent.graph import finish_target


def _run_finish_target(source="inline", crm_row_id=None):
    state = {
        "current_target": Target(name="Acme", source=source, crm_row_id=crm_row_id),
        "targets_remaining": 2,
        "reports": [],
        "research_sufficient": True,
        "lead_qualified": True,
        "lead_score": "8.0",
    }
    with patch("sales_outreach.utils.get_lead_loader") as loader:
        command = asyncio.run(finish_target(state, {}))
    return command.update, loader


def test_finish_target_resets_every_per_target_field():
    update, _ = _run_finish_target()
    missing = PER_TARGET_FIELDS - set(update)
    assert not missing, (
        f"finish_target does not reset {sorted(missing)} - these would carry into the "
        "next target. Reset them there, or add them to CROSS_TARGET_FIELDS with a reason."
    )


def test_reports_cleared_via_override_envelope():
    update, _ = _run_finish_target()
    assert update["reports"] == {"type": "override", "value": []}


def test_crm_skipped_for_non_sheet_targets():
    _, loader = _run_finish_target(source="inline")
    loader.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/agent/tests/test_per_target_reset.py -q`
Expected: FAIL with `ImportError: cannot import name 'finish_target'` if Task 5 is incomplete; otherwise all three pass immediately (the guard is written against an implementation that already satisfies it).

- [ ] **Step 3: Prove the guard actually catches a regression**

Temporarily delete the `"lead_score": "",` line from `finish_target`'s update dict, re-run the test, and confirm it fails naming `lead_score`. Restore the line and confirm it passes again. This verifies the guard is real rather than vacuous.

- [ ] **Step 4: Commit**

```bash
git add src/agent/tests/test_per_target_reset.py
git commit -m "test(agent): guard per-target state reset and CRM source gating"
```

---

### Task 7: End-to-end intent-depth verification and docs

**Files:**
- Create: `src/agent/tests/test_intent_depth.py`
- Modify: `docs/unified-agent-design.md` (mark implemented), `src/sales_outreach/docs/system-workflow.md` (add unified-entry section), `README.md` (fork section)

**Interfaces:**
- Consumes: `unified_agent` from `agent.graph`.

- [ ] **Step 1: Write the failing test**

Research and all LLM calls are mocked, so this costs no quota and never sends mail.

```python
# src/agent/tests/test_intent_depth.py
"""Each intent must stop at exactly its own depth. Fully mocked - no quota, no email."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agent.graph import unified_agent

CACHED_REPORT = "Acme Corp is an IT services firm founded in 2004, serving manufacturing clients."


def _run(intent):
    calls = {"score": 0, "materials": 0, "draft": 0, "send": 0}

    async def fake_research(inputs, config=None):
        return {"final_report": CACHED_REPORT, "entity_type": "company"}

    async def fake_invoke_llm(system_prompt, user_message, model_name, config, response_format=None):
        if response_format is not None and hasattr(response_format, "sufficient"):
            return response_format(sufficient=True, gaps="")
        if "SPIN" in system_prompt or "score" in system_prompt.lower():
            calls["score"] += 1
            return "9.0"
        if response_format is not None:
            calls["materials"] += 1
            return response_format(subject="s", email="b")
        calls["materials"] += 1
        return "generated text"

    class _Gmail:
        def create_draft_email(self, **kw):
            calls["draft"] += 1

        def send_email(self, **kw):
            calls["send"] += 1

    with patch("agent.graph.deep_researcher") as dr, \
         patch("agent.graph.invoke_llm", side_effect=fake_invoke_llm), \
         patch("agent.graph.GmailTools", _Gmail), \
         patch("agent.graph.save_reports_locally"):
        dr.ainvoke = AsyncMock(side_effect=fake_research)
        config = {
            "recursion_limit": 100,
            "configurable": {"intent": intent, "require_send_approval": False},
        }
        asyncio.run(unified_agent.ainvoke(
            {"messages": [("user", "research Acme Corp")]}, config
        ))
    return calls


@pytest.mark.parametrize("intent,expect_score,expect_draft", [
    ("research", False, False),
    ("qualify", True, False),
    ("draft", True, True),
])
def test_intent_stops_at_correct_depth(intent, expect_score, expect_draft):
    calls = _run(intent)
    assert (calls["score"] > 0) is expect_score
    assert (calls["draft"] > 0) is expect_draft
    # No intent below send may ever send.
    assert calls["send"] == 0
```

Note: `test_intent_depth.py` deliberately omits a `send` case — sending is covered by the structural gate test in Task 5 (`approve_send` is the only path into `send_email`). Exercising a real send path in an automated test is not worth the risk of a misconfigured run reaching a real inbox.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/agent/tests/test_intent_depth.py -q`
Expected: FAIL until the mock target names in `patch(...)` match the real import names in `agent/graph.py`. Adjust the patch targets, not the graph.

- [ ] **Step 3: Make it pass**

Fix patch targets so they name the symbols as imported in `agent/graph.py` (`agent.graph.invoke_llm`, `agent.graph.GmailTools`, `agent.graph.deep_researcher`). No production code changes should be needed; if one is, that indicates a real coupling problem worth fixing rather than mocking around.

- [ ] **Step 4: Run the full suite and lint**

Run: `python -m pytest tests/unit/ src/sales_outreach/tests/ src/agent/tests/ -q`
Expected: all pass, including the 48 pre-existing tests.

Run: `python -m ruff check src/agent/`
Expected: All checks passed.

- [ ] **Step 5: Update the docs**

In `docs/unified-agent-design.md`, change the status line to `**Status:** implemented` and delete the "Open technical question" paragraph under `## State`, replacing it with the verified finding: native subgraph nesting works with partial key overlap, and the parent's `messages` reducer must replace rather than append because the subgraph echoes messages back.

In `src/sales_outreach/docs/system-workflow.md`, add a short section noting that the sales graph is now also reachable through the unified agent, and that `intent` controls depth.

In the root `README.md` fork section, add the unified agent to the list of what this fork adds.

- [ ] **Step 6: Commit**

```bash
git add src/agent/tests/test_intent_depth.py docs/unified-agent-design.md src/sales_outreach/docs/system-workflow.md README.md
git commit -m "test(agent): verify intent depth end-to-end; update docs"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: intents and approval → Tasks 1, 5; four target sources → Tasks 3, 4; native subgraph nesting → Task 5; `Target` without a kind field and `entity_type` read from research → Tasks 2, 5; CRM only for sheet sources → Tasks 5, 6; per-target reset → Tasks 2, 6; error handling (loud resolution failure, `max_targets`, per-target failure isolation) → Tasks 3, 5; testing → Tasks 1-7; docs deliverable → Task 7.

**Placeholder scan.** No TBD/TODO. Every code step carries runnable code; every test step carries real assertions.

**Type consistency.** `Target`, `AgentState`, `PER_TARGET_FIELDS`, `replace_reducer` are defined in Task 2 and used with the same names in Tasks 3-6. `INTENT_DEPTH` and `OutreachIntent` are defined in Task 1 and used in Task 5. `resolve_targets` and `parse_listing_content` are defined in Tasks 3-4 and consumed in Task 5. Node names in the Task 5 tests match the `add_node` calls exactly.

**Known gap, deliberate.** Per-target failure isolation is specified in the spec's error table but only partially implemented: `failures` is in state and logged by `next_target`, but no task wraps node bodies in try/except. This is deferred rather than hidden — wrapping every node adds noise, and the right shape (a LangGraph retry/error policy versus manual handling) is worth deciding once the graph is running. Revisit before any batch run larger than a handful of targets.
