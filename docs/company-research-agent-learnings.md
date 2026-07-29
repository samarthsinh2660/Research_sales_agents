# What to adapt from company-research-agent

Checked against the actual source (`guy-hartstein/company-research-agent`, cloned and inspected directly — LangGraph+Tavily, 2.2k stars, updated today). This one's structurally the closest thing to what we're building for the research side, so worth a proper look.

## Architecture (`backend/nodes/`)

```
grounding.py          → crawls the company's own website first (Tavily .crawl(), not .search())
nodes/researchers/
  ├── company.py       → company-analyzer sub-agent
  ├── industry.py       → industry-analyzer sub-agent
  ├── financial.py       → financial-analyzer sub-agent
  └── news.py             → news-analyzer sub-agent
curator.py             → filters documents by Tavily relevance score (threshold 0.4) before use
enricher.py             → enriches curated docs
briefing.py               → per-category briefing synthesis
editor.py                   → final report assembly
```

Same `.py` also has a **React+Vite UI** (`ui/`) streaming live progress events (`research_init`, `crawl_start`, etc.) to the browser — this is literally the "web dashboard" we said wasn't needed yet. Worth keeping as a concrete reference to build against when we do get there, rather than starting from a blank page.

**Confirms, again**: no LinkedIn module, no RapidAPI usage anywhere (grepped the whole repo, zero matches) — same as GPT-Researcher. Nobody in this space has solved the LinkedIn/structured-people-data problem; it's genuinely ours to build.

## What's actually worth adapting

### 1. `tavily_client.crawl()`, not just `.search()`
`grounding.py` uses Tavily's **crawl** endpoint, not search: give it a known URL + natural-language instructions, and it crawls the site (`max_depth`, `max_breadth`, `extract_depth="advanced"`) looking for what you asked for. We only use `tavily_search` (generic query-based search) right now.

**Adapt directly**: once our entity-type layer resolves a known official URL (company site, gov dept portal, college site), use `.crawl()` instead of `.search()` for that source — far richer than a generic web search when you already know exactly where to look. This is a genuinely better fit for the `gov_source_module`/`website_module` we planned than what we have now.

### 2. Relevance-score filtering before compression
`curator.py` filters documents using **Tavily's own relevance score** (threshold 0.4) before they ever reach summarization. We don't do this — `compress_research` takes everything the researcher found, no filtering step. This directly explains part of why our OpsHub report only had 4 sources: no filtering means noise competes equally with signal, and nothing forces breadth.

**Adapt**: add a curation/filter step between search and summarization, dropping low-relevance results before spending model calls summarizing them. Cheap to add, directly improves both cost and report quality.

### 3. Fixed 4-node researcher split — validates our Entity Schema Registry design
This repo hardcodes exactly one "row" of what would be our registry: entity type = `company`, modules = `[company, industry, financial, news]`, each with its own dedicated query-generation prompt. It's proof the pattern works in production (2.2k stars, actively maintained) — we're not inventing something untested, we're generalizing something that already works for one entity type into a registry that works for several (`company`, `college`, `government_dept`, `person`).

### 4. Streaming dashboard — options for when we get there
Not urgent (we already deferred the dashboard decision), but worth documenting properly since we deferred it, not ruled it out. Checked the actual setup: root `application.py` is a FastAPI server exposing WebSocket-style event streaming, `ui/` is a separate React+Vite frontend consuming it, and the whole repo is **Apache-2.0 licensed** — permissive, safe to fork/reuse with attribution (unlike `kaymen99`'s sales repo, which has no license file at all).

Three real options, not just "study it":
- **Fork it directly** — `application.py` + `ui/` as a starting point, rewire the event names/state shape from their `job_status`/`company_data` model to our graph's node names and Research Profile shape. Fastest path to a working dashboard, most reuse, least design control.
- **Use it as a reference architecture only** — build our own FastAPI+React pair from scratch, but follow its event-streaming pattern (LangGraph node → event dict → WebSocket → frontend) rather than inventing our own. More control, more work.
- **Skip it entirely for now** — keep using LangGraph Studio as the only interface until there's a concrete need (e.g. Pranav needing to trigger runs without touching code). Zero work now, revisit later.

No decision needed yet — just making sure "not needed yet" doesn't quietly become "forgotten." The original trigger condition still holds: once Pranav (non-engineer) needs to repeatedly trigger runs and review output without touching code, option 1 (fork) is the fastest real path given the Apache-2.0 license permits it outright.

## Bottom line

This repo is the strongest evidence yet that our planned entity-type layer is the right direction — it's essentially a single hardcoded instance of it, working well in production. Two concrete technical upgrades worth pulling in regardless of entity-layer timing: **Tavily's `.crawl()` for known-URL sources** and **relevance-score filtering before summarization** — both are small, contained changes to `utils.py` that would measurably improve report quality without waiting on the bigger entity-type work.
