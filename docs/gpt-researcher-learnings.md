# What to adapt from GPT-Researcher

Checked against the actual source (`assafelovic/gpt-researcher`, cloned and grepped directly), not assumptions. Two direct questions get answered first, then what's actually worth borrowing.

## Does GPT-Researcher have a LinkedIn module / use RAPIDAPI_KEY?

**No — confirmed by grepping the entire repo for `rapidapi` and `linkedin`: zero matches.** It has no LinkedIn-specific capability at all, free or otherwise. This isn't something we're missing relative to them — it doesn't exist there either. Their retriever list (`gpt_researcher/retrievers/`) is entirely generic web/academic search: `tavily`, `google`, `bing`, `serper`, `brave`, `duckduckgo`, `exa`, `searx`, `searchapi`, `bocha`, `arxiv`, `pubmed_central`, `semantic_scholar`, `openalex`, plus MCP and custom retrievers. Nothing structured-data/social-profile-specific.

So the LinkedIn module (via RapidAPI, ticket already raised) is genuinely our own build, not something to copy or compare against here.

## Does GPT-Researcher have entity-type awareness (person vs. company vs. gov dept)?

**No.** Checked `gpt_researcher/utils/enum.py` — it has `ReportType` (research/detailed/outline/resource/subtopic/deep — output *format*), `Tone` (objective/formal/persuasive/etc. — writing *style*), and `ReportSource` (web/local/azure/vectorstore/hybrid — where content comes from). None of these represent "what kind of subject is this query about" — there's no concept of adapting research strategy per entity type.

**This means our planned entity-type layer (`classify_entity_type` → Entity Schema Registry → `build_research_plan`) is genuine white space** — not something to adapt from GPT-Researcher, something they don't have at all. Worth being confident about, not worth second-guessing against their approach.

## What IS worth adapting

### 1. Multi-retriever aggregation (the real reason they hit 20+ sources)
`RETRIEVER` env var accepts a **comma-separated list** (`config/config.py:78-83, 188-201`), e.g. `tavily,google,serper` — a single query runs across *all* configured retrievers in parallel and aggregates results. We currently only wire up Tavily. This is the actual mechanism behind their higher source counts, not a smarter single-retriever algorithm.

**Adapt**: extend our search module to optionally run multiple search backends per query (Tavily + one more, e.g. Google/Serper) and merge results, instead of a single-source ceiling.

### 2. Explicit subtopic decomposition (`DetailedReport`)
`backend/report_type/detailed_report/detailed_report.py` — before researching, it generates a list of subtopics for the query, then runs a full parallel sub-research pass *per subtopic*, then merges into one report. This forces breadth deliberately rather than trusting the model to decide when it has "enough."

**Adapt**: our supervisor already does parallel research delegation, but doesn't force explicit subtopic decomposition as a discrete upfront step. Worth adding a "break this entity into N research angles" step before the supervisor loop — this pairs naturally with the entity-type layer, since the Entity Schema Registry's `targets` list (per module, what to look for) is essentially a pre-defined subtopic list per entity type already.

### 3. A dedicated publish/citation-formatting step
Their report generation separates "write" from "publish" more explicitly than ours does. Worth adding a citation-verification pass in `final_report_generation` rather than trusting the writer model to self-cite correctly in one shot.

### 4. Configurable report depth/tone as first-class dimensions
`ReportType` + `Tone` are both plain config enums. Once the entity-type layer exists, it's natural to pair it with something similar — e.g. a `government_dept` report defaulting to Formal/Analytical tone and detailed depth, a quick person lookup defaulting to a shorter outline-style report. Not urgent, but a clean extension point to keep in mind when designing the registry schema.

## Bottom line

GPT-Researcher is stronger on raw source breadth (multi-retriever aggregation + forced subtopic decomposition) — both adaptable without adopting their codebase. It has **no** entity-type awareness and **no** structured-data/LinkedIn capability — both are genuinely ours to build, not gaps relative to them.
