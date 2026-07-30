# Is what we've found enough? Honest gap analysis

Direct answer: **no.** What's in `gpt-researcher-learnings.md` and `company-research-agent-learnings.md` are real, worthwhile improvements — but they only improve *generic research quality*. They don't touch the actual reasons this project exists. Laying out what implementing "just that much" gets us, and what's still missing.

## If we implement everything from both learnings docs

| Change | Effect | Status |
|---|---|---|
| Subtopic decomposition | Forced breadth instead of trusting the model to stop early | **Done** — new `decompose_subtopics` node, verified via before/after test (OpsHub: 4 sources → 14, more diverse product/partnership coverage) |
| Relevance-score filtering | Less noise reaching summarization, fewer wasted model calls | **Done** — threshold tuned to 0.2 (not 0.4) after checking real Tavily scores; 0.4 was silently cutting legitimate third-party sources like G2 |
| Citation/publish step | More reliable, verifiable sourcing in the final report | **Done** — `verify_citations` cross-checks every cited URL against real findings; also found and fixed 2 pre-existing bugs where Gemini's "thinking mode" content (list-of-blocks, not a string) was getting corrupted by raw `str()` calls in `compress_research` and `final_report_generation`, which had been silently poisoning findings/citations this whole session |
| Multi-retriever aggregation | More sources per report (closer to GPT-Researcher's 20+ vs. our 4) | Not started — blocked on a Google Custom Search Engine ID (`cx`), have the API key |
| `tavily.crawl()` for known URLs | Richer extraction when we already know where to look | Not started |

**Net result: we become roughly as good as the two tools we studied, at generic web research.** That's a real, useful floor to reach — but it's not the mission. Pranav's own framing was explicit: *"we cannot build a better generic researcher than Google Deep Research, but we can build a better government policy researcher, and an even better Uttarakhand government policy researcher."* None of the above touches that at all.

**Also newly confirmed, not from the learnings docs: reports have no contact information.** Even a good report (14 verified sources) names zero people — no decision-maker, no email, no LinkedIn profile. The sales/outreach agent has nothing to send an email *to*. This is exactly the `people_research`/`linkedin` module gap in item 3 below, and it's more urgent than it looked before we actually generated a report and checked — deliberately deferred for now, revisit before the outreach side is wired up.

## What's still 0% built, and matters more than anything on the list above

1. **Uttarakhand/government source-of-truth embedding** — the actual differentiator. Neither GPT-Researcher nor company-research-agent has *any* India/government-specific knowledge (confirmed — neither repo references gov portals, gazette formats, RTI structures, scheme databases). This is 100% ours to build, and it's the one thing that makes us better than generic tools rather than merely competitive with them. Not started.
2. **The sales graph trim** — scoped in conversation (drop kaymen99's research nodes, wire the shared research core's output into score→email→send) but not implemented in code yet.
3. **The entity-type layer itself** — **Done.** `classify_entity_type` + `build_research_plan` + `entity_registry.py`, verified with live model calls across all 4 types (company/college/government_dept/person) correctly classifying real test queries and producing entity-specific brief guidance. Still missing: dedicated per-entity tool modules (LinkedIn, gov_source) — this only steers the existing generic search so far, doesn't add new sources.
4. **Evaluation** — we have no way to systematically check report quality beyond spot-checking one output (like the OpsHub report). GPT-Researcher participates in a public benchmark (DeepResearchGym); we have nothing, not even a small internal eval set. Without this, every future change is a guess about whether it helped.
5. **Cross-run memory / dedup** — a 90-day hackathon will likely research the same departments, colleges, and companies more than once. Neither donor repo persists findings between runs. Worth considering once the entity layer exists — re-researching the same government department from scratch every time is wasteful.
6. **Cost/quota management at scale** — multi-retriever + subtopic decomposition both mean *more* model calls per report, not fewer. We're already hitting Gemini free-tier limits with the current, simpler pipeline. Implementing the breadth improvements without addressing this makes the quota problem worse, not better.

## Straight answer

Implementing only what's in the two learnings docs makes us **competitive with existing open-source tools at generic research** — a reasonable, bounded win, worth doing. It does **not** make us better than them for the actual job (government/Uttarakhand research + sales outreach), because the thing that would make us better isn't "research technique," it's "domain knowledge embedded into the harness" — and that's still unbuilt. The entity-type layer + gov-source module is the highest-leverage next step, not the retrieval/breadth improvements, even though the retrieval improvements are individually easier and worth doing alongside it.
