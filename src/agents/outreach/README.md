# Sales Outreach Agent

Turns a list of leads into researched, qualified, personalized outreach — and writes the
outcome back to your CRM.

Research is delegated entirely to the shared `open_deep_research` core; this graph owns
only the sales-specific work. Originally derived from
[kaymen99/sales-outreach-automation-langgraph](https://github.com/kaymen99/sales-outreach-automation-langgraph),
since substantially rewritten. See [`docs/idea.md`](../../docs/idea.md) at the repo root
for the overall architecture.

## Workflow

Per lead, in order:

1. **`get_new_leads`** — load leads from the configured source (Google Sheets / Airtable / HubSpot)
2. **`run_shared_research`** — research the person and their company via `open_deep_research`
3. **`check_research_sufficiency`** — gate on whether the research is substantial enough to
   pitch on. If not, retry once with a query aimed at the specific gap, then give up and
   flag the lead rather than sending a weak, generic email
4. **`score_lead`** — score partnership fit; below threshold routes straight to reporting
5. **`generate_custom_outreach_report`** — write and proofread the outreach report, then
   fan out to both material generators in parallel
6. **`generate_personalized_email`** / **`generate_interview_script`** — email (as a Gmail
   draft) and a partnership call script
7. **`save_reports_to_google_docs`** — persist reports locally, and to Google Docs if enabled
8. **`update_CRM`** — write status, score, and links back; reset per-lead state

## Setup

Dependencies and the `.env` file are shared with the rest of the monorepo — install from
the repo root (`pip install -e .`) and see the root `.env.example`. This app uses
`SHEET_ID` (or `AIRTABLE_*` / `HUBSPOT_API_KEY`) plus `GOOGLE_API_KEY`.

Google OAuth (Gmail, Sheets, Docs, Drive) needs a `credentials.json` in this directory:
create a **Desktop app** OAuth client in Google Cloud Console, download it here, and the
first run will produce `token.json`. Both are gitignored.

## Running

```sh
python main.py
```

Or open the **Sales Outreach** graph in LangGraph Studio (`langgraph dev` from the repo root).

Generated reports are written to `reports/` (gitignored — they contain real prospect data).

## Configuration

All behaviour is configured through `SalesConfiguration` (`configuration.py`), resolved
from environment variables first, then the runtime `configurable` dict, then defaults.
Editable in the Studio UI or passed directly:

```python
config = {"configurable": {"lead_loader_type": "airtable", "lead_score_threshold": 8.0}}
```

Key fields: `lead_loader_type`, `lead_score_threshold` (default 7.0), `max_research_retries`,
`send_email_directly`, `save_to_google_docs`, the per-step model fields, and
`sales_fallback_model`.

**Two safety defaults are off deliberately:** `send_email_directly` (emails are created as
Gmail drafts, never auto-sent) and `save_to_google_docs`. Turn them on only once you trust
the generated output.

## Extending

- **New CRM / lead source:** subclass `LeadLoaderBase` (`tools/leads_loader/`), implementing
  `fetch_records` and `update_record`, then add it to `LeadLoaderType` and `_build_lead_loader`
  in `utils.py`.
- **New research capability:** add it to `open_deep_research` as a tool — this graph picks
  it up automatically through the shared core.
- **Changing prompts:** all prompts live in `prompts.py`; `PACE_UTTARAKHAND_CONTEXT` is the
  single source for the programme description shared across them.
