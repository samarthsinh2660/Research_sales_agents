"""PACE outreach CLI - one entry point for finding, researching and drafting.

Everything the pipeline knows about a lead is visible from here: score, partner track,
why it scored that way, how to reach them, and whether the drafted email survived its
grounding check. Loose report files on disk were the previous answer to "where do I see
my leads", which is not an answer.

Commands
    find     <query>   discover organizations from a search query or a listing URL
    run      <target>  research a target, then draft the email (see --intent)
    leads              table of everything researched so far
    show     <name>    full report for one lead
    status             progress of a batch against a source URL

Examples:
    python scripts/pace.py find "IT companies in Vadodara"
    python scripts/pace.py run "Shital Infotech"          # researches + drafts the email
    python scripts/pace.py run https://cio.economictimes.indiatimes.com/annual-conclave2025
    python scripts/pace.py leads --min-score 7
    python scripts/pace.py show "Rishabh Software"
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
load_dotenv(dotenv_path=REPO / ".env")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
for _noisy in (
    "httpx", "langsmith", "google_genai", "urllib3", "langchain_core.tracers",
    # Emits "Error in LangChainTracer.on_llm_error callback: No indexed run ID ..." for
    # every retried call. It reports that LangSmith could not index a trace, never that
    # the research failed - but it buries the one line that matters under dozens of its own.
    "langchain_core.callbacks.manager",
    # The provider logs its own "Retrying ... as it raised 429" line per attempt; the CLI
    # reports rate limiting once, in a form that says what to do about it.
    "google.genai._api_client", "google_genai.models",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# LangSmith tracing is what produces the "No indexed run ID" storm, and a batch run has no
# use for it. Opt back in with LANGSMITH_TRACING=true if a trace is actually wanted.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.research.deep_researcher import deep_researcher  # noqa: E402
from orchestrator.graph import build_unified_agent  # noqa: E402
from orchestrator.state import Target  # noqa: E402
from orchestrator.targets import _targets_from_page, parse_inline_list  # noqa: E402

console = Console()

REPORTS_DIR = REPO / "src" / "agents" / "outreach" / "reports"
CACHE_DIR = REPO / ".batch_cache"
LEDGER = CACHE_DIR / "leads.json"
REPORT_SUFFIX = " - General Lead Research Report.txt"

# Captured from node updates as they stream past, because finish_target resets them all.
CAPTURED_FIELDS = {
    "lead_score", "lead_track", "lead_qualified", "lead_reasoning", "lead_angle",
    "contact_route", "recipient_name", "email_grounded", "unsupported_claims", "failures",
}


def build_config(args, intent: str | None = None) -> dict:
    """Assemble runtime config from CLI flags.

    Models are deliberately not set here. Configuration already picks them per role, and
    overriding every role from the CLI meant a single flag silently replaced a considered
    default with a weaker model - which is exactly what happened: the CLI pinned
    gemini-3.1-flash-lite over the configured gemini-3.5-flash and made every run worse.
    Change models in configuration.py or the environment, where the choice is recorded.
    """
    configurable = {
        "intent": intent or getattr(args, "intent", "draft"),
        "max_targets": getattr(args, "max_targets", 200),
        # Clarifying questions are only useful when someone is watching the terminal.
        "allow_clarification": getattr(args, "ask", False),
        "require_qualification": not getattr(args, "force", False),
    }
    return {"recursion_limit": 100, "configurable": configurable}


def add_run_flags(parser) -> None:
    """Add the flags shared by every command that actually runs the graph."""
    parser.add_argument("--intent", default="draft",
                        choices=["research", "qualify", "draft", "send"])
    parser.add_argument("--threshold", type=float, help="qualification score cutoff (default 7)")
    parser.add_argument("--force", action="store_true",
                        help="draft even if the lead scores below the threshold")
    parser.add_argument("--ask", action="store_true",
                        help="let the agent ask clarifying questions in the terminal")
    parser.add_argument("--depth", type=int, default=2, help="research iterations per target")
    parser.add_argument("--parallel", type=int, default=1, help="concurrent research units")
    parser.add_argument("--max-targets", type=int, default=200)


def answer_interrupt(payload) -> object:
    """Put the agent's question to the user and return their answer.

    The graph pauses through interrupt() whenever it needs a human - approving a send, or
    a clarifying question during research. Without this the run would either stall or,
    worse, proceed on an assumption nobody made.
    """
    value = payload.value if hasattr(payload, "value") else payload

    if isinstance(value, dict) and value.get("action") == "confirm_send":
        console.print()
        console.print(Panel(
            f"[bold]To:[/bold] {value.get('recipient')}\n"
            f"[bold]Target:[/bold] {value.get('target')}\n\n{value.get('email', '')}",
            title="[yellow]Send this email?[/yellow]", border_style="yellow",
        ))
        reply = console.input("[bold yellow]send? [y/N][/bold yellow] ").strip().lower()
        return {"approved": reply in ("y", "yes")}

    question = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
    console.print()
    console.print(Panel(question, title="[cyan]The agent is asking[/cyan]", border_style="cyan"))
    return console.input("[bold cyan]your answer:[/bold cyan] ").strip()


##########################
# Ledger
##########################

def load_ledger() -> dict:
    """Everything researched so far, keyed by target name."""
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {}


def save_lead(name: str, record: dict) -> None:
    """Record one lead's outcome, so `leads` can show it without re-running anything."""
    LEDGER.parent.mkdir(exist_ok=True)
    ledger = load_ledger()
    ledger[name] = {**ledger.get(name, {}), **record}
    LEDGER.write_text(json.dumps(ledger, indent=2))


def report_path(name: str) -> Path:
    """Where finish_target writes this target's research report."""
    safe = re.sub(r"[^\w\-. ]", "_", name).strip()[:60]
    return REPORTS_DIR / f"{safe}{REPORT_SUFFIX}"


def save_discovery_report(query: str, report: str) -> Path | None:
    """Write a discovery report to disk, so the pass that produced it is not wasted.

    Discovery runs a full research pass. Reading names out of the result and dropping the
    report meant paying for research and keeping none of it, then researching the same
    organizations again one by one.
    """
    if not report.strip():
        return None
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-. ]", "_", query).strip()[:60]
    path = REPORTS_DIR / f"{safe}{REPORT_SUFFIX}"
    path.write_text(report)
    return path


async def targets_for_url(url: str, config: dict, request: str = "") -> list[Target]:
    """Targets for a listing page, extracted once and cached.

    Re-extracting on every run is not free and not stable: extraction quality follows
    whichever model is configured, so a resume with --local reduced a 123-target page to
    2 and then found nothing to resume from, because the names no longer matched the
    reports on disk. The page's target list is a fact about the page, not about the run,
    so it is captured once and reused.

    The request is part of the cache key: extraction now returns only what was asked for,
    so the same page yields a different list for "colleges" than for "recruiters", and
    keying on the URL alone would serve one answer to the other question.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    stem = re.sub(r"[^\w]", "_", url)[:80]
    if request.strip():
        stem += "_" + hashlib.sha1(request.strip().lower().encode()).hexdigest()[:8]
    cache = CACHE_DIR / f"{stem}.json"

    if cache.exists():
        targets = [Target(**t) for t in json.loads(cache.read_text())]
        console.print(f"[dim]{len(targets)} targets (cached)[/dim]")
        return targets

    targets = await _targets_from_page(url, config, request)
    if targets:
        cache.write_text(json.dumps([t.model_dump() for t in targets], indent=2))
    return targets


##########################
# Commands
##########################

async def discover(query: str, config: dict) -> tuple[list[Target], str]:
    """Find organizations matching a request, using the research agent.

    A single search page is not an answer to "the top 10 engineering colleges": the page
    that ranks first is usually one college's own marketing page, which lists itself at
    number one and pads the rest with its recruiters. The research agent searches several
    sources and reconciles them, which is the job it already does well - so discovery asks
    it, rather than reimplementing a worse version of it.

    Args:
        query: What to find, e.g. "top 10 engineering colleges in Dehradun"
        config: Runtime configuration

    Returns:
        (targets named in the findings, the research report itself)
    """
    from orchestrator.targets import _extract_targets

    console.print(f"[dim]researching: {query}[/dim]")
    result = await deep_researcher.ainvoke(
        {"messages": [("user",
            f"{query}. For each organization give its name, what it is, and how to contact "
            "it - a published email, the relevant office, or its contact page. Only include "
            "organizations that match the request."
        )]},
        config,
    )
    report = result.get("final_report", "")
    if not report.strip():
        return [], ""

    targets = await _extract_targets(report, query, config, query)
    return targets, report


async def cmd_find(args) -> tuple[list[Target], str]:
    """Discover organizations, show them, and return them with any research report."""
    config = build_config(args, intent="research")
    report = ""

    if args.query.startswith("http"):
        targets = await targets_for_url(args.query, config, "")
    else:
        targets, report = await discover(args.query, config)

    if not targets:
        console.print("[red]nothing found[/red]")
        return [], report

    table = Table(title=f"{len(targets)} found", header_style="bold cyan")
    table.add_column("#", width=4)
    table.add_column("Name")
    table.add_column("What it is", overflow="fold")
    for i, t in enumerate(targets[:args.limit or len(targets)], 1):
        table.add_row(str(i), t.name, (t.context or "")[:60])
    console.print(table)
    return targets, report


def configured_research_model() -> str:
    """Return the research model the code is configured to use, not a CLI override."""
    from agents.research.configuration import Configuration
    return Configuration().research_model


async def check_model_available(model: str) -> str | None:
    """Return a human-readable reason the model cannot be used, or None if it is fine.

    Checked once before a batch rather than discovered target by target. An exhausted
    daily quota still "works" - every call retries, falls back, and limps on at a fraction
    of the speed - so without an upfront check a run looks alive while producing far worse
    research than it should.
    """
    if model.startswith("ollama:"):
        return None
    try:
        from langchain.chat_models import init_chat_model
        await init_chat_model(
            model, api_key=os.getenv("GOOGLE_API_KEY"), max_retries=0
        ).ainvoke("ok")
        return None
    except Exception as e:
        text = str(e)
        if "RESOURCE_EXHAUSTED" in text or "429" in text:
            return "daily quota exhausted"
        if "UNAVAILABLE" in text or "503" in text:
            return "model temporarily unavailable"
        return type(e).__name__


async def run_target(graph, target: Target, config: dict) -> dict:
    """Run one target to completion, answering any question the agent raises.

    Node updates are read as they stream rather than from the final state: finish_target
    clears every per-target field on the way out, so the returned state has an empty score
    and route by design. The reset streams past too, which is why that node is skipped.
    """
    captured: dict = {}
    payload: object = {
        "targets": [target],
        "messages": [("user", f"Research {target.name}")],
    }

    while True:
        interrupted = None
        async for update in graph.astream(payload, config, stream_mode="updates"):
            for node_name, node_output in update.items():
                if node_name == "__interrupt__":
                    interrupted = node_output
                    continue
                if node_name == "finish_target" or not isinstance(node_output, dict):
                    continue
                captured.update(
                    {k: v for k, v in node_output.items() if k in CAPTURED_FIELDS}
                )
        if not interrupted:
            return captured
        # Resume the same thread with the human's answer.
        first = interrupted[0] if isinstance(interrupted, (list, tuple)) else interrupted
        payload = Command(resume=answer_interrupt(first))


async def cmd_run(args, targets: list[Target] | None = None) -> None:
    """Research targets end to end, one graph invocation each so progress is checkpointed.

    Args:
        args: Parsed CLI flags
        targets: Already-resolved targets, so discovery does not have to be redone
    """
    config = build_config(args)

    if targets is None:
        # A URL anywhere in the line wins: "find the people from <url>" names the page to
        # use, and treating the sentence as a list of names would invent targets from words.
        url = url_in(args.target)
        if url:
            targets = await targets_for_url(url, config, "")
        else:
            targets = [Target(name=n, source="inline") for n in parse_inline_list(args.target)]

    if args.resume:
        before = len(targets)
        targets = [t for t in targets if not report_path(t.name).exists()]
        console.print(f"[dim]resuming: {before - len(targets)} already done[/dim]")

    targets = targets[: args.limit] if args.limit else targets
    if not targets:
        console.print("[green]nothing to do[/green]")
        return

    notes = []
    if args.force:
        notes.append("ignoring score gate")
    if args.ask:
        notes.append("clarifying questions on")
    suffix = f" ({', '.join(notes)})" if notes else ""
    console.print(f"[bold]{len(targets)} target(s) - "
                  f"{MODE_LABELS.get(args.intent, args.intent)}{suffix}[/bold]")

    problem = await check_model_available(configured_research_model())
    if problem:
        console.print(Panel(
            f"[bold]{configured_research_model()}[/bold]\n{problem}\n\n"
            "It would still run - every call retries, falls back to a weaker model and\n"
            "limps on - but the research would be far worse than it should be.\n\n"
            "[dim]Wait for the daily quota to reset, or change the model in\n"
            "src/agents/research/configuration.py[/dim]",
            title="[red]model not usable right now[/red]", border_style="red",
        ))
        reply = console.input("[bold yellow]run anyway? [y/N][/bold yellow] ").strip().lower()
        if reply not in ("y", "yes"):
            return
    console.print()

    # A checkpointer is what makes interrupt() resumable, so send approval and clarifying
    # questions can pause the run and come back with the operator's answer.
    graph = build_unified_agent(checkpointer=InMemorySaver())

    started = time.time()
    for i, target in enumerate(targets, 1):
        console.print(f"[cyan][{i}/{len(targets)}][/cyan] {target.name} ...", end=" ")
        try:
            run_config = {
                **config,
                "configurable": {**config["configurable"], "thread_id": f"{target.name}-{i}"},
            }
            result = await run_target(graph, target, run_config)
            failures = result.get("failures", [])
            if failures:
                console.print(f"[red]FAILED[/red] {failures[-1][:90]}")
                save_lead(target.name, {"status": "failed", "error": failures[-1][:300]})
                continue

            record = {
                "status": "researched",
                "score": result.get("lead_score", ""),
                "track": result.get("lead_track", ""),
                "qualified": result.get("lead_qualified", False),
                "reasoning": result.get("lead_reasoning", ""),
                "angle": result.get("lead_angle", ""),
                "route": result.get("contact_route", ""),
                "recipient": result.get("recipient_name", ""),
                "grounded": result.get("email_grounded", None),
                "unsupported": result.get("unsupported_claims", []),
            }
            save_lead(target.name, record)
            flag = "" if record["grounded"] is not False else " [yellow]ungrounded[/yellow]"
            console.print(f"[green]done[/green] {record['score'] or '-'} {record['track']}{flag}")
        except Exception as e:
            console.print(f"[red]ERROR[/red] {str(e)[:90]}")
            save_lead(target.name, {"status": "error", "error": str(e)[:300]})

    console.print(f"\n[dim]{(time.time()-started)/60:.1f} min. See: pace.py leads[/dim]")


def cmd_leads(args) -> None:
    """Table of every lead researched so far."""
    ledger = load_ledger()
    if not ledger:
        console.print("[yellow]no leads yet - run: pace.py run <target>[/yellow]")
        return

    rows = []
    for name, r in ledger.items():
        try:
            score = float(r.get("score") or 0)
        except ValueError:
            score = 0.0
        if args.min_score and score < args.min_score:
            continue
        if args.qualified and not r.get("qualified"):
            continue
        rows.append((score, name, r))
    rows.sort(key=lambda x: -x[0])

    table = Table(title=f"{len(rows)} leads", header_style="bold cyan")
    for col, style in (("Score", "bold"), ("Lead", ""), ("Track", ""), ("Route", ""),
                       ("Recipient", ""), ("Email", "")):
        table.add_column(col, style=style)

    for score, name, r in rows:
        if r.get("status") in ("failed", "error"):
            table.add_row("-", name, "[red]failed[/red]", "", "", "")
            continue
        # Passing the grounding check is not the same as a draft existing: without a
        # published address there is nothing to draft to, and we never invent one.
        grounded = r.get("grounded")
        has_address = r.get("route") in ("personal_email", "role_inbox")
        if grounded is False:
            email_cell = "[yellow]blocked[/yellow]"
        elif grounded and has_address:
            email_cell = "[green]drafted[/green]"
        elif grounded:
            email_cell = "[dim]no address[/dim]"
        else:
            email_cell = "-"
        colour = "green" if score >= 7 else "yellow" if score >= 5 else "white"
        table.add_row(
            f"[{colour}]{r.get('score') or '-'}[/{colour}]", name, r.get("track", ""),
            r.get("route", ""), r.get("recipient", ""), email_cell,
        )
    console.print(table)
    console.print("[dim]blocked = email made claims the research did not support[/dim]")


def cmd_show(args) -> None:
    """Everything known about one lead, including the report."""
    ledger = load_ledger()
    match = next((n for n in ledger if args.name.lower() in n.lower()), None)
    if not match:
        console.print(f"[red]no lead matching '{args.name}'[/red]")
        return

    r = ledger[match]
    body = [
        f"[bold]Score[/bold]      {r.get('score','-')}   [bold]Track[/bold] {r.get('track','-')}",
        f"[bold]Qualified[/bold]  {r.get('qualified')}",
        f"[bold]Route[/bold]      {r.get('route','-')} -> {r.get('recipient') or '(no name found)'}",
        "",
        f"[bold]Why[/bold]        {r.get('reasoning','-')}",
        f"[bold]Angle[/bold]      {r.get('angle') or '(none strong enough)'}",
    ]
    if r.get("unsupported"):
        body += ["", "[yellow]Email blocked - unsupported claims:[/yellow]"]
        body += [f"  - {c}" for c in r["unsupported"]]
    console.print(Panel("\n".join(body), title=match, border_style="cyan"))

    path = report_path(match)
    if path.exists() and args.full:
        console.print(path.read_text())
    elif path.exists():
        console.print(f"[dim]{path.name} - add --full to print it[/dim]")


def cmd_status(args) -> None:
    """How far a batch has progressed, judged by reports actually on disk."""
    ledger = load_ledger()
    done = sum(1 for r in ledger.values() if r.get("status") == "researched")
    failed = sum(1 for r in ledger.values() if r.get("status") in ("failed", "error"))
    blocked = sum(1 for r in ledger.values() if r.get("grounded") is False)
    reports = len(list(REPORTS_DIR.glob(f"*{REPORT_SUFFIX}"))) if REPORTS_DIR.exists() else 0
    console.print(Panel(
        f"researched   {done}\nfailed       {failed}\n"
        f"email blocked {blocked}\nreports on disk {reports}",
        title="status", border_style="cyan",
    ))


##########################
# Interactive session
##########################

# "intent" is the config field name, but it tells a user nothing about what will happen.
# The CLI talks in outcomes instead and translates at the edge, so the graph keeps one
# vocabulary and the person at the terminal gets another.
MODE_LABELS = {
    "research": "research only - no email written",
    "qualify": "research + score the fit",
    "draft": "write the email, save as draft - nothing sent",
    "send": "write and send - asks you first",
}

# What someone is likely to type for each mode, rather than the internal name.
MODE_ALIASES = {
    "research": "research", "info": "research", "learn": "research", "only": "research",
    "qualify": "qualify", "score": "qualify",
    "draft": "draft", "email": "draft", "mail": "draft", "write": "draft",
    "send": "send",
}


class Settings(argparse.Namespace):
    """Mutable session settings, shaped like parsed args so build_config takes either."""

    def __init__(self):
        """Start from the defaults a session is most likely to want."""
        super().__init__(
            intent="draft", threshold=None, force=False,
            ask=False, depth=2, parallel=1, max_targets=200, limit=0, resume=False,
            min_score=0, qualified=False, full=False,
        )

    def as_rows(self):
        """Return the current settings, ordered for display."""
        return [
            ("mode", MODE_LABELS.get(self.intent, self.intent), "what the agent does with each target"),
            ("threshold", self.threshold if self.threshold is not None else "7.0 (default)", "score needed to qualify"),
            ("force", self.force, "draft even below the threshold"),
            ("ask", self.ask, "let the agent ask clarifying questions"),
            ("depth", self.depth, "research iterations per target"),
            ("parallel", self.parallel, "concurrent research units"),
        ]


HELP = """
[bold cyan]WHAT YOU CAN TYPE[/bold cyan]

  [bold]A company name[/bold] - runs the pipeline on it
     [dim]Rishabh Software[/dim]

  [bold]Several names[/bold], comma separated
     [dim]Rishabh Software, Prakash Software, Spaculus[/dim]

  [bold]A link[/bold] - pulls every organization and person off the page, then runs them
     [dim]https://cio.economictimes.indiatimes.com/annual-conclave2025[/dim]
     Works on event pages, speaker lists and company directories.

  [bold]A request to find organizations[/bold] - discovers them, then offers to run them
     [dim]find all the colleges in Dehradun we can sell PACE to[/dim]
     [dim]list IT companies in Vadodara[/dim]
     Start with find/list/search and say where. One step, no link to copy.

  [bold]@name[/bold] - show one lead you already researched
     [dim]@rishabh[/dim]

[bold cyan]WHAT IT DOES WITH EACH ONE - /mode[/bold cyan]

  Each mode includes everything above it.

  [bold]research[/bold]   find out who they are and how to reach them. No email written.
  [bold]score[/bold]      also score how well they fit PACE, and pick the right track
  [bold]email[/bold]      also write the email and save it as a Gmail draft.
             [dim]Nothing is sent. This is the default.[/dim]
  [bold]send[/bold]       also send it - shows you the full email and asks first

  [dim]/mode email[/dim]   [dim](or research / score / send)[/dim]

[bold cyan]SETTINGS - change any time, they stick[/bold cyan]

  /config              show everything and what it means
  /mode <what>         research | score | email | send
  /threshold <n>       score needed to qualify (default 7)
  /force               write the email even if the lead scores low
  /depth <n>           research rounds per target. 1 = fast/shallow,
                       2 = default, 3+ = deeper and much more expensive
  /ask                 let the agent ask you questions while it works

[bold cyan]OTHER COMMANDS[/bold cyan]

  /find <what and where>   discover companies
                           [dim]/find IT companies in Vadodara[/dim]
  /leads [minscore]        table of everything researched
  /status                  counts
  /help  /exit

[bold cyan]TYPICAL SESSIONS[/bold cyan]

  [bold]Just tell me about them[/bold]
     /mode research
     Rishabh Software

  [bold]Find who to contact and write the mail[/bold]
     /mode email
     Rishabh Software
     [dim]-> finds the contact, writes the email, saves a Gmail draft[/dim]

  [bold]Work a whole event page[/bold]
     /mode research
     https://some-conference.com/speakers

  [bold]Prospect a city, start to finish[/bold]
     find engineering colleges in Dehradun
     [dim]-> lists them, asks "research these now?", then runs them[/dim]

[dim]An email is only drafted when a real published address was found. If none was
published you'll see "no address" - the agent never invents one.[/dim]
"""


def show_config(settings: Settings) -> None:
    """Print current settings as a table."""
    table = Table(title="settings", header_style="bold cyan")
    table.add_column("setting")
    table.add_column("value", style="bold")
    table.add_column("meaning", style="dim")
    for name, value, meaning in settings.as_rows():
        table.add_row(name, str(value), meaning)
    console.print(table)
    console.print("[dim]change with /mode email, /threshold 6, /force[/dim]")


# A request to discover organizations rather than a list of names to research, so
# "find all the colleges we can sell PACE to" works as typed instead of being split on
# commas into nonsense target names. The \b matters: it keeps a company actually called
# "Finder Technologies" being treated as a name, not a search.
_DISCOVERY_VERB = re.compile(r"^\s*(find|search|discover|list|get|show)\b", re.IGNORECASE)

# A place to search. Without one the search has nothing to narrow on, so the user is asked.
_LOCATION_HINT = re.compile(r"\b(in|near|around|across|from|at)\b\s+\w", re.IGNORECASE)


_URL_ANYWHERE = re.compile(r"https?://\S+")


def url_in(line: str) -> str | None:
    """Return the first URL mentioned anywhere in the line, if there is one."""
    match = _URL_ANYWHERE.search(line or "")
    return match.group(0) if match else None


def looks_like_discovery(line: str) -> bool:
    """Whether a plain-text line is asking to find organizations, not naming them.

    A line naming a URL is never discovery, wherever the URL sits. "find the people from
    <url>" means use that page, and routing it to a search because of the leading verb
    would quietly answer a different question than the one asked.

    Deliberately does not require a location: "find all the colleges we can sell PACE to"
    is plainly a search, and refusing it because it lacks "in Dehradun" would send it down
    the name-parsing path and produce nonsense targets. The location is asked for instead.
    """
    if url_in(line) or not _DISCOVERY_VERB.match(line):
        return False
    return len(line.split()) >= 3


def needs_location(line: str) -> bool:
    """Whether a discovery request has no place to search in."""
    return not _LOCATION_HINT.search(line)


async def find_and_run(query: str, settings: Settings) -> None:
    """Discover organizations and research them, as one step.

    No confirmation between the two: asking "run these now?" adds a decision without
    adding information, since the request already said what was wanted and extraction
    now returns only that. Ctrl-C stops a run that is going the wrong way.
    """
    if needs_location(query):
        where = console.input(
            "[bold cyan]where should I look?[/bold cyan] [dim](e.g. Dehradun, Uttarakhand)[/dim] "
        ).strip()
        if not where:
            console.print("[yellow]need a place to search - try: find colleges in Dehradun[/yellow]")
            return
        query = f"{query} in {where}"

    args = Settings()
    args.__dict__.update(settings.__dict__)
    args.query, args.limit = query, 30

    targets, report = await cmd_find(args)
    if not targets:
        return

    if settings.intent == "research":
        # The discovery pass already researched these and wrote a report covering all of
        # them. Researching each one again would pay roughly ten times over for the same
        # question, and the report - the thing actually asked for - used to be discarded.
        saved = save_discovery_report(query, report)
        if saved:
            # The path, not a `show` command: `show` reads the per-lead ledger, and a
            # discovery report covers many organizations rather than being one lead.
            console.print(f"\n[green]report saved:[/green] {saved}")
            console.print(f"[dim]{len(report.splitlines())} lines - open it, or: cat \"{saved}\"[/dim]")
        else:
            console.print("[yellow]no report was produced[/yellow]")
        return

    # Deeper modes need per-organization work - each one's own contact, score and email -
    # so here the fan-out earns its cost.
    run_args = Settings()
    run_args.__dict__.update(settings.__dict__)
    run_args.resume = True
    await cmd_run(run_args, targets=targets)


async def handle_slash(line: str, settings: Settings) -> bool:
    """Run one slash command. Returns False when the session should end."""
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit", "q"):
        return False
    if cmd in ("help", "h", "?"):
        console.print(HELP)
    elif cmd == "config":
        show_config(settings)
    elif cmd in ("mode", "intent"):
        chosen = MODE_ALIASES.get(rest.lower())
        if chosen:
            settings.intent = chosen
            console.print(f"[green]mode = {MODE_LABELS[chosen]}[/green]")
        else:
            console.print("[red]usage: /mode research | score | email | send[/red]")
            for name, label in MODE_LABELS.items():
                console.print(f"  [bold]{name:9}[/bold] [dim]{label}[/dim]")
    elif cmd == "threshold":
        try:
            settings.threshold = float(rest)
            console.print(f"[green]threshold = {settings.threshold}[/green]")
        except ValueError:
            console.print("[red]usage: /threshold 6.5[/red]")
    elif cmd == "force":
        settings.force = not settings.force
        console.print(f"[green]force = {settings.force}[/green]")
    elif cmd == "ask":
        settings.ask = not settings.ask
        console.print(f"[green]ask = {settings.ask}[/green]")
    elif cmd == "depth":
        try:
            settings.depth = int(rest)
            console.print(f"[green]depth = {settings.depth}[/green]")
        except ValueError:
            console.print("[red]usage: /depth 3[/red]")
    elif cmd == "find":
        if not rest:
            console.print("[red]usage: /find engineering colleges in Dehradun[/red]")
        else:
            await find_and_run(rest, settings)
    elif cmd == "leads":
        args = Settings()
        args.min_score = float(rest) if rest else 0
        cmd_leads(args)
    elif cmd == "status":
        cmd_status(settings)
    else:
        console.print(f"[red]unknown command: /{cmd}[/red]  [dim]try /help[/dim]")
    return True


async def cmd_repl(_args) -> None:
    """Run the interactive session.

    A persistent prompt rather than one process per action: settings stay put between
    targets, so changing model or intent does not mean retyping flags every time.
    """
    settings = Settings()
    console.print(Panel(
        "[bold]PACE outreach[/bold]\n\n"
        "Type a [bold]company name[/bold], a [bold]list of names[/bold], or a [bold]link[/bold] to an event or "
        "directory page.\n"
        f"Right now I will: [bold]{MODE_LABELS[settings.intent]}[/bold]\n"
        "[dim]change that with /mode[/dim]\n\n"
        "[dim]/help for everything you can do  ·  /config for settings  ·  /exit to quit[/dim]",
        border_style="cyan",
    ))

    while True:
        try:
            line = console.input(f"\n[bold cyan]pace[/bold cyan] [dim]({settings.intent})[/dim] › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if not line:
            continue

        if line.startswith("/"):
            if not await handle_slash(line, settings):
                console.print("[dim]bye[/dim]")
                return
            continue

        if line.startswith("@"):
            args = Settings()
            args.name, args.full = line[1:].strip(), False
            cmd_show(args)
            continue

        if looks_like_discovery(line):
            try:
                await find_and_run(line, settings)
            except Exception as e:
                console.print(f"[red]error:[/red] {str(e)[:200]}")
            continue

        args = Settings()
        args.__dict__.update(settings.__dict__)
        args.target = line
        try:
            await cmd_run(args)
        except Exception as e:
            console.print(f"[red]error:[/red] {str(e)[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PACE outreach pipeline")
    # Not required: bare `pace.py` opens the interactive session, which is the usual way in.
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("find", help="discover organizations")
    p.add_argument("query", help="search query, or a listing page URL")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--max-targets", type=int, default=200)

    p = sub.add_parser("run", help="research targets end to end")
    p.add_argument("target", help="name(s), or a listing page URL")
    p.add_argument("--limit", type=int, default=0, help="stop after N targets")
    p.add_argument("--resume", action="store_true", help="skip targets that already have a report")
    add_run_flags(p)

    # The common request - "research this company, find who to contact, write me the mail" -
    # is `run --intent draft --force`, so it gets its own verb rather than three flags.
    p = sub.add_parser("outreach", help="research a company, find the contact, draft the email")
    p.add_argument("target", help="company name(s), or a listing page URL")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    add_run_flags(p)
    p.set_defaults(intent="draft", force=True)

    p = sub.add_parser("leads", help="table of researched leads")
    p.add_argument("--min-score", type=float, default=0)
    p.add_argument("--qualified", action="store_true")

    p = sub.add_parser("show", help="full detail for one lead")
    p.add_argument("name")
    p.add_argument("--full", action="store_true", help="print the whole research report")

    sub.add_parser("status", help="batch progress")
    sub.add_parser("chat", help="interactive session (also the default)")

    args = parser.parse_args()
    if args.command in (None, "chat"):
        asyncio.run(cmd_repl(args))
    elif args.command in ("find", "run", "outreach"):
        handler = cmd_find if args.command == "find" else cmd_run
        asyncio.run(handler(args))
    else:
        {"leads": cmd_leads, "show": cmd_show, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
