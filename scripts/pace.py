"""PACE outreach CLI - one entry point for finding, researching and drafting.

Everything the pipeline knows about a lead is visible from here: score, partner track,
why it scored that way, how to reach them, and whether the drafted email survived its
grounding check. Loose report files on disk were the previous answer to "where do I see
my leads", which is not an answer.

Commands
    find     <query>   discover organizations from a search query or a listing URL
    run      <target>  research one or more targets end to end
    leads              table of everything researched so far
    show     <name>    full report for one lead
    status             progress of a batch against a source URL

Examples:
    python scripts/pace.py find "IT companies in Vadodara"
    python scripts/pace.py run "Shital Infotech" --intent draft
    python scripts/pace.py run https://cio.economictimes.indiatimes.com/annual-conclave2025
    python scripts/pace.py leads --min-score 7
    python scripts/pace.py show "Rishabh Software"
"""
import argparse
import asyncio
import json
import logging
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
for _noisy in ("httpx", "langsmith", "google_genai", "urllib3", "langchain_core.tracers"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from orchestrator.graph import build_unified_agent  # noqa: E402
from orchestrator.state import Target  # noqa: E402
from orchestrator.targets import _targets_from_page, parse_inline_list  # noqa: E402

console = Console()

REPORTS_DIR = REPO / "src" / "agents" / "outreach" / "reports"
LEDGER = REPO / ".batch_cache" / "leads.json"
REPORT_SUFFIX = " - General Lead Research Report.txt"

CLOUD = "google_genai:gemini-3.1-flash-lite"
LOCAL = "ollama:qwen3:4b"

# Captured from node updates as they stream past, because finish_target resets them all.
CAPTURED_FIELDS = {
    "lead_score", "lead_track", "lead_qualified", "lead_reasoning", "lead_angle",
    "contact_route", "recipient_name", "email_grounded", "unsupported_claims", "failures",
}

MODEL_KEYS = (
    "research_model", "summarization_model", "compression_model", "final_report_model",
    "research_sufficiency_model", "lead_scoring_model", "outreach_report_model", "email_model",
)


def build_config(args, intent: str | None = None) -> dict:
    """Assemble runtime config from CLI flags, pointing every model at cloud or local."""
    model = getattr(args, "model", None) or (LOCAL if getattr(args, "local", False) else CLOUD)
    configurable = {
        "intent": intent or getattr(args, "intent", "research"),
        "max_targets": getattr(args, "max_targets", 200),
        # Clarifying questions are only useful when someone is watching the terminal.
        "allow_clarification": getattr(args, "ask", False),
        "require_qualification": not getattr(args, "force", False),
        "max_concurrent_research_units": getattr(args, "parallel", 1),
        "max_researcher_iterations": getattr(args, "depth", 2),
        **{key: model for key in MODEL_KEYS},
    }
    if getattr(args, "threshold", None) is not None:
        configurable["lead_score_threshold"] = args.threshold
    return {"recursion_limit": 100, "configurable": configurable}


def add_run_flags(parser) -> None:
    """Add the flags shared by every command that actually runs the graph."""
    parser.add_argument("--intent", default="research",
                        choices=["research", "qualify", "draft", "send"])
    parser.add_argument("--model", help="override the model, e.g. google_genai:gemini-3.5-flash")
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


##########################
# Commands
##########################

async def cmd_find(args) -> None:
    """Discover organizations from a listing URL, or from a search query."""
    config = build_config(args, intent="research")
    url = args.query if args.query.startswith("http") else None

    if not url:
        import os

        from tavily import AsyncTavilyClient

        console.print(f"[dim]searching for: {args.query}[/dim]")
        client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        found = await client.search(args.query, max_results=5, search_depth="advanced")
        results = found.get("results", [])
        if not results:
            console.print("[red]nothing found[/red]")
            return
        # Directory pages list many companies at once, so they beat a single company site
        # as a starting point; the first result is the fallback when none look like one.
        url = next(
            (r["url"] for r in results if re.search(r"top|list|director|best", r["title"], re.I)),
            results[0]["url"],
        )
        console.print(f"[dim]best listing page: {url}[/dim]")

    targets = await _targets_from_page(url, config)
    if not targets:
        console.print("[red]no organizations extracted[/red]")
        return

    table = Table(title=f"{len(targets)} found", header_style="bold cyan")
    table.add_column("#", width=4)
    table.add_column("Name")
    table.add_column("Context", overflow="fold")
    for i, t in enumerate(targets[:args.limit], 1):
        table.add_row(str(i), t.name, (t.context or "")[:60])
    console.print(table)
    console.print(f'[dim]research these with: pace.py run "{url}"[/dim]')


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


async def cmd_run(args) -> None:
    """Research targets end to end, one graph invocation each so progress is checkpointed."""
    config = build_config(args)

    if args.target.startswith("http"):
        targets = await _targets_from_page(args.target, config)
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

    where = args.model or ("local qwen3:4b" if args.local else "cloud")
    notes = []
    if args.force:
        notes.append("ignoring score gate")
    if args.ask:
        notes.append("clarifying questions on")
    suffix = f" ({', '.join(notes)})" if notes else ""
    console.print(f"[bold]{len(targets)} target(s), intent={args.intent}, {where}{suffix}[/bold]\n")

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

class Settings(argparse.Namespace):
    """Mutable session settings, shaped like parsed args so build_config takes either."""

    def __init__(self):
        """Start from the defaults a session is most likely to want."""
        super().__init__(
            intent="draft", model=None, local=False, threshold=None, force=False,
            ask=False, depth=2, parallel=1, max_targets=200, limit=0, resume=False,
            min_score=0, qualified=False, full=False,
        )

    def as_rows(self):
        """Return the current settings, ordered for display."""
        return [
            ("intent", self.intent, "how far to go: research / qualify / draft / send"),
            ("model", self.model or ("qwen3:4b (local)" if self.local else "cloud default"), "which model runs everything"),
            ("threshold", self.threshold if self.threshold is not None else "7.0 (default)", "score needed to qualify"),
            ("force", self.force, "draft even below the threshold"),
            ("ask", self.ask, "let the agent ask clarifying questions"),
            ("depth", self.depth, "research iterations per target"),
            ("parallel", self.parallel, "concurrent research units"),
        ]


HELP = """
[bold]Just type a company name[/bold] to research it with the current settings.
  acme corp                      research/outreach one target
  Acme, Globex, Initech          several at once
  https://site.com/speakers      extract targets from a page, then run them

[bold]Slash commands[/bold]
  /config                        show every setting
  /intent <research|qualify|draft|send>
  /model <name>                  e.g. google_genai:gemini-3.5-flash
  /local                         toggle the local qwen3:4b model
  /threshold <n>                 qualification cutoff
  /force                         toggle drafting below the threshold
  /ask                           toggle clarifying questions
  /depth <n>                     research iterations
  /find <query>                  discover companies, e.g. /find IT companies in Vadodara
  /leads [minscore]              table of everything researched
  /status                        counts
  /help  /exit

[bold]@lead[/bold]  show one lead, e.g. [dim]@rishabh[/dim]
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
    console.print("[dim]change with /intent draft, /model ..., /force[/dim]")


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
    elif cmd == "intent":
        if rest in ("research", "qualify", "draft", "send"):
            settings.intent = rest
            console.print(f"[green]intent = {rest}[/green]")
        else:
            console.print("[red]usage: /intent research|qualify|draft|send[/red]")
    elif cmd == "model":
        settings.model = rest or None
        console.print(f"[green]model = {settings.model or 'default'}[/green]")
    elif cmd == "local":
        settings.local = not settings.local
        settings.model = None
        console.print(f"[green]local = {settings.local}[/green]")
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
            console.print("[red]usage: /find IT companies in Vadodara[/red]")
        else:
            args = Settings()
            args.query, args.limit = rest, 30
            await cmd_find(args)
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
        "[bold]PACE outreach[/bold]\n"
        "Type a company name to research it. [dim]/help for commands, /exit to quit.[/dim]",
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

        args = Settings()
        args.__dict__.update(settings.__dict__)
        args.target = line
        try:
            await cmd_run(args)
        except Exception as e:
            console.print(f"[red]error:[/red] {str(e)[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PACE outreach pipeline")
    parser.add_argument("--local", action="store_true", help="use local qwen3:4b instead of cloud")
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
