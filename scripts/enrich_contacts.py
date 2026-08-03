"""Add a contact card to every target we have already researched.

The existing reports were produced before the contact agent existed, when contact hunting
was an instruction in the research supervisor's prompt that got skipped: of 64 reports, 3
carry an email and 1 a phone number. This re-runs only the contact agent over them, so the
research is reused rather than paid for again.

Resumable by design: progress is the set of contact cards already on disk, so a run killed
by a quota wall picks up exactly where it stopped.

Usage:
    python scripts/enrich_contacts.py                 # everything still missing a card
    python scripts/enrich_contacts.py --limit 5       # try a few first
    python scripts/enrich_contacts.py --delay 30      # slower, if quota is tight
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

load_dotenv()

from agents.research.contact_agent import contact_agent  # noqa: E402
from agents.research.state import ContactCard  # noqa: E402
from orchestrator.graph import _render_contact_card  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent.parent / "src/agents/outreach/reports"
RESEARCH_SUFFIX = " - General Lead Research Report.txt"
CARD_SUFFIX = " - Contact Card.txt"

# Sites that host pages *about* an organization without being its own domain. Picking one
# of these as the target's website sends the crawler to the wrong place entirely.
AGGREGATORS = (
    "linkedin.", "wikipedia.", "youtube.", "twitter.", "x.com", "facebook.",
    "instagram.", "crunchbase.", "zaubacorp.", "tofler.", "bloomberg.", "reuters.",
    "economictimes.", "timesofindia.", "business-standard.", "moneycontrol.",
    "indiamart.", "justdial.", "glassdoor.", "indeed.", "medium.", "github.",
    "google.", "youtu.be", "tracxn.", "zoominfo.", "rocketreach.",
)

URL_RE = re.compile(r"https?://([A-Za-z0-9.-]+)")


def guess_website(report_text: str) -> str:
    """Guess the target's own site as the most-cited non-aggregator domain in its report.

    Passing a URL lets the contact agent start with a crawl, which costs no summarization
    calls, instead of spending a search round rediscovering a site we already have.
    """
    domains = [
        d.lower() for d in URL_RE.findall(report_text)
        if not any(a in d.lower() for a in AGGREGATORS)
    ]
    if not domains:
        return ""
    return f"https://{Counter(domains).most_common(1)[0][0]}"


def pending_targets(limit: int | None) -> list[tuple[str, Path]]:
    """List researched targets that have no contact card yet."""
    targets = []
    for report in sorted(REPORTS_DIR.glob(f"*{RESEARCH_SUFFIX}")):
        name = report.name[: -len(RESEARCH_SUFFIX)]
        if (REPORTS_DIR / f"{name}{CARD_SUFFIX}").exists():
            continue
        targets.append((name, report))
    return targets[:limit] if limit else targets


async def enrich_one(name: str, report_path: Path) -> ContactCard | None:
    """Run the contact agent for one already-researched target."""
    report_text = report_path.read_text(encoding="utf-8", errors="ignore")
    result = await contact_agent.ainvoke(
        {
            "target_name": name,
            "target_website": guess_website(report_text),
            # The research we already paid for, as seed context: it names people and roles
            # the finder would otherwise have to rediscover.
            "target_context": report_text[:3000],
            "entity_type": "",
        },
        config={"recursion_limit": 50, "configurable": {}},
    )
    return result.get("contact_card")


async def main(limit: int | None, delay: float) -> int:
    """Enrich every pending target, writing each card as it completes."""
    targets = pending_targets(limit)
    if not targets:
        print("Every researched target already has a contact card.")
        return 0

    print(f"{len(targets)} target(s) to enrich. Cards land in {REPORTS_DIR}\n")
    stats = Counter()

    for i, (name, report_path) in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {name}", flush=True)
        try:
            card = await enrich_one(name, report_path)
        except Exception as e:
            logging.warning(f"  failed: {e}")
            stats["failed"] += 1
            card = None

        if card is not None:
            (REPORTS_DIR / f"{name}{CARD_SUFFIX}").write_text(
                _render_contact_card(card), encoding="utf-8"
            )
            stats["done"] += 1
            stats["with_email"] += bool(card.emails)
            stats["with_phone"] += bool(card.phones)
            stats["unreachable"] += card.best_route == "unreachable"
            print(
                f"  route={card.best_route} emails={len(card.emails)} "
                f"phones={len(card.phones)} linkedin={len(card.linkedin_urls)}",
                flush=True,
            )

        # Spread calls out: the free-tier request limit is per minute, and the ladder only
        # helps once a bucket is already empty - pacing keeps us off the wall in the first place.
        if i < len(targets) and delay:
            await asyncio.sleep(delay)

    print(
        f"\nDone. {stats['done']} card(s) written, {stats['failed']} failed.\n"
        f"  with an email: {stats['with_email']}\n"
        f"  with a phone:  {stats['with_phone']}\n"
        f"  unreachable:   {stats['unreachable']}"
    )
    return 0 if not stats["failed"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only process this many targets")
    parser.add_argument("--delay", type=float, default=20.0, help="Seconds between targets")
    args = parser.parse_args()
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("GOOGLE_API_KEY is not set - check your .env")
    sys.exit(asyncio.run(main(args.limit, args.delay)))
