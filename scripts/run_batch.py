"""Batch-research every target on a listing page, resumably.

Drives the unified agent one target per invocation rather than handing it the whole
list. That costs a little graph setup per target and buys checkpointing: a crash, a
daily-quota wall, or a Ctrl-C resumes from where it stopped.

Progress is derived from the reports directory, not a separate progress file. An
earlier version tracked progress in a JSON file under /tmp, which the system's temp
cleanup deleted overnight along with the target cache - the research output survived
only because it is written into the repo. Reading the output directory back is
self-healing: whatever has a report is done, wherever the run stopped.

intent=research, so no emails are drafted and nothing is sent.

Usage:
    python scripts/run_batch.py <url>
    python scripts/run_batch.py <url> --status    # progress only, no model calls
    python scripts/run_batch.py <url> --limit 20  # stop after N targets this session
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
load_dotenv(dotenv_path=REPO / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
for _noisy in ("httpx", "langsmith", "google_genai", "urllib3", "langchain_core.tracers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from orchestrator.graph import unified_agent  # noqa: E402
from orchestrator.state import Target  # noqa: E402
from orchestrator.targets import _targets_from_page  # noqa: E402

REPORTS_DIR = REPO / "src" / "agents" / "outreach" / "reports"
CACHE_DIR = REPO / ".batch_cache"
REPORT_SUFFIX = " - General Lead Research Report.txt"

FLASH_LITE = "google_genai:gemini-3.1-flash-lite"

# Shallow on purpose: a 120-target list at full depth does not fit in the free-tier
# daily allowance, so this trades per-target depth for finishing the list.
CONFIG = {
    "recursion_limit": 100,
    "configurable": {
        "intent": "research",
        "max_targets": 5,
        "research_model": FLASH_LITE,
        "summarization_model": FLASH_LITE,
        "compression_model": FLASH_LITE,
        "final_report_model": FLASH_LITE,
        "research_sufficiency_model": FLASH_LITE,
        "allow_clarification": False,
        "max_concurrent_research_units": 1,
        "max_researcher_iterations": 2,
    },
}

DELAY_SECONDS = 8          # free tier caps requests per minute; pacing beats absorbing 429s
MAX_CONSECUTIVE_FAILURES = 5   # a daily-quota wall fails every remaining target instantly


def report_name(target_name: str) -> str:
    """Report filename for a target, matching how finish_target writes it."""
    safe = re.sub(r"[^\w\-. ]", "_", target_name).strip()[:60]
    return f"{safe}{REPORT_SUFFIX}"


def already_done(targets: list[Target]) -> set[str]:
    """Names whose research report is already on disk."""
    return {t.name for t in targets if (REPORTS_DIR / report_name(t.name)).exists()}


async def load_targets(url: str) -> list[Target]:
    """Extract targets from the page, caching so a resume does not re-render it."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / (re.sub(r"[^\w]", "_", url)[:80] + ".json")

    if cache.exists():
        targets = [Target(**t) for t in json.loads(cache.read_text())]
        logging.info(f"Loaded {len(targets)} cached targets")
        return targets

    logging.info(f"Rendering and extracting targets from {url} ...")
    targets = await _targets_from_page(url, CONFIG)
    cache.write_text(json.dumps([t.model_dump() for t in targets], indent=2))
    logging.info(f"Extracted and cached {len(targets)} targets")
    return targets


async def research_one(target: Target) -> None:
    """Run the unified agent for a single target, raising if it failed."""
    result = await unified_agent.ainvoke(
        {
            "targets": [target],
            "messages": [("user", f"Research {target.name} and find contact details")],
        },
        CONFIG,
    )
    failures = result.get("failures", [])
    if failures:
        raise RuntimeError(failures[-1])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    parser.add_argument("--limit", type=int, help="stop after this many targets this session")
    args = parser.parse_args()

    targets = await load_targets(args.url)
    done = already_done(targets)

    if args.status:
        print(f"{len(done)}/{len(targets)} done")
        for t in targets:
            if t.name not in done:
                print(f"  pending: {t.name}")
        return

    pending = [t for t in targets if t.name not in done]
    if args.limit:
        pending = pending[: args.limit]
    logging.info(f"{len(done)}/{len(targets)} already done, {len(pending)} queued now")

    consecutive = 0
    failed: dict[str, str] = {}
    started = time.time()

    for i, target in enumerate(pending, 1):
        label = f"[{i}/{len(pending)}] {target.name}"
        try:
            logging.info(f"{label} - researching ...")
            await research_one(target)
            consecutive = 0
            logging.info(f"{label} - done")
        except Exception as e:
            consecutive += 1
            failed[target.name] = str(e)[:300]
            logging.error(f"{label} - FAILED ({consecutive} in a row): {e}")
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                logging.error(
                    f"Stopping: {consecutive} failures in a row, most likely an exhausted "
                    f"daily quota. {len(pending) - i} queued targets left - rerun to resume."
                )
                break
        await asyncio.sleep(DELAY_SECONDS)

    final_done = already_done(targets)
    print(f"\n===== {len(final_done)}/{len(targets)} done in {(time.time()-started)/60:.0f} min =====")
    if failed:
        print(f"{len(failed)} failed this session:")
        for name, err in list(failed.items())[:10]:
            print(f"  {name}: {err[:120]}")
    print(f"Reports: {REPORTS_DIR}")


asyncio.run(main())
