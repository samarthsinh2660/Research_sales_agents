"""CLI entrypoint for running the sales outreach agent."""

import asyncio
import logging

from dotenv import load_dotenv

from agents.outreach.outreach_automation import outreach_automation

load_dotenv()


async def main():
    """Run the outreach automation over all new leads in the configured source."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Lead source and behaviour come from SalesConfiguration. Anything omitted here falls
    # back to its env var (e.g. SHEET_ID) and then to the field default, so the common
    # case needs no configurable block at all.
    #
    # To use Airtable instead:
    #   "lead_loader_type": "airtable",
    #   "airtable_base_id": os.getenv("AIRTABLE_BASE_ID"),
    #   "airtable_table_name": os.getenv("AIRTABLE_TABLE_NAME"),
    config = {
        "recursion_limit": 1000,
        "configurable": {
            "lead_loader_type": "google_sheets",
        },
    }

    # Lead ids to be processed, leave empty to fetch all new leads
    inputs = {"leads_ids": []}

    output = await outreach_automation.ainvoke(inputs, config)
    print(output)  # noqa: T201 - final result of a CLI run belongs on stdout


if __name__ == "__main__":
    asyncio.run(main())
