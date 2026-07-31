"""Guards that per-lead state is fully reset between leads - no API calls.

The graph loops over leads reusing one state object, so any per-lead field left unreset
by update_CRM silently carries into the next lead. This has already caused real bugs:
reports accumulating across leads, and (before this guard) a stale outreach report link
that would put one lead's report in the next lead's email.
"""
import asyncio
import inspect
from unittest.mock import patch

from sales_outreach.outreach_automation import update_CRM
from sales_outreach.state import GraphState, LeadData

# Fields that legitimately persist across the per-lead loop rather than being reset.
CROSS_LEAD_FIELDS = {
    "leads_ids",      # original input
    "leads_data",     # the remaining queue, popped by check_for_remaining_leads
    "number_leads",   # decremented, not reset
    "current_lead",   # overwritten by check_for_remaining_leads on the next iteration
    "company_data",   # overwritten wholesale by run_shared_research
}


def _run_update_crm():
    """Invoke update_CRM with the lead loader stubbed out, returning its state update."""
    state = {
        "current_lead": LeadData(
            id="lead-1", name="Test", address="", email="t@example.com",
            phone="", profile="",
        ),
        "number_leads": 2,
        "research_sufficient": True,
        "lead_qualified": True,
        "lead_score": "8.0",
    }

    class _StubLoader:
        def update_record(self, lead_id, fields):
            return None

    with patch("sales_outreach.outreach_automation.get_lead_loader", return_value=_StubLoader()):
        command = asyncio.run(update_CRM(state, {}))
    return command.update


def test_update_crm_resets_every_per_lead_field():
    update = _run_update_crm()
    per_lead_fields = set(GraphState.__annotations__) - CROSS_LEAD_FIELDS
    missing = per_lead_fields - set(update)
    assert not missing, (
        f"update_CRM does not reset {sorted(missing)} - these would carry into the next lead. "
        "Either reset them there or add them to CROSS_LEAD_FIELDS with a reason."
    )


def test_reports_are_cleared_via_override_envelope():
    # A plain [] would be appended by the reducer, not clear it.
    assert _run_update_crm()["reports"] == {"type": "override", "value": []}


def test_outreach_report_link_is_cleared():
    # Specifically guarded: a stale link here ends up inside the next lead's email body.
    assert _run_update_crm()["custom_outreach_report_link"] == ""


def test_retry_budget_is_restored():
    # A spent counter would deny the next lead its one gap-focused retry.
    assert _run_update_crm()["research_retry_count"] == 0


def test_cross_lead_fields_are_not_accidentally_reset():
    update = _run_update_crm()
    assert "leads_data" not in update, "resetting the queue here would loop forever"
    assert update["number_leads"] == 1, "number_leads must decrement, not reset"


def test_update_crm_signature_matches_node_convention():
    params = list(inspect.signature(update_CRM).parameters)
    assert params == ["state", "config"]
