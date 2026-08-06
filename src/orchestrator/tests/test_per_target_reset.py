"""Guards that per-target state is fully reset between targets."""
from orchestrator.state import PER_TARGET_FIELDS, AgentState, Target, replace_reducer

# Fields that legitimately persist across the loop rather than being reset.
CROSS_TARGET_FIELDS = {
    "messages", "targets", "current_target", "targets_remaining", "failures",
    # Counts a run of failures across targets, so resetting it per target would
    # defeat the circuit breaker it exists to feed.
    "consecutive_failures",
}


def test_target_defaults():
    t = Target(name="Acme", source="prompt")
    assert t.website is None and t.email is None and t.crm_row_id is None


def test_replace_reducer_replaces_rather_than_appends():
    # The research subgraph echoes messages back; appending would duplicate them.
    assert replace_reducer(["old"], ["new"]) == ["new"]


def test_per_target_fields_cover_state_minus_cross_target():
    assert PER_TARGET_FIELDS == set(AgentState.__annotations__) - CROSS_TARGET_FIELDS


import asyncio
from unittest.mock import patch

from orchestrator.graph import finish_target


def _run_finish_target(source="inline", crm_row_id=None):
    state = {
        "current_target": Target(name="Acme", source=source, crm_row_id=crm_row_id),
        "targets_remaining": 2,
        "reports": [],
        "research_sufficient": True,
        "lead_qualified": True,
        "lead_score": "8.0",
    }
    with patch("orchestrator.graph.get_lead_loader") as loader:
        command = asyncio.run(finish_target(state, {}))
    return command.update, loader


def test_finish_target_resets_every_per_target_field():
    update, _ = _run_finish_target()
    missing = PER_TARGET_FIELDS - set(update)
    assert not missing, (
        f"finish_target does not reset {sorted(missing)} - these would carry into the "
        "next target. Reset them there, or add them to CROSS_TARGET_FIELDS with a reason."
    )


def test_reports_cleared_via_override_envelope():
    update, _ = _run_finish_target()
    assert update["reports"] == {"type": "override", "value": []}


def test_crm_skipped_for_non_sheet_targets():
    _, loader = _run_finish_target(source="inline")
    loader.assert_not_called()


def test_crm_written_for_sheet_targets():
    _, loader = _run_finish_target(source="sheet", crm_row_id="7")
    loader.assert_called_once()
