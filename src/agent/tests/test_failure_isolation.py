"""Guards that one target's failure cannot take down a batch, and that a systemic
failure stops the batch instead of consuming the queue."""
import asyncio
from unittest.mock import patch

from agent.graph import _isolated, finish_target, next_target
from agent.state import Target


def _boom(state, config):
    raise RuntimeError("quota exhausted")


async def _ok(state, config):
    return "fine"


def test_isolated_passes_success_through_untouched():
    assert asyncio.run(_isolated(_ok)({}, {})) == "fine"


def test_isolated_routes_failure_to_finish_target():
    state = {"current_target": Target(name="Acme", source="inline")}
    command = asyncio.run(_isolated(_boom)(state, {}))
    assert command.goto == "finish_target"
    assert command.update["current_target_failed"] is True


def test_isolated_records_the_failure_with_target_and_node():
    state = {"current_target": Target(name="Acme", source="inline"), "failures": ["earlier"]}
    command = asyncio.run(_isolated(_boom)(state, {}))
    recorded = command.update["failures"]
    assert recorded["type"] == "override"
    # Prior failures survive: the override envelope replaces the list, so the node has
    # to carry them forward itself.
    assert recorded["value"][0] == "earlier"
    assert "Acme" in recorded["value"][1] and "_boom" in recorded["value"][1]


def test_isolated_survives_a_missing_current_target():
    # start_run failing leaves no current_target; the wrapper must not raise itself.
    command = asyncio.run(_isolated(_boom)({}, {}))
    assert "<unknown>" in command.update["failures"]["value"][0]


def _finish(failed):
    state = {
        "current_target": Target(name="Acme", source="inline"),
        "targets_remaining": 2,
        "reports": [],
        "current_target_failed": failed,
        "consecutive_failures": 3,
    }
    with patch("agent.graph.get_lead_loader"):
        return asyncio.run(finish_target(state, {})).update


def test_finish_target_increments_the_streak_on_failure():
    assert _finish(failed=True)["consecutive_failures"] == 4


def test_finish_target_clears_the_streak_on_success():
    # Scattered failures across a long batch are normal; only an unbroken run is systemic.
    assert _finish(failed=False)["consecutive_failures"] == 0


def _next(consecutive, remaining):
    state = {
        "targets": [Target(name=f"T{i}", source="inline") for i in range(remaining)],
        "consecutive_failures": consecutive,
        "failures": ["something"],
    }
    return asyncio.run(next_target(state, {"configurable": {"max_consecutive_failures": 5}}))


def test_next_target_aborts_once_the_streak_hits_the_limit():
    assert _next(consecutive=5, remaining=80).goto == "__end__"


def test_next_target_keeps_going_below_the_limit():
    assert _next(consecutive=4, remaining=80).goto == "prepare_research"
