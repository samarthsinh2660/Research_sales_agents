"""Guards for per-target failure isolation.

One target's failure must not take down a batch, and a systemic failure must stop the
batch rather than consume the queue.
"""
import asyncio
from unittest.mock import patch

import pytest
from langgraph.types import Command

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


def test_isolated_lets_the_interrupt_pause_through():
    # interrupt() pauses the graph by raising GraphInterrupt, which subclasses Exception.
    # Catching it would turn the send-approval pause into a silently skipped target.
    from langgraph.errors import GraphInterrupt

    async def pauses(state, config):
        raise GraphInterrupt(("confirm_send",))

    with pytest.raises(GraphInterrupt):
        asyncio.run(_isolated(pauses)({}, {}))


def test_isolated_lets_parent_commands_through():
    from langgraph.errors import ParentCommand

    async def bubbles(state, config):
        raise ParentCommand(Command(goto="elsewhere"))

    with pytest.raises(ParentCommand):
        asyncio.run(_isolated(bubbles)({}, {}))


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


def test_finish_failure_still_advances_and_resets():
    # A Sheets write is a network call and can fail; that must cost one target, not the batch.
    from agent.graph import _isolated_finish
    from agent.state import PER_TARGET_FIELDS

    state = {
        "current_target": Target(name="Acme", source="inline"),
        "targets_remaining": 9,
        "consecutive_failures": 0,
    }
    command = asyncio.run(_isolated_finish(_boom)(state, {}))

    assert command.goto == "next_target"
    assert command.update["targets_remaining"] == 8
    assert command.update["consecutive_failures"] == 1
    missing = PER_TARGET_FIELDS - set(command.update)
    assert not missing, f"failure path leaks {sorted(missing)} into the next target"


def test_reset_constant_covers_every_per_target_field():
    # Both the success and failure paths spread PER_TARGET_RESET, so covering it once
    # keeps them from drifting apart.
    from agent.graph import PER_TARGET_RESET
    from agent.state import PER_TARGET_FIELDS

    assert set(PER_TARGET_RESET) == PER_TARGET_FIELDS


def test_caller_supplied_targets_respect_the_cap():
    from agent.graph import start_run

    many = [Target(name=f"T{i}", source="inline") for i in range(30)]
    with pytest.raises(ValueError, match="above the max_targets limit"):
        asyncio.run(start_run({"targets": many}, {"configurable": {"max_targets": 25}}))


def test_caller_supplied_targets_within_the_cap_are_used_as_is():
    from agent.graph import start_run

    given = [Target(name="Acme", source="page", context="CIO, Mahindra")]
    command = asyncio.run(start_run({"targets": given}, {"configurable": {"max_targets": 25}}))
    # Context must survive: it is why callers pass targets instead of a comma-joined string.
    assert command.update["targets"][0].context == "CIO, Mahindra"


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
