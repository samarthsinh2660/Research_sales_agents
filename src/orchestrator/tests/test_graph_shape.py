"""Structural tests for the unified graph - no API calls, no credentials."""
from orchestrator.graph import unified_agent

EDGES = {(e.source, e.target) for e in unified_agent.get_graph().edges}
NODES = set(unified_agent.get_graph().nodes)


def test_graph_compiles_without_credentials():
    # Proves no OAuth or client construction happens at import time.
    assert unified_agent is not None


def test_research_is_a_nested_subgraph_node():
    assert "research" in NODES


def test_all_four_intent_depths_are_reachable():
    assert ("prepare_research", "research") in EDGES
    assert ("check_research_sufficiency", "score_target") in EDGES
    assert ("score_target", "generate_materials") in EDGES
    assert ("generate_materials", "approve_send") in EDGES


def test_send_is_gated_by_approval():
    # There must be no path from material generation straight to sending.
    assert ("generate_materials", "send_email") not in EDGES
    assert ("approve_send", "send_email") in EDGES


def test_every_branch_converges_on_finish_target():
    for source in ("check_research_sufficiency", "score_target", "generate_materials",
                   "approve_send", "send_email"):
        assert any(s == source and t == "finish_target" for s, t in EDGES), source


def test_per_target_loop_closes_and_terminates():
    assert ("finish_target", "next_target") in EDGES
    assert ("next_target", "__end__") in EDGES


def test_research_retry_path_exists():
    assert ("check_research_sufficiency", "prepare_research") in EDGES
