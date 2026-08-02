"""Unit tests for AgentConfiguration - no API calls."""
from orchestrator.configuration import INTENT_DEPTH, AgentConfiguration, OutreachIntent


def test_default_intent_is_research():
    # research is the only intent with no outward-facing side effect
    assert AgentConfiguration.from_runnable_config({}).intent == OutreachIntent.RESEARCH


def test_approval_required_by_default():
    assert AgentConfiguration.from_runnable_config({}).require_send_approval is True


def test_runtime_override():
    cfg = AgentConfiguration.from_runnable_config({"configurable": {"intent": "draft"}})
    assert cfg.intent == OutreachIntent.DRAFT


def test_intent_depth_ordering():
    assert (INTENT_DEPTH[OutreachIntent.RESEARCH]
            < INTENT_DEPTH[OutreachIntent.QUALIFY]
            < INTENT_DEPTH[OutreachIntent.DRAFT]
            < INTENT_DEPTH[OutreachIntent.SEND])


def test_no_env_var_collision_with_other_configs():
    from agents.outreach.configuration import SalesConfiguration
    from agents.research.configuration import Configuration
    mine = set(AgentConfiguration.model_fields)
    assert not (mine & set(Configuration.model_fields))
    assert not (mine & set(SalesConfiguration.model_fields))
