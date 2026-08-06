"""Guards that an email making unsupported claims never becomes a draft.

The research pipeline verifies citations, but the email was written freely from the report
and nothing checked it. A confidently wrong detail sent to a real partner cannot be walked
back, so the gate fails closed - including when the check itself errors.
"""
import asyncio
from unittest.mock import AsyncMock, patch

from orchestrator.graph import _check_email_grounding, build_unified_agent
from orchestrator.state import Target

REPORT = "Acme Corp is an IT services firm founded in 2004, serving manufacturing clients."


async def _fake_research(state, config=None):
    return {"final_report": REPORT, "entity_type": "company"}


def _run(grounded, claims=()):
    """Run to draft depth with the grounding verdict forced, counting drafts created."""
    calls = {"draft": 0}

    async def fake_invoke_llm(system_prompt, user_message, model_name, config, response_format=None):
        if response_format is not None:
            name = response_format.__name__
            if name == "ResearchSufficiency":
                return response_format(sufficient=True, gaps="")
            if name == "LeadScore":
                return response_format(score=9.0, track="Technology", reasoning="r", angle="a")
            if name == "ContactRoute":
                return response_format(
                    recipient_name="Asha Rao", recipient_role="CTO",
                    email="asha@acme.test", route_type="personal_email",
                )
            if name == "EmailGrounding":
                return response_format(grounded=grounded, unsupported_claims=list(claims))
            return response_format(subject="s", email="body")
        return "text"

    class _Gmail:
        def create_draft_email(self, **kwargs):
            calls["draft"] += 1

        def send_email(self, **kwargs):
            raise AssertionError("must not send")

    graph = build_unified_agent(research_node=_fake_research)
    target = Target(name="Acme Corp", source="inline", email="hi@acme.test")

    with patch("orchestrator.graph.invoke_llm", side_effect=fake_invoke_llm), \
         patch("orchestrator.graph.GmailTools", _Gmail), \
         patch("orchestrator.graph.save_reports_locally"), \
         patch("orchestrator.graph.resolve_targets", new=AsyncMock(return_value=[target])):
        asyncio.run(graph.ainvoke(
            {"messages": [("user", "research Acme")]} ,
            {"recursion_limit": 100, "configurable": {"intent": "draft"}},
        ))
    return calls


def test_grounded_email_becomes_a_draft():
    assert _run(grounded=True)["draft"] == 1


def test_ungrounded_email_is_not_drafted():
    # "Recently opened a Dubai office" appears nowhere in the report.
    assert _run(grounded=False, claims=["recently opened a Dubai office"])["draft"] == 0


def test_grounding_check_failure_blocks_the_draft():
    # Fail closed: an errored check must not read as approval.
    class Cfg:
        research_sufficiency_model = "x"

    async def boom(**kwargs):
        raise RuntimeError("model unavailable")

    with patch("orchestrator.graph.invoke_llm", side_effect=boom):
        result = asyncio.run(_check_email_grounding("body", REPORT, Cfg(), {}))
    assert result.grounded is False
    assert result.unsupported_claims


def test_grounding_passes_through_a_clean_verdict():
    class Cfg:
        research_sufficiency_model = "x"

    async def ok(**kwargs):
        from agents.outreach.state import EmailGrounding
        return EmailGrounding(grounded=True, unsupported_claims=[])

    with patch("orchestrator.graph.invoke_llm", side_effect=ok):
        result = asyncio.run(_check_email_grounding("body", REPORT, Cfg(), {}))
    assert result.grounded is True
