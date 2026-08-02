"""Each intent must stop at exactly its own depth. Fully mocked - no quota, no email."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agents.research.state import ContactCard, ContactPoint
from orchestrator.graph import build_unified_agent
from orchestrator.state import Target

CACHED_REPORT = "Acme Corp is an IT services firm founded in 2004, serving manufacturing clients."

CACHED_CARD = ContactCard(
    organization="Acme Corp",
    emails=[ContactPoint(value="asha@acme.test", kind="personal", source_url="https://acme.test/team")],
    best_route="direct_email",
    best_route_value="asha@acme.test",
    sources_checked=["official website crawl", "targeted search", "MCA filing"],
)


async def _fake_research(state, config=None):
    """Stand in for the research subgraph, returning what it would have written to state."""
    return {"final_report": CACHED_REPORT, "entity_type": "company"}


def _run(intent):
    """Run the graph end-to-end with every external call mocked, counting what happened."""
    calls = {"score": 0, "materials": 0, "draft": 0, "send": 0}

    async def fake_invoke_llm(system_prompt, user_message, model_name, config, response_format=None):
        if response_format is not None:
            name = response_format.__name__
            if name == "ResearchSufficiency":
                return response_format(sufficient=True, gaps="")
            if name == "LeadScore":
                calls["score"] += 1
                return response_format(
                    score=9.0, track="Technology",
                    reasoning="Runs a large delivery centre.", angle="Recent SDK launch.",
                )
            if name == "ContactRoute":
                return response_format(
                    recipient_name="Asha Rao", recipient_role="CTO",
                    email="asha@acme.test", route_type="personal_email",
                )
            if name == "EmailGrounding":
                # Grounded by default: the ungrounded path is covered in test_email_grounding.
                return response_format(grounded=True, unsupported_claims=[])
            calls["materials"] += 1
            return response_format(subject="subject", email="body")
        calls["materials"] += 1
        return "generated text"

    class _Gmail:
        def create_draft_email(self, **kwargs):
            calls["draft"] += 1

        def send_email(self, **kwargs):
            calls["send"] += 1

    graph = build_unified_agent(research_node=_fake_research)
    target = Target(name="Acme Corp", source="inline", email="hi@acme.test")

    fake_contact_agent = AsyncMock()
    fake_contact_agent.ainvoke = AsyncMock(return_value={"contact_card": CACHED_CARD})

    with patch("orchestrator.graph.invoke_llm", side_effect=fake_invoke_llm), \
         patch("orchestrator.graph.contact_agent", fake_contact_agent), \
         patch("orchestrator.graph.GmailTools", _Gmail), \
         patch("orchestrator.graph.save_reports_locally"), \
         patch("orchestrator.graph.resolve_targets", new=AsyncMock(return_value=[target])):
        config = {
            "recursion_limit": 100,
            "configurable": {"intent": intent, "require_send_approval": False},
        }
        asyncio.run(graph.ainvoke({"messages": [("user", "research Acme Corp")]}, config))

    return calls


@pytest.mark.parametrize("intent,expect_score,expect_draft", [
    ("research", False, False),
    ("qualify", True, False),
    ("draft", True, True),
])
def test_intent_stops_at_correct_depth(intent, expect_score, expect_draft):
    calls = _run(intent)
    assert (calls["score"] > 0) is expect_score, f"{intent}: score={calls['score']}"
    assert (calls["draft"] > 0) is expect_draft, f"{intent}: draft={calls['draft']}"
    # No intent below send may ever send.
    assert calls["send"] == 0


def test_tuple_message_input_is_accepted():
    # ("user", "text") is the common input shape; the replacing reducer does not coerce
    # it into a Message, so the graph must read it directly.
    assert _run("research") is not None
