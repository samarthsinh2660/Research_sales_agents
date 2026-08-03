"""The contact-finding agent.

A deliberately small sibling of deep_researcher: plan, a bounded tool loop, then assemble
a structured card. It shares this package's tools, models and configuration, and must not
grow into a second research agent - it has three tools, a hard cap on tool-calling rounds,
no sub-agents and no compression step.

The structured output is the point. Contacts previously reached the caller inside the
research report's prose, where compress_research and the report writer dropped them:
phone numbers were extracted by the crawler and then discarded because the report
template had no field for them. A ContactCard cannot lose a detail for lack of a slot.
"""

import asyncio
import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents.research.configuration import Configuration
from agents.research.deep_researcher import (
    configurable_model,
    execute_tool_safely,
    fallback_model,
)
from agents.research.prompts import (
    CONTACT_SOURCE_GUIDE,
    NO_GUESSING_RULE,
    contact_finder_prompt,
    contact_plan_prompt,
    extract_contact_card_prompt,
)
from agents.research.state import (
    ContactAgentInputState,
    ContactAgentState,
    ContactCard,
    ContactPlan,
)
from agents.research.utils import (
    get_api_key_for_model,
    get_contact_tools,
    get_fallback_configs,
    get_today_str,
)

###################
# Model Setup
###################

# Reserved and placeholder domains. Real contact pages ship these inside form templates
# ("you@example.com"), and the card assembler has copied them out as though they were a
# named person's address - complete with a source URL, which makes a fake look verified.
# Enforced in code rather than in the prompt because a prompt rule can be ignored.
_PLACEHOLDER_EMAIL = re.compile(
    r"@(example|test|sample|domain|yourdomain|yourcompany|email|mail)\.(com|org|net|co)$",
    re.IGNORECASE,
)


def drop_placeholder_contacts(card: ContactCard) -> list[str]:
    """Strip placeholder addresses from a card, returning what was removed.

    A bounced email is recoverable; an invented address presented with a citation is the
    one failure that discredits every other contact on the card.
    """
    dropped = [p.value for p in card.emails if _PLACEHOLDER_EMAIL.search(p.value)]
    if dropped:
        card.emails = [p for p in card.emails if not _PLACEHOLDER_EMAIL.search(p.value)]
        if card.best_route_value in dropped:
            card.best_route = "unreachable" if card.is_empty() else "role_inbox_attn"
            card.best_route_value = card.emails[0].value if card.emails else ""
            card.best_route_reason = "best route was a placeholder address and was removed"
    return dropped


def _model_config(model_name: str, config: RunnableConfig, max_tokens: int) -> dict:
    """Build the .with_config() dict for a contact-agent model call."""
    return {
        "model": model_name,
        "max_tokens": max_tokens,
        "api_key": get_api_key_for_model(model_name, config),
        "tags": ["langsmith:nostream"],
    }


###################
# Nodes
###################

async def plan_contact_search(state: ContactAgentState, config: RunnableConfig) -> Command[Literal["find_contacts"]]:
    """Decide which sources to try for this target, and in what order.

    This is where discretion lives: a college with a public staff directory and a
    conglomerate executive need completely different hunts. What is *not* discretionary
    is whether the hunt happens at all - contact research used to be an instruction in
    the supervisor prompt, and the supervisor skipped it silently.

    Args:
        state: Target identity and entity type
        config: Runtime configuration with model settings

    Returns:
        Command to the finder loop, carrying the plan
    """
    configurable = Configuration.from_runnable_config(config)
    max_tokens = configurable.research_model_max_tokens

    model = (
        configurable_model
        .with_structured_output(ContactPlan)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(_model_config(configurable.research_model, config, max_tokens))
        .with_fallbacks([
            fallback_model.with_config(cfg).with_structured_output(ContactPlan)
            for cfg in get_fallback_configs(configurable, config, max_tokens)
        ])
    )

    prompt = contact_plan_prompt.format(
        date=get_today_str(),
        target_name=state.get("target_name", ""),
        entity_type=state.get("entity_type") or "unknown",
        website=state.get("target_website") or "not known - find it first",
        context=state.get("target_context") or "nothing beyond the name",
        source_guide=CONTACT_SOURCE_GUIDE,
    )

    try:
        plan = await model.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        # A planning failure must not cost us the lookup entirely: the finder works from
        # the source guide alone, just without target-specific prioritization.
        logging.warning(f"Contact planning failed for {state.get('target_name')}, using default order: {e}")
        plan = ContactPlan(
            priority_sources=["the organization's own contact/team pages", "targeted web search"],
            queries=[f"{state.get('target_name', '')} contact email phone"],
            reasoning="planning call failed; falling back to the default ladder",
        )

    return Command(goto="find_contacts", update={"contact_plan": plan})


async def find_contacts(state: ContactAgentState, config: RunnableConfig) -> Command[Literal["find_contacts_tools"]]:
    """Call tools to hunt for contact details.

    Args:
        state: Target identity, plan, and the conversation so far
        config: Runtime configuration with model settings

    Returns:
        Command to the tool executor
    """
    configurable = Configuration.from_runnable_config(config)
    max_tokens = configurable.research_model_max_tokens
    tools = get_contact_tools()

    model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(_model_config(configurable.research_model, config, max_tokens))
        .with_fallbacks([
            fallback_model.with_config(cfg).bind_tools(tools)
            for cfg in get_fallback_configs(configurable, config, max_tokens)
        ])
    )

    plan: ContactPlan | None = state.get("contact_plan")
    system_prompt = contact_finder_prompt.format(
        date=get_today_str(),
        target_name=state.get("target_name", ""),
        entity_type=state.get("entity_type") or "unknown",
        website=state.get("target_website") or "not known - find it first",
        context=state.get("target_context") or "nothing beyond the name",
        priority_sources="; ".join(plan.priority_sources) if plan else "the ladder above",
        queries="; ".join(plan.queries) if plan else "search for the organization's contact page",
        source_guide=CONTACT_SOURCE_GUIDE,
        no_guessing_rule=NO_GUESSING_RULE,
        max_tool_calls=configurable.max_contact_tool_calls,
    )

    prior = state.get("contact_messages", [])
    # The opening instruction is added to state on the first pass only, so the transcript
    # that build_contact_card later reads starts with what was actually asked for.
    opening = [] if prior else [HumanMessage(content="Find every way to contact this target.")]
    response = await model.ainvoke([SystemMessage(content=system_prompt)] + prior + opening)

    return Command(
        goto="find_contacts_tools",
        update={
            "contact_messages": opening + [response],
            "contact_tool_calls": state.get("contact_tool_calls", 0) + 1,
        },
    )


async def find_contacts_tools(state: ContactAgentState, config: RunnableConfig) -> Command[Literal["find_contacts", "build_contact_card"]]:
    """Execute the finder's tool calls, or move on when it is done.

    Args:
        state: Conversation so far, including any pending tool calls
        config: Runtime configuration with the tool-call cap

    Returns:
        Command back to the finder, or on to card assembly
    """
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("contact_messages", [])
    last = messages[-1] if messages else None
    tool_calls = getattr(last, "tool_calls", None) or []

    if not tool_calls:
        return Command(goto="build_contact_card")

    tools_by_name = {t.name: t for t in get_contact_tools()}
    known = [c for c in tool_calls if c["name"] in tools_by_name]
    results = await asyncio.gather(*[
        execute_tool_safely(tools_by_name[c["name"]], c["args"], config) for c in known
    ])
    outputs = [
        ToolMessage(content=str(result), name=call["name"], tool_call_id=call["id"])
        for call, result in zip(known, results)
    ]

    if state.get("contact_tool_calls", 0) >= configurable.max_contact_tool_calls:
        logging.info(
            f"Contact agent hit its {configurable.max_contact_tool_calls}-round cap for "
            f"{state.get('target_name')}; assembling the card from what it has"
        )
        return Command(goto="build_contact_card", update={"contact_messages": outputs})

    return Command(goto="find_contacts", update={"contact_messages": outputs})


async def build_contact_card(state: ContactAgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """Assemble the structured card from the finder's transcript.

    Fails to an empty `unreachable` card rather than raising: a target we could not build
    a card for is still one the batch should finish, and it is the caller's post-check
    that decides whether that emptiness is acceptable.

    Args:
        state: The finder's full transcript
        config: Runtime configuration with model settings

    Returns:
        Command to END, carrying the contact card
    """
    configurable = Configuration.from_runnable_config(config)
    max_tokens = configurable.research_model_max_tokens

    model = (
        configurable_model
        .with_structured_output(ContactCard)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(_model_config(configurable.research_model, config, max_tokens))
        .with_fallbacks([
            fallback_model.with_config(cfg).with_structured_output(ContactCard)
            for cfg in get_fallback_configs(configurable, config, max_tokens)
        ])
    )

    transcript = "\n\n".join(
        f"[{type(m).__name__}] {getattr(m, 'content', '')}" for m in state.get("contact_messages", [])
    )
    user_message = (
        f"# Target\n\n{state.get('target_name', '')}\n\n"
        f"# Contact-finding transcript\n\n{transcript}"
    )

    try:
        card = await model.ainvoke([
            SystemMessage(content=extract_contact_card_prompt.format(no_guessing_rule=NO_GUESSING_RULE)),
            HumanMessage(content=user_message),
        ])
    except Exception as e:
        logging.warning(f"Contact card assembly failed for {state.get('target_name')}: {e}")
        card = ContactCard(
            organization=state.get("target_name", ""),
            best_route="unreachable",
            best_route_reason=f"card assembly failed: {e}",
        )

    card.organization = card.organization or state.get("target_name", "")
    dropped = drop_placeholder_contacts(card)
    if dropped:
        logging.warning(
            f"{state.get('target_name')}: dropped {len(dropped)} placeholder address(es) "
            f"from the contact card: {dropped}"
        )
    logging.info(
        f"{state.get('target_name')}: contact card - route={card.best_route} "
        f"emails={len(card.emails)} phones={len(card.phones)} "
        f"linkedin={len(card.linkedin_urls)} sources_checked={len(card.sources_checked)}"
    )
    return Command(goto=END, update={"contact_card": card})


###################
# Graph Construction
###################

contact_agent_builder = StateGraph(
    ContactAgentState,
    input_schema=ContactAgentInputState,
    config_schema=Configuration,
)

contact_agent_builder.add_node("plan_contact_search", plan_contact_search)   # Per-target source selection
contact_agent_builder.add_node("find_contacts", find_contacts)               # Bounded tool loop
contact_agent_builder.add_node("find_contacts_tools", find_contacts_tools)   # Tool execution
contact_agent_builder.add_node("build_contact_card", build_contact_card)     # Structured assembly

contact_agent_builder.add_edge(START, "plan_contact_search")

contact_agent = contact_agent_builder.compile()
