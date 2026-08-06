"""Main LangGraph implementation for the unified research + outreach agent.

One entry point: the prompt names the subject, config names how deep to go. Research is
nested as a native subgraph so its internals stay visible in traces and Studio, and the
sales work is reused from agents.outreach rather than reimplemented.
"""

import functools
import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.outreach.configuration import SalesConfiguration
from agents.outreach.prompts import (
    CHECK_EMAIL_GROUNDING_PROMPT,
    CHECK_RESEARCH_SUFFICIENCY_PROMPT,
    GENERATE_OUTREACH_REPORT_PROMPT,
    PERSONALIZE_EMAIL_PROMPT,
    PROOF_READER_PROMPT,
    SCORE_LEAD_PROMPT,
    SELECT_CONTACT_ROUTE_PROMPT,
)
from agents.outreach.state import (
    ContactRoute,
    EmailGrounding,
    EmailResponse,
    LeadScore,
    Report,
    ResearchSufficiency,
)
from agents.outreach.tools.base.gmail_tools import GmailTools
from agents.outreach.utils import (
    get_current_date,
    get_lead_loader,
    get_report,
    invoke_llm,
    qualification_decision,
    research_sufficiency_decision,
    save_reports_locally,
)
from agents.research.configuration import Configuration
from agents.research.contact_agent import contact_agent
from agents.research.deep_researcher import deep_researcher
from agents.research.entity_registry import get_sales_context
from agents.research.state import ContactCard
from orchestrator.configuration import INTENT_DEPTH, AgentConfiguration, OutreachIntent
from orchestrator.state import AgentInputState, AgentState, Target
from orchestrator.targets import resolve_targets

RESEARCH_REPORT_TITLE = "General Lead Research Report"
EMAIL_REPORT_TITLE = "Personalized Email"
CONTACT_CARD_TITLE = "Contact Card"

# Cleared when a target ends, however it ends. Shared by the normal path and the failure
# path so the two cannot drift: a field reset in one but not the other would leak that
# target's report or score into the next one. Coverage is asserted against
# PER_TARGET_FIELDS by test.
PER_TARGET_RESET: dict = {
    "current_target_failed": False,
    "reports": {"type": "override", "value": []},
    "final_report": "",
    "entity_type": None,
    "research_sufficient": False,
    "research_gaps": "",
    "research_retry_count": 0,
    "lead_score": "",
    "lead_qualified": False,
    "lead_track": "",
    "lead_reasoning": "",
    "lead_angle": "",
    "contact_card": None,
    "contact_route": "",
    "recipient_name": "",
    "email_grounded": False,
    "unsupported_claims": [],
    "outreach_report_link": "",
    "send_approved": None,
}


def _at_least(configurable: AgentConfiguration, intent: OutreachIntent) -> bool:
    """Whether the configured intent goes at least as deep as the given one."""
    return INTENT_DEPTH[configurable.intent] >= INTENT_DEPTH[intent]


def _prompt_text(messages) -> str:
    """Read the prompt text from the last message, whatever shape it arrived in.

    Callers pass messages as Message objects, ("user", "text") tuples, or
    {"role":..., "content":...} dicts. The usual `add_messages` reducer coerces all
    three, but this graph uses a replacing reducer (the research subgraph echoes
    messages back), so the raw shape survives and has to be handled here.
    """
    if not messages:
        return ""
    last = messages[-1]
    if hasattr(last, "content"):
        return str(last.content)
    if isinstance(last, dict):
        return str(last.get("content", ""))
    if isinstance(last, (tuple, list)) and len(last) == 2:
        return str(last[1])
    return str(last)


def _isolated(node):
    """Wrap a per-target node so one target's failure does not abort the whole batch.

    Without this, a single transient error - an exhausted model quota, a network blip,
    a malformed structured output - propagates out of the graph and loses every target
    still queued behind it. A long batch makes that near-certain, so a failing target is
    recorded and skipped instead.

    finish_target is wrapped separately by _isolated_finish: it is the recovery
    destination, so routing its own failures back to itself would loop forever.

    GraphBubbleUp is re-raised rather than caught. LangGraph signals control flow through
    exceptions - interrupt() raises GraphInterrupt to pause for human input, and
    Command(graph=PARENT) raises ParentCommand - both of which subclass Exception. Treating
    those as target failures silently turns the send-approval pause into a skipped target,
    disabling the human-in-the-loop gate exactly when it matters.
    """
    @functools.wraps(node)
    async def wrapper(state: AgentState, config: RunnableConfig):
        try:
            return await node(state, config)
        except GraphBubbleUp:
            raise
        except Exception as e:
            target = state.get("current_target")
            name = target.name if target else "<unknown>"
            logging.exception(f"Target '{name}' failed in {node.__name__}: {e}")
            return Command(
                goto="finish_target",
                update={
                    "current_target_failed": True,
                    "failures": {
                        "type": "override",
                        "value": [*state.get("failures", []), f"{name} [{node.__name__}]: {e}"],
                    },
                },
            )
    return wrapper


def _isolated_finish(node):
    """Wrap finish_target so its own failure skips one target rather than the batch.

    It writes files and, for sheet-sourced targets, makes a Google Sheets network call -
    either can fail transiently. Recovery goes forward to next_target instead of back to
    finish_target, because a wrapper that routes failures to itself would loop forever.

    Per-target state is reset on this path too. Without that, a target that failed here
    would leak its report and score into the next one, which is the leak the reset in
    finish_target exists to prevent.
    """
    @functools.wraps(node)
    async def wrapper(state: AgentState, config: RunnableConfig):
        try:
            return await node(state, config)
        except GraphBubbleUp:
            raise
        except Exception as e:
            target = state.get("current_target")
            name = target.name if target else "<unknown>"
            logging.exception(f"Target '{name}' failed while finishing: {e}")
            return Command(
                goto="next_target",
                update={
                    **PER_TARGET_RESET,
                    "targets_remaining": max(0, state.get("targets_remaining", 1) - 1),
                    "consecutive_failures": state.get("consecutive_failures", 0) + 1,
                    "failures": {
                        "type": "override",
                        "value": [*state.get("failures", []), f"{name} [finish_target]: {e}"],
                    },
                },
            )
    return wrapper


async def start_run(state: AgentState, config: RunnableConfig) -> Command[Literal["next_target"]]:
    """Resolve the prompt into a target list, unless the caller supplied one.

    Fails loudly rather than processing zero targets silently, which would otherwise look
    like a successful no-op run.

    Args:
        state: Input state carrying the user's prompt, or a pre-resolved target list
        config: Runtime configuration for target cap and sheet loading

    Returns:
        Command to enter the per-target loop

    Raises:
        ValueError: if a caller-supplied list exceeds max_targets
    """
    targets = state.get("targets")
    if targets:
        # resolve_targets enforces the cap on the paths it owns; a caller-supplied list
        # skips it, and an uncapped batch is exactly the cost surprise the cap exists
        # to prevent, so it is enforced here too.
        limit = AgentConfiguration.from_runnable_config(config).max_targets
        if len(targets) > limit:
            raise ValueError(
                f"Received {len(targets)} targets, above the max_targets limit of "
                f"{limit}. Raise the limit or pass fewer targets."
            )
    else:
        targets = await resolve_targets(_prompt_text(state.get("messages")), config)

    return Command(
        goto="next_target",
        update={"targets": targets, "targets_remaining": len(targets), "failures": []},
    )


async def next_target(state: AgentState, config: RunnableConfig) -> Command[Literal["prepare_research", "__end__"]]:
    """Pop the next target, or end when the queue is empty.

    Aborts early on a run of consecutive failures. When a daily model quota is exhausted
    every remaining target fails the instant it is tried, so without this the batch would
    "complete" in seconds having researched nothing, and the queue would be gone.

    Args:
        state: Current state holding the remaining targets
        config: Runtime configuration for the consecutive-failure limit

    Returns:
        Command to research the next target, or to end
    """
    configurable = AgentConfiguration.from_runnable_config(config)
    remaining = list(state.get("targets", []))
    failures = state.get("failures", [])

    consecutive = state.get("consecutive_failures", 0)
    if consecutive >= configurable.max_consecutive_failures:
        logging.error(
            f"Aborting: {consecutive} targets failed back to back, {len(remaining)} left "
            f"unprocessed. Last failure: {failures[-1] if failures else 'unknown'}"
        )
        return Command(goto=END)

    if not remaining:
        if failures:
            logging.warning(f"Finished with {len(failures)} failed target(s): {failures}")
        logging.info("Finished - no targets remaining")
        return Command(goto=END)

    current = remaining.pop(0)
    logging.info(f"Processing target: {current.name} ({len(remaining)} remaining)")
    return Command(
        goto="prepare_research",
        update={"current_target": current, "targets": remaining},
    )


async def prepare_research(state: AgentState, config: RunnableConfig) -> Command[Literal["research"]]:
    """Build the research query for the current target and hand off to the subgraph.

    Exists as its own node because the research subgraph reads `messages` from parent
    state; this is where that gets set, and where a retry adds gap focus.

    Args:
        state: Current state holding the target and any prior research gap
        config: Runtime configuration (unused, kept for node signature consistency)

    Returns:
        Command into the research subgraph
    """
    target: Target = state["current_target"]

    query = f"Research {target.name}"
    if target.website:
        query += f" ({target.website})"
    if target.context:
        query += f". Known context: {target.context}"
    query += (
        " - what they do, services/products offered, size, and any named contacts "
        "or contact information."
    )

    gaps = state.get("research_gaps", "")
    if gaps:
        query += f" A previous pass found this gap - focus on filling it: {gaps}"

    return Command(goto="research", update={"messages": [HumanMessage(content=query)]})


async def check_research_sufficiency(state: AgentState, config: RunnableConfig) -> Command[Literal["find_target_contacts", "prepare_research", "finish_target"]]:
    """Gate on whether research has enough substance to act on.

    Args:
        state: Current state holding the research report
        config: Runtime configuration with model settings and retry budget

    Returns:
        Command to score, retry research, or stop with this target
    """
    sales_cfg = SalesConfiguration.from_runnable_config(config)
    report = state.get("final_report", "")

    result = await invoke_llm(
        system_prompt=CHECK_RESEARCH_SUFFICIENCY_PROMPT,
        user_message=report,
        model_name=sales_cfg.research_sufficiency_model,
        config=config,
        response_format=ResearchSufficiency,
    )

    update = {
        "research_sufficient": result.sufficient,
        "research_gaps": result.gaps,
        "reports": [Report(title=RESEARCH_REPORT_TITLE, content=report, is_markdown=True)],
    }
    decision = research_sufficiency_decision({**state, **update}, sales_cfg.max_research_retries)

    if decision == "retry":
        logging.warning(f"Research insufficient, retrying with gap focus: {result.gaps}")
        update["research_retry_count"] = state.get("research_retry_count", 0) + 1
        return Command(goto="prepare_research", update=update)

    if decision == "insufficient":
        return Command(goto="finish_target", update=update)

    # Contacts are wanted at every depth, including research-only: "find me who to talk to
    # at these colleges" is a research request, and gating contact finding behind the
    # qualify intent would skip it for exactly the runs that exist to produce contacts.
    # The intent gate is applied after the card is built, in find_target_contacts.
    return Command(goto="find_target_contacts", update=update)


def _render_contact_card(card: ContactCard) -> str:
    """Render a contact card as markdown for the saved report.

    Every line carries the URL the detail was read from, so a human can check any of them
    in one click - which is the only way to tell a real address from a plausible one.
    """
    lines = [f"# Contact Card: {card.organization}", ""]
    lines.append(f"**Best route:** `{card.best_route}`" + (f" - {card.best_route_value}" if card.best_route_value else ""))
    if card.best_route_reason:
        lines.append(f"_{card.best_route_reason}_")

    if card.people:
        lines += ["", "## Named people"]
        for p in card.people:
            li = f" - [LinkedIn]({p.linkedin_url})" if p.linkedin_url else ""
            lines.append(f"- **{p.name}**{f' - {p.role}' if p.role else ''}{li} - found at: {p.source_url}")

    for heading, points in (
        ("Emails", card.emails), ("Phones", card.phones), ("LinkedIn", card.linkedin_urls)
    ):
        if points:
            lines += ["", f"## {heading}"]
            for pt in points:
                who = f" ({pt.belongs_to})" if pt.belongs_to else ""
                lines.append(f"- `{pt.value}` [{pt.kind}]{who} - found at: {pt.source_url}")

    if card.contact_page:
        lines += ["", f"**Contact page:** {card.contact_page}"]
    if card.postal_address:
        lines += ["", f"**Postal address:** {card.postal_address}"]

    lines += ["", "## Sources checked", ""]
    lines += [f"- {s}" for s in card.sources_checked] or ["- (none recorded)"]
    return "\n".join(lines)


async def find_target_contacts(state: AgentState, config: RunnableConfig) -> Command[Literal["score_target", "finish_target"]]:
    """Run the contact agent and record every route it evidenced.

    Runs unconditionally rather than at the research agent's discretion. Contact finding
    used to be an instruction in the research supervisor's prompt ("this is not optional"),
    and the supervisor skipped it silently: of 64 targets researched that way, 3 came back
    with an email and 1 with a phone number. Discretion over *effort* lives inside the
    contact agent's planner; whether it runs at all is not up for negotiation.

    Never fails the target. An unreachable target is still worth its research, and the
    thinness of a result is recorded on the card rather than raised.

    Args:
        state: Current state holding the target and its entity type
        config: Runtime configuration with the effort cap and evidence floor

    Returns:
        Command to scoring, carrying the contact card
    """
    research_cfg = Configuration.from_runnable_config(config)
    agent_cfg = AgentConfiguration.from_runnable_config(config)
    target: Target = state["current_target"]
    onward = "score_target" if _at_least(agent_cfg, OutreachIntent.QUALIFY) else "finish_target"

    result = await contact_agent.ainvoke(
        {
            "target_name": target.name,
            "target_website": target.website or "",
            "target_context": target.context or "",
            "entity_type": state.get("entity_type") or "",
        },
        config=config,
    )
    card: ContactCard | None = result.get("contact_card")

    if card is None:
        logging.warning(f"{target.name}: contact agent returned no card")
        return Command(goto=onward, update={"contact_card": None})

    # "Nothing found" is only credible once the sources were actually tried. Distinguishing
    # a genuine dead end from an early exit is the whole reason sources_checked exists -
    # without it, a two-call give-up and an eight-call exhaustive search look identical.
    if card.is_empty() and len(card.sources_checked) < research_cfg.min_contact_sources_checked:
        logging.warning(
            f"{target.name}: contact agent found nothing after only "
            f"{len(card.sources_checked)} source(s) - treating as an early exit, not a dead end"
        )

    return Command(
        goto=onward,
        update={
            "contact_card": card,
            "reports": [Report(title=CONTACT_CARD_TITLE, content=_render_contact_card(card), is_markdown=True)],
        },
    )


async def score_target(state: AgentState, config: RunnableConfig) -> Command[Literal["generate_materials", "finish_target"]]:
    """Score partnership fit, then route on qualification and intent depth.

    For a person target, scoring evaluates their employer: SCORE_LEAD_PROMPT assesses
    company partner-track fit and is meaningless applied to an individual.

    Args:
        state: Current state holding the research report and entity type
        config: Runtime configuration with model settings and score threshold

    Returns:
        Command to generate materials, or to stop with this target
    """
    sales_cfg = SalesConfiguration.from_runnable_config(config)
    agent_cfg = AgentConfiguration.from_runnable_config(config)
    target: Target = state["current_target"]

    # Score against an offer that applies to this kind of organization. Judging a college
    # by the company tracks rates it badly for a reason that cannot apply to it.
    entity_type = state.get("entity_type") or ""
    subject = f"{get_sales_context(entity_type)}\n\n# Research report\n\n{state.get('final_report', '')}"
    if entity_type == "person":
        subject = (
            "Score the partnership fit of the EMPLOYER of this person, not the "
            f"individual.\n\n{subject}"
        )

    verdict = await invoke_llm(
        system_prompt=SCORE_LEAD_PROMPT,
        user_message=subject,
        model_name=sales_cfg.lead_scoring_model,
        config=config,
        response_format=LeadScore,
    )

    score = f"{verdict.score:.1f}"
    qualified = qualification_decision(score, sales_cfg.lead_score_threshold) == "qualified"
    logging.info(
        f"{target.name} scored {score} [{verdict.track}] "
        f"({'qualified' if qualified else 'not qualified'}) - {verdict.reasoning[:120]}"
    )
    update = {
        "lead_score": score,
        "lead_qualified": qualified,
        "lead_track": verdict.track,
        "lead_reasoning": verdict.reasoning,
        "lead_angle": verdict.angle,
    }

    if not _at_least(agent_cfg, OutreachIntent.DRAFT):
        return Command(goto="finish_target", update=update)

    if not qualified:
        if agent_cfg.require_qualification:
            return Command(goto="finish_target", update=update)
        # The operator asked for outreach to this specific target. The score is advice -
        # they may know a relationship or context the research cannot see - so it is
        # recorded and overridden rather than treated as a veto.
        logging.info(f"{target.name} scored below threshold but qualification is not required")

    return Command(goto="generate_materials", update=update)


def _route_from_card(card: ContactCard) -> ContactRoute:
    """Read the outreach route off an evidenced contact card.

    Everything here was copied from a page the contact agent actually fetched, so nothing
    in the resulting route can be a name-pattern guess.
    """
    person = card.people[0] if card.people else None
    written = card.best_route in ("direct_email", "role_inbox_attn")
    email = card.best_route_value if written else (card.emails[0].value if card.emails else "")
    linked = card.best_route in ("linkedin_dm", "contact_form", "phone", "postal")
    return ContactRoute(
        recipient_name=person.name if person else "",
        recipient_role=person.role if person else "",
        email=email,
        route_type=card.best_route,
        route_url=card.best_route_value if linked else (card.contact_page or ""),
    )


# What to write, per channel. A LinkedIn connection note and a covering email to a role
# inbox are different pieces of craft, and writing one when the route needs the other is
# how a good draft still fails to reach anyone.
_CHANNEL_GUIDANCE: dict[str, str] = {
    "direct_email": "Channel: direct email to the named person. Write a normal outreach email.",
    "role_inbox_attn": (
        "Channel: a role inbox, not a personal one. A gatekeeper reads this first and "
        "forwards it, so open by naming who it is for and why it belongs with them, and "
        "put 'Attn: <name>, <title>' at the front of the subject line."
    ),
    "linkedin_dm": (
        "Channel: a LinkedIn connection note. Hard limit 300 characters - no subject "
        "line, no signature, no links. One specific reason for reaching out and one ask."
    ),
    "phone": (
        "Channel: a phone call. Write talking points, not an email: the opening line, "
        "the single reason this is worth their time, and the ask."
    ),
    "contact_form": "Channel: a website contact form. Keep it short and self-contained - assume no attachments and no formatting.",
    "postal": "Channel: a physical letter. Formal register, full postal salutation, and no links the reader cannot type.",
}


def _channel_guidance(route: ContactRoute) -> str:
    """Tell the writer which channel it is writing for."""
    return _CHANNEL_GUIDANCE.get(
        route.route_type,
        "Channel: email. No route was evidenced, so keep claims general and verifiable.",
    )


async def generate_materials(state: AgentState, config: RunnableConfig) -> Command[Literal["approve_send", "finish_target"]]:
    """Write the outreach report and email, and create a Gmail draft.

    A draft is always created; sending is a separate, gated step so a bad generation can
    never reach a real prospect without an explicit decision.

    Args:
        state: Current state holding the research report and target
        config: Runtime configuration with model settings

    Returns:
        Command to the approval gate when sending, else to finish this target
    """
    sales_cfg = SalesConfiguration.from_runnable_config(config)
    agent_cfg = AgentConfiguration.from_runnable_config(config)
    target: Target = state["current_target"]
    report = state.get("final_report", "")

    outreach = await invoke_llm(
        system_prompt=GENERATE_OUTREACH_REPORT_PROMPT,
        user_message=f"**Research Report:**\n\n{report}",
        model_name=sales_cfg.outreach_report_model,
        config=config,
    )
    outreach = await invoke_llm(
        system_prompt=PROOF_READER_PROMPT,
        user_message=outreach,
        model_name=sales_cfg.outreach_report_model,
        config=config,
    )

    # Who to address, and how. The contact agent already decided this from evidence, so
    # prefer its card; the LLM read of the report is only a fallback for callers that
    # skipped contact finding.
    card: ContactCard | None = state.get("contact_card")
    if card is not None:
        route = _route_from_card(card)
    else:
        route = await invoke_llm(
            system_prompt=SELECT_CONTACT_ROUTE_PROMPT,
            user_message=report,
            model_name=sales_cfg.email_model,
            config=config,
            response_format=ContactRoute,
        )
    logging.info(
        f"{target.name}: route={route.route_type} "
        f"recipient={route.recipient_name or '(none named)'} {route.recipient_role}"
    )

    brief = [get_sales_context(state.get("entity_type") or "")]
    if state.get("lead_track"):
        brief.append(f"Recommended partner track: {state['lead_track']}")
    if state.get("lead_angle"):
        brief.append(f"Opening angle to use: {state['lead_angle']}")
    if route.recipient_name:
        brief.append(f"Address this person: {route.recipient_name} ({route.recipient_role})")
    brief.append(_channel_guidance(route))
    guidance = "\n\n".join(brief)

    email = await invoke_llm(
        system_prompt=PERSONALIZE_EMAIL_PROMPT,
        user_message=(
            f"# **Lead & company Information:**\n\n{report}\n\n"
            f"# **Outreach brief**\n\n{guidance}\n\n"
            "# Outreach report Link:\n\nNo hosted report link is available - omit that "
            "line from the email entirely rather than including a placeholder."
        ),
        model_name=sales_cfg.email_model,
        config=config,
        response_format=EmailResponse,
    )

    grounding = await _check_email_grounding(email.email, report, sales_cfg, config)

    update = {
        "contact_route": route.route_type,
        "recipient_name": route.recipient_name,
        "email_grounded": grounding.grounded,
        "unsupported_claims": grounding.unsupported_claims,
        "reports": [
            Report(title="Outreach Report", content=outreach, is_markdown=True),
            Report(title=EMAIL_REPORT_TITLE, content=email.email, is_markdown=False),
        ]
    }

    # A supplied address wins over a discovered one: it came from the operator, not a model.
    recipient = target.email or route.email

    if not grounding.grounded:
        # The draft is kept for inspection but never leaves as a draft a human might send
        # unread. A confidently wrong claim to a real partner cannot be walked back.
        logging.warning(
            f"{target.name}: email makes {len(grounding.unsupported_claims)} unsupported "
            f"claim(s); no draft created. {grounding.unsupported_claims[:3]}"
        )
        return Command(goto="finish_target", update=update)

    if recipient:
        GmailTools().create_draft_email(
            recipient=recipient, subject=email.subject, email_content=email.email
        )
    else:
        logging.warning(
            f"No email address for {target.name} (best route: {route.route_type}); draft not created"
        )

    if _at_least(agent_cfg, OutreachIntent.SEND) and recipient:
        return Command(goto="approve_send", update=update)
    return Command(goto="finish_target", update=update)


async def _check_email_grounding(email_text, report, sales_cfg, config) -> EmailGrounding:
    """Verify the email's factual claims against the research it was written from.

    Fails closed: if the check itself errors, the email is treated as ungrounded rather
    than waved through, because the whole point is to stop an unverified claim reaching
    a real recipient.
    """
    try:
        result = await invoke_llm(
            system_prompt=CHECK_EMAIL_GROUNDING_PROMPT,
            user_message=f"# Research report\n\n{report}\n\n# Drafted email\n\n{email_text}",
            model_name=sales_cfg.research_sufficiency_model,
            config=config,
            response_format=EmailGrounding,
        )
    except Exception as e:
        logging.warning(f"Grounding check failed, treating email as ungrounded: {e}")
        return EmailGrounding(grounded=False, unsupported_claims=[f"grounding check failed: {e}"])
    return result


async def approve_send(state: AgentState, config: RunnableConfig) -> Command[Literal["send_email", "finish_target"]]:
    """Pause for explicit human approval before sending.

    Selecting the send intent is authorization, but in batch mode one toggle would
    otherwise fire N real emails unattended, so the default is to confirm each one.

    Args:
        state: Current state holding the drafted email and target
        config: Runtime configuration with the approval toggle

    Returns:
        Command to send, or to skip sending for this target
    """
    agent_cfg = AgentConfiguration.from_runnable_config(config)
    target: Target = state["current_target"]

    if not agent_cfg.require_send_approval:
        return Command(goto="send_email", update={"send_approved": True})

    decision = interrupt({
        "action": "confirm_send",
        "recipient": target.email,
        "target": target.name,
        "email": get_report(state.get("reports", []), EMAIL_REPORT_TITLE),
    })
    approved = decision is True or (isinstance(decision, dict) and decision.get("approved") is True)

    if not approved:
        logging.warning(f"Send rejected for {target.name}; keeping the draft")
        return Command(goto="finish_target", update={"send_approved": False})
    return Command(goto="send_email", update={"send_approved": True})


async def send_email(state: AgentState, config: RunnableConfig) -> Command[Literal["finish_target"]]:
    """Send the approved email.

    Args:
        state: Current state holding the approved email and target
        config: Runtime configuration (unused, kept for node signature consistency)

    Returns:
        Command to finish this target
    """
    target: Target = state["current_target"]
    body = get_report(state.get("reports", []), EMAIL_REPORT_TITLE)
    GmailTools().send_email(
        recipient=target.email,
        subject=f"PACE Uttarakhand - {target.name}",
        email_content=body,
    )
    logging.info(f"Email sent to {target.email}")
    return Command(goto="finish_target")


async def finish_target(state: AgentState, config: RunnableConfig) -> Command[Literal["next_target"]]:
    """Persist this target's output, write back to CRM if applicable, and reset state.

    CRM write-back only happens for sheet-sourced targets, since an inline or page-derived
    name has no row to update. Every per-target field is reset here: anything left behind
    carries into the next target, which previously leaked one lead's report link into the
    next lead's email.

    Args:
        state: Current state holding this target's results
        config: Runtime configuration for the lead loader

    Returns:
        Command back to the loop head
    """
    target: Target = state["current_target"]
    reports = state.get("reports", [])
    if reports:
        # Prefix with the target name before saving: report titles are identical across
        # targets ("General Lead Research Report"), and save_reports_locally names files
        # by title, so an N-target run would otherwise leave only the last one on disk.
        safe_name = re.sub(r"[^\w\-. ]", "_", target.name).strip()[:60]
        save_reports_locally([
            Report(title=f"{safe_name} - {r.title}", content=r.content, is_markdown=r.is_markdown)
            for r in reports
        ])

    if target.source == "sheet" and target.crm_row_id:
        if not state.get("research_sufficient", True):
            status = "NEEDS_MORE_RESEARCH"
        elif state.get("lead_qualified"):
            status = "ATTEMPTED_TO_CONTACT"
        else:
            status = "NOT_QUALIFIED"
        get_lead_loader(config).update_record(target.crm_row_id, {
            "Status": status,
            "Score": state.get("lead_score", "N/A"),
            "Last Contacted": get_current_date(),
        })

    # Counts runs of failures, not the total: scattered failures across a long batch are
    # normal, but a run of them means something systemic (quota, network) and the batch
    # should stop rather than consume the queue.
    failed = state.get("current_target_failed", False)
    consecutive = state.get("consecutive_failures", 0) + 1 if failed else 0

    return Command(
        goto="next_target",
        update={
            **PER_TARGET_RESET,
            "targets_remaining": max(0, state.get("targets_remaining", 1) - 1),
            "consecutive_failures": consecutive,
        },
    )


# Unified Agent Graph Construction
# One prompt in, one target list, then research -> qualify -> draft -> send as deep as
# the configured intent allows.

def build_unified_agent(research_node=deep_researcher, checkpointer=None):
    """Build and compile the unified graph.

    The research node is a parameter so tests can substitute a stub: it is a compiled
    subgraph bound at construction time, so patching the module attribute afterwards
    cannot reach it, and running the real one would spend model and search quota.

    Args:
        research_node: Compiled graph (or callable) used for the research phase
        checkpointer: Persistence for interrupt/resume. Required by any caller that runs
            the send intent, because interrupt() needs somewhere to store the paused run.
            Left None for the module-level graph: LangGraph's own platform supplies one at
            deploy time and rejects a graph that brings its own.

    Returns:
        The compiled unified agent graph
    """
    builder = StateGraph(
        AgentState,
        input=AgentInputState,
        config_schema=AgentConfiguration,
    )

    # start_run and next_target are deliberately not isolated: a failure there is about the
    # run as a whole, not one target, and should surface rather than be swallowed.
    builder.add_node("start_run", start_run)
    builder.add_node("next_target", next_target)                          # Per-target loop head
    builder.add_node("prepare_research", _isolated(prepare_research))
    builder.add_node("research", research_node)                           # Nested research subgraph
    builder.add_node("check_research_sufficiency", _isolated(check_research_sufficiency))
    builder.add_node("find_target_contacts", _isolated(find_target_contacts))  # Contact-finding subgraph
    builder.add_node("score_target", _isolated(score_target))
    builder.add_node("generate_materials", _isolated(generate_materials))
    builder.add_node("approve_send", _isolated(approve_send))             # Human-in-the-loop gate
    builder.add_node("send_email", _isolated(send_email))
    builder.add_node("finish_target", _isolated_finish(finish_target))

    # A compiled subgraph node cannot return a Command, so its one outgoing edge is declared.
    builder.add_edge(START, "start_run")
    builder.add_edge("research", "check_research_sufficiency")

    return builder.compile(checkpointer=checkpointer)


unified_agent = build_unified_agent()
