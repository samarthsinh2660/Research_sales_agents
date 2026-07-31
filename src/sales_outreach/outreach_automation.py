"""Main LangGraph implementation for the sales outreach agent.

Research is delegated entirely to the shared open_deep_research core - this graph is
responsible only for the sales-specific work: judging whether research is good enough to
pitch on, scoring partnership fit, generating outreach materials, and writing back to CRM.
"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from sales_outreach.configuration import SalesConfiguration
from sales_outreach.prompts import (
    CHECK_RESEARCH_SUFFICIENCY_PROMPT,
    GENERATE_OUTREACH_REPORT_PROMPT,
    GENERATE_SPIN_QUESTIONS_PROMPT,
    PERSONALIZE_EMAIL_PROMPT,
    PROOF_READER_PROMPT,
    SCORE_LEAD_PROMPT,
    WRITE_INTERVIEW_SCRIPT_PROMPT,
)
from sales_outreach.state import (
    CompanyData,
    EmailResponse,
    GraphInputState,
    GraphState,
    LeadData,
    Report,
    ResearchSufficiency,
)
from sales_outreach.tools.base.gmail_tools import GmailTools
from sales_outreach.tools.lead_research import extract_company_name
from sales_outreach.utils import (
    get_current_date,
    get_docs_manager,
    get_lead_loader,
    get_report,
    has_remaining_leads,
    invoke_llm,
    qualification_decision,
    research_sufficiency_decision,
    save_reports_locally,
)

RESEARCH_REPORT_TITLE = "General Lead Research Report"


async def get_new_leads(state: GraphInputState, config: RunnableConfig) -> Command[Literal["check_for_remaining_leads"]]:
    """Fetch new leads from the configured lead source.

    Args:
        state: Graph input state, optionally naming specific lead ids
        config: Runtime configuration selecting the lead source

    Returns:
        Command to proceed to the per-lead loop with the fetched leads
    """
    logging.info("Fetching new leads")
    lead_loader = get_lead_loader(config)
    raw_leads = lead_loader.fetch_records()

    leads = [
        LeadData(
            id=lead["id"],
            name=f'{lead.get("First Name", "")} {lead.get("Last Name", "")}'.strip(),
            email=lead.get("Email", ""),
            phone=lead.get("Phone", ""),
            address=lead.get("Address", ""),
            profile="",  # will be constructed by the research pass
            company_name=lead.get("Company Name", ""),
            company_website=lead.get("Company Website", ""),
        )
        for lead in raw_leads
    ]

    logging.info(f"Fetched {len(leads)} leads")
    return Command(
        goto="check_for_remaining_leads",
        update={"leads_data": leads, "number_leads": len(leads)}
    )


async def check_for_remaining_leads(state: GraphState, config: RunnableConfig) -> Command[Literal["run_shared_research", "__end__"]]:
    """Pop the next lead to process, or end when the queue is empty.

    The shortened list is returned in the state update rather than popped in place -
    in-place mutation of state happens to survive in-memory runs but is silently lost
    under any checkpointer.

    Args:
        state: Current graph state holding the remaining leads
        config: Runtime configuration (unused, kept for node signature consistency)

    Returns:
        Command to research the next lead, or to end when none remain
    """
    if has_remaining_leads(state) == "No more leads":
        logging.info("Finished - no more leads")
        return Command(goto=END)

    remaining = list(state["leads_data"])
    current_lead = remaining.pop() if remaining else None
    logging.info(f"{state['number_leads']} lead(s) remaining")

    return Command(
        goto="run_shared_research",
        update={"current_lead": current_lead, "leads_data": remaining}
    )


async def run_shared_research(state: GraphState, config: RunnableConfig) -> Command[Literal["check_research_sufficiency"]]:
    """Research the lead and their company via the shared open_deep_research core.

    Runs two passes when a real lead name is known - one for the person (role, background,
    LinkedIn) and one for their company - rather than maintaining a second, separate
    research pipeline here. On a retry, the query is steered explicitly at the gap the
    sufficiency check identified instead of repeating the same request.

    Args:
        state: Current graph state holding the lead being processed
        config: Runtime configuration, forwarded to the research core

    Returns:
        Command to proceed to the research sufficiency check
    """
    logging.info("Running shared research core")
    from open_deep_research.deep_researcher import deep_researcher

    lead_data = state["current_lead"]
    company_name = lead_data.company_name or extract_company_name(lead_data.email)
    company_website = lead_data.company_website
    person_name = lead_data.name.strip()

    research_gaps = state.get("research_gaps", "")
    retry_count = state.get("research_retry_count", 0)
    gap_instruction = ""
    if research_gaps:
        gap_instruction = f" A previous research pass found this gap - focus specifically on filling it this time: {research_gaps}"
        retry_count += 1

    queries = []
    if person_name:
        person_query = f"Research {person_name}, who works at {company_name}."
        if lead_data.email:
            person_query += f" Contact email: {lead_data.email}."
        person_query += " Find their current role, professional background, and public LinkedIn profile."
        queries.append(("Person", person_query + gap_instruction))

    company_query = f"Research {company_name}"
    if company_website:
        company_query += f" ({company_website})"
    company_query += " - what they do, services/products offered, company size, and any named contacts or contact information."
    queries.append(("Company", company_query + gap_instruction))

    # Merge rather than replace the parent config, so runtime model/search settings
    # actually reach the research sub-graph instead of being silently discarded.
    research_config = {
        **config,
        "configurable": {**config.get("configurable", {}), "allow_clarification": False},
    }

    report_sections = []
    for label, query in queries:
        result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=research_config,
        )
        report_sections.append(f"## {label} Research\n\n{result.get('final_report', '')}")

    company_data = state.get("company_data") or CompanyData()
    company_data.name = company_name
    company_data.website = company_website

    report = Report(
        title=RESEARCH_REPORT_TITLE,
        content="\n\n---\n\n".join(report_sections),
        is_markdown=True,
    )
    return Command(
        goto="check_research_sufficiency",
        update={
            "current_lead": lead_data,
            "company_data": company_data,
            "reports": [report],
            "research_retry_count": retry_count,
            "drive_folder_name": f"{lead_data.name}_{company_name}".strip("_"),
        }
    )


async def check_research_sufficiency(state: GraphState, config: RunnableConfig) -> Command[Literal["score_lead", "run_shared_research", "save_reports_to_google_docs"]]:
    """Gate on whether research has enough substance to write a credible pitch.

    Prevents the two failure modes at either extreme: duplicating research work here, and
    generating a weak, generic email from thin data. One gap-focused retry is allowed
    before giving up and flagging the lead as needing more research.

    Args:
        state: Current graph state holding the research report
        config: Runtime configuration with model settings and retry budget

    Returns:
        Command to score the lead, retry research, or bail out to reporting
    """
    logging.info("Checking research sufficiency")
    configurable = SalesConfiguration.from_runnable_config(config)
    research_report = get_report(state["reports"], RESEARCH_REPORT_TITLE)

    result = await invoke_llm(
        system_prompt=CHECK_RESEARCH_SUFFICIENCY_PROMPT,
        user_message=research_report,
        model_name=configurable.research_sufficiency_model,
        config=config,
        response_format=ResearchSufficiency,
    )

    update = {"research_sufficient": result.sufficient, "research_gaps": result.gaps}
    decision = research_sufficiency_decision(
        {**state, **update}, configurable.max_research_retries
    )

    if decision == "sufficient":
        return Command(goto="score_lead", update=update)
    if decision == "retry":
        logging.warning(f"Research insufficient, retrying with gap focus: {result.gaps}")
        return Command(goto="run_shared_research", update=update)

    logging.warning(f"Research still insufficient after retry: {result.gaps}")
    return Command(goto="save_reports_to_google_docs", update=update)


async def score_lead(state: GraphState, config: RunnableConfig) -> Command[Literal["generate_custom_outreach_report", "save_reports_to_google_docs"]]:
    """Score how well the company fits a PACE Uttarakhand partner track, then route.

    Qualification is recorded explicitly in state rather than inferred later from whether
    a Google Docs link exists - that inference would mislabel qualified leads whenever
    Docs saving is disabled.

    Args:
        state: Current graph state holding the research report
        config: Runtime configuration with model settings and score threshold

    Returns:
        Command to generate outreach materials, or to skip to reporting if unqualified
    """
    logging.info("Scoring lead")
    configurable = SalesConfiguration.from_runnable_config(config)
    research_report = get_report(state["reports"], RESEARCH_REPORT_TITLE)

    lead_score = await invoke_llm(
        system_prompt=SCORE_LEAD_PROMPT,
        user_message=research_report,
        model_name=configurable.lead_scoring_model,
        config=config,
    )
    lead_score = lead_score.strip()

    decision = qualification_decision(lead_score, configurable.lead_score_threshold)
    qualified = decision == "qualified"
    logging.info(f"Lead score: {lead_score} ({decision})")

    update = {"lead_score": lead_score, "lead_qualified": qualified}
    if qualified:
        return Command(goto="generate_custom_outreach_report", update=update)
    return Command(goto="save_reports_to_google_docs", update=update)


async def generate_custom_outreach_report(state: GraphState, config: RunnableConfig) -> Command[Literal["generate_personalized_email", "generate_interview_script"]]:
    """Write and proofread the outreach report, then fan out to the two material generators.

    Args:
        state: Current graph state holding the research report
        config: Runtime configuration with model settings and the Docs toggle

    Returns:
        Command dispatching both outreach material nodes in parallel
    """
    logging.info("Crafting custom outreach report")
    configurable = SalesConfiguration.from_runnable_config(config)
    research_report = get_report(state["reports"], RESEARCH_REPORT_TITLE)

    custom_outreach_report = await invoke_llm(
        system_prompt=GENERATE_OUTREACH_REPORT_PROMPT,
        user_message=f"**Research Report:**\n\n{research_report}",
        model_name=configurable.outreach_report_model,
        config=config,
    )
    revised_outreach_report = await invoke_llm(
        system_prompt=PROOF_READER_PROMPT,
        user_message=custom_outreach_report,
        model_name=configurable.outreach_report_model,
        config=config,
    )

    update = {
        "reports": [Report(
            title="Outreach Report",
            content=revised_outreach_report,
            is_markdown=True,
        )]
    }

    # The email links to this report, so it only gets a link when Docs saving is on.
    if configurable.save_to_google_docs:
        new_doc = get_docs_manager().add_document(
            content=revised_outreach_report,
            doc_title="Outreach Report",
            folder_name=state.get("drive_folder_name", ""),
            make_shareable=True,
            folder_shareable=True,  # Set to false if only personal, true if with a team
            markdown=True,
        )
        update["custom_outreach_report_link"] = new_doc["shareable_url"]
        update["reports_folder_link"] = new_doc["folder_url"]

    return Command(
        goto=["generate_personalized_email", "generate_interview_script"],
        update=update
    )


async def generate_personalized_email(state: GraphState, config: RunnableConfig) -> Command[Literal["save_reports_to_google_docs"]]:
    """Generate a personalized outreach email and create a Gmail draft.

    Creating a draft is always safe; actually sending is gated behind an explicit config
    flag that defaults off, so a bad generation can never auto-send to a real prospect.

    Args:
        state: Current graph state holding the research report and lead
        config: Runtime configuration with model settings and the send toggle

    Returns:
        Command to proceed to report saving
    """
    logging.info("Generating personalized email")
    configurable = SalesConfiguration.from_runnable_config(config)
    research_report = get_report(state["reports"], RESEARCH_REPORT_TITLE)

    report_link = state.get("custom_outreach_report_link")
    link_section = (
        f"# Outreach report Link:\n\n{report_link}"
        if report_link
        else "# Outreach report Link:\n\nNo hosted report link is available - omit that line from the email entirely rather than including a broken or placeholder link."
    )
    lead_context = f"# **Lead & company Information:**\n\n{research_report}\n\n{link_section}"

    output = await invoke_llm(
        system_prompt=PERSONALIZE_EMAIL_PROMPT,
        user_message=lead_context,
        model_name=configurable.email_model,
        config=config,
        response_format=EmailResponse,
    )

    recipient = state["current_lead"].email
    gmail = GmailTools()
    gmail.create_draft_email(
        recipient=recipient,
        subject=output.subject,
        email_content=output.email,
    )

    if configurable.send_email_directly:
        gmail.send_email(
            recipient=recipient,
            subject=output.subject,
            email_content=output.email,
        )

    return Command(
        goto="save_reports_to_google_docs",
        update={"reports": [Report(
            title="Personalized Email",
            content=output.email,
            is_markdown=False,
        )]}
    )


async def generate_interview_script(state: GraphState, config: RunnableConfig) -> Command[Literal["save_reports_to_google_docs"]]:
    """Generate SPIN questions and a partnership call script for the lead.

    Args:
        state: Current graph state holding the research report
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to report saving
    """
    logging.info("Generating interview script")
    configurable = SalesConfiguration.from_runnable_config(config)
    research_report = get_report(state["reports"], RESEARCH_REPORT_TITLE)

    spin_questions = await invoke_llm(
        system_prompt=GENERATE_SPIN_QUESTIONS_PROMPT,
        user_message=research_report,
        model_name=configurable.interview_script_model,
        config=config,
    )
    interview_script = await invoke_llm(
        system_prompt=WRITE_INTERVIEW_SCRIPT_PROMPT,
        user_message=f"# **Lead & company Information:**\n\n{research_report}\n\n# **SPIN questions:**\n\n{spin_questions}",
        model_name=configurable.interview_script_model,
        config=config,
    )

    return Command(
        goto="save_reports_to_google_docs",
        update={"reports": [Report(
            title="Interview Script",
            content=interview_script,
            is_markdown=True,
        )]}
    )


async def save_reports_to_google_docs(state: GraphState, config: RunnableConfig) -> Command[Literal["update_CRM"]]:
    """Persist all generated reports, locally always and to Google Docs when enabled.

    Also the fan-in point for the two parallel material generators: LangGraph's per-node
    trigger channels tolerate multiple writes in one superstep, so this node runs exactly
    once after both branches complete.

    Args:
        state: Current graph state holding all generated reports
        config: Runtime configuration with the Docs toggle

    Returns:
        Command to proceed to the CRM update
    """
    logging.info("Saving reports")
    configurable = SalesConfiguration.from_runnable_config(config)
    reports = state["reports"]

    save_reports_locally(reports)

    if configurable.save_to_google_docs:
        docs_manager = get_docs_manager()
        for report in reports:
            docs_manager.add_document(
                content=report.content,
                doc_title=report.title,
                folder_name=state.get("drive_folder_name", ""),
                markdown=report.is_markdown,
            )

    return Command(goto="update_CRM")


async def update_CRM(state: GraphState, config: RunnableConfig) -> Command[Literal["check_for_remaining_leads"]]:
    """Write the outcome back to the CRM and reset per-lead state.

    Status reflects the stage this lead actually stopped at. Reports are cleared through
    the override reducer - assigning to state directly does not clear an accumulating
    channel, which previously leaked one lead's reports into the next lead's prompts.

    Args:
        state: Current graph state holding the outcome of this lead
        config: Runtime configuration selecting the lead source

    Returns:
        Command to loop back for the next lead
    """
    logging.info("Updating CRM records")

    if not state.get("research_sufficient", True):
        status = "NEEDS_MORE_RESEARCH"
    elif state.get("lead_qualified"):
        status = "ATTEMPTED_TO_CONTACT"
    else:
        status = "NOT_QUALIFIED"

    new_data = {
        "Status": status,
        "Score": state.get("lead_score", "N/A"),
        "Analysis Reports": state.get("reports_folder_link", ""),
        "Outreach Report": state.get("custom_outreach_report_link", ""),
        "Last Contacted": get_current_date(),
    }
    get_lead_loader(config).update_record(state["current_lead"].id, new_data)

    # Every per-lead field must be reset here, not just reports. Anything left behind
    # carries into the next lead: a stale custom_outreach_report_link would put lead N's
    # report link in lead N+1's email, stale research_gaps would contaminate the next
    # lead's very first research query, and a spent research_retry_count would deny the
    # next lead its retry budget.
    return Command(
        goto="check_for_remaining_leads",
        update={
            "number_leads": state["number_leads"] - 1,
            "reports": {"type": "override", "value": []},
            "research_sufficient": False,
            "research_gaps": "",
            "research_retry_count": 0,
            "lead_score": "",
            "lead_qualified": False,
            "custom_outreach_report_link": "",
            "reports_folder_link": "",
            "drive_folder_name": "",
        }
    )


# Sales Outreach Graph Construction
# Loops over leads: research -> qualify -> generate materials -> write back to CRM
outreach_automation_builder = StateGraph(
    GraphState,
    input=GraphInputState,
    config_schema=SalesConfiguration
)

outreach_automation_builder.add_node("get_new_leads", get_new_leads)
outreach_automation_builder.add_node("check_for_remaining_leads", check_for_remaining_leads)          # Per-lead loop head
outreach_automation_builder.add_node("run_shared_research", run_shared_research)
outreach_automation_builder.add_node("check_research_sufficiency", check_research_sufficiency)
outreach_automation_builder.add_node("score_lead", score_lead)
outreach_automation_builder.add_node("generate_custom_outreach_report", generate_custom_outreach_report)  # Fans out to both generators below
outreach_automation_builder.add_node("generate_personalized_email", generate_personalized_email)
outreach_automation_builder.add_node("generate_interview_script", generate_interview_script)
outreach_automation_builder.add_node("save_reports_to_google_docs", save_reports_to_google_docs)      # Fan-in point for both generators
outreach_automation_builder.add_node("update_CRM", update_CRM)

# The only edge that isn't expressed by Command(goto=...) returned from a node
outreach_automation_builder.add_edge(START, "get_new_leads")

outreach_automation = outreach_automation_builder.compile()
