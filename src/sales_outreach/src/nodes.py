from colorama import Fore, Style
from .tools.base.gmail_tools import GmailTools
from .tools.google_docs_tools import GoogleDocsManager
from .tools.lead_research import extract_company_name
from .prompts import *
from .state import LeadData, CompanyData, Report, GraphInputState, GraphState
from .structured_outputs import EmailResponse, ResearchSufficiency
from .utils import invoke_llm, get_report, get_current_date, save_reports_locally

# Enable or disable sending emails directly using GMAIL
# Should be confident about the quality of the email
SEND_EMAIL_DIRECTLY = False
# Enable or disable saving emails to Google Docs
# By defauly all reports are save locally in `reports` folder
SAVE_TO_GOOGLE_DOCS = False

class OutReachAutomationNodes:
    def __init__(self, loader):
        self.lead_loader = loader
        self.docs_manager = GoogleDocsManager()
        self.drive_folder_name = ""

    def get_new_leads(self, state: GraphInputState):
        print(Fore.YELLOW + "----- Fetching new leads -----\n" + Style.RESET_ALL)
        
        # Fetch new leads using the provided loader
        raw_leads = self.lead_loader.fetch_records()
        
        # Structure the leads
        leads = [
            LeadData(
                id=lead["id"],
                name=f'{lead.get("First Name", "")} {lead.get("Last Name", "")}',
                email=lead.get("Email", ""),
                phone=lead.get("Phone", ""),
                address=lead.get("Address", ""),
                profile="", # will be constructed
                company_name=lead.get("Company Name", ""),
                company_website=lead.get("Company Website", ""),
            )
            for lead in raw_leads
        ]
        
        print(Fore.YELLOW + f"----- Fetched {len(leads)} leads -----\n" + Style.RESET_ALL)
        return {"leads_data": leads, "number_leads": len(leads)}
    
    @staticmethod
    def check_for_remaining_leads(state: GraphState):
        """Checks for remaining leads and updates lead_data in the state."""
        print(Fore.YELLOW + "----- Checking for remaining leads -----\n" + Style.RESET_ALL)
        
        current_lead = None
        if state["leads_data"]:
            current_lead = state["leads_data"].pop()
        return {"current_lead": current_lead}

    @staticmethod
    def check_if_there_more_leads(state: GraphState):
        # Number of leads remaining
        num_leads = state["number_leads"]
        if num_leads > 0:
            print(Fore.YELLOW + f"----- Found {num_leads} more leads -----\n" + Style.RESET_ALL)
            return "Found leads"
        else:
            print(Fore.GREEN + "----- Finished, No more leads -----\n" + Style.RESET_ALL)
            return "No more leads"

    async def run_shared_research(self, state: GraphState):
        """Research the lead (person, if named) and their company using the shared
        open_deep_research core (entity-aware, fallback model, citation verification,
        contact-finder module) instead of kaymen99's original LinkedIn/SERPER-based
        pipeline - which needed RAPIDAPI/SERPER keys that were never configured in this
        project and never actually worked.

        Runs two research passes when a real lead name is known - one for the person
        (role, background, LinkedIn) and one for their company (what they do, size,
        financials) - matching the old pipeline's fetch_linkedin_profile_data (person)
        + review_company_website (company) split, just via the shared core instead of
        two separate broken tool sets.
        """
        print(Fore.YELLOW + "----- Running shared research core -----\n" + Style.RESET_ALL)
        from langchain_core.messages import HumanMessage as ODRHumanMessage
        from open_deep_research.deep_researcher import deep_researcher

        lead_data = state["current_lead"]
        company_name = lead_data.company_name or extract_company_name(lead_data.email)
        company_website = lead_data.company_website
        person_name = lead_data.name.strip()

        # If this is a retry after an insufficient-data verdict, steer both passes
        # explicitly at what was missing instead of repeating the same queries
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

        report_sections = []
        for label, query in queries:
            result = await deep_researcher.ainvoke(
                {"messages": [ODRHumanMessage(content=query)]},
                config={"configurable": {"allow_clarification": False}}
            )
            report_sections.append(f"## {label} Research\n\n{result.get('final_report', '')}")

        research_report = "\n\n---\n\n".join(report_sections)

        company_data = state.get("company_data") or CompanyData()
        company_data.name = company_name
        company_data.website = company_website

        # Update folder name for saving reports in Drive
        self.drive_folder_name = f"{lead_data.name}_{company_data.name}".strip("_")

        report = Report(
            title="General Lead Research Report",
            content=research_report,
            is_markdown=True
        )
        return {
            "current_lead": lead_data,
            "company_data": company_data,
            "reports": [report],
            "research_retry_count": retry_count
        }

    @staticmethod
    def check_research_sufficiency(state: GraphState):
        """Check whether the research report has enough real substance for a credible
        pitch, instead of either duplicating research or generating a weak, generic
        email from thin data.
        """
        print(Fore.YELLOW + "----- Checking research sufficiency -----\n" + Style.RESET_ALL)
        reports = state["reports"]
        research_report = get_report(reports, "General Lead Research Report")

        result = invoke_llm(
            system_prompt=CHECK_RESEARCH_SUFFICIENCY_PROMPT,
            user_message=research_report,
            model="gemini-3.1-flash-lite",
            response_format=ResearchSufficiency
        )

        if not result.sufficient:
            print(Fore.RED + f"----- Research insufficient: {result.gaps} -----\n" + Style.RESET_ALL)

        return {
            "research_sufficient": result.sufficient,
            "research_gaps": result.gaps
        }

    @staticmethod
    def check_if_research_sufficient(state: GraphState):
        if state.get("research_sufficient"):
            return "sufficient"
        # One retry with a gap-focused query before giving up - not unlimited retries
        if state.get("research_retry_count", 0) < 1:
            return "retry"
        return "insufficient"

    @staticmethod
    def score_lead(state: GraphState):
        """
        Score the lead based on the company profile and open positions.

        @param state: The current state of the application.
        @return: Updated state with the lead score.
        """
        print(Fore.YELLOW + "----- Scoring lead -----\n" + Style.RESET_ALL)
        
        # Load reports
        reports = state["reports"]
        research_report = get_report(reports, "General Lead Research Report")

        # Scoring lead
        lead_score = invoke_llm(
            system_prompt=SCORE_LEAD_PROMPT,
            user_message=research_report,
            model="gemini-3.1-flash-lite"
        )
        return {"lead_score": lead_score.strip()}

    @staticmethod
    def is_lead_qualified(state: GraphState):
        """
        Check if the lead is qualified based on the lead score.

        @param state: The current state of the application.
        @return: Updated state with the qualification status.
        """
        print(Fore.YELLOW + "----- Checking if lead is qualified -----\n" + Style.RESET_ALL)
        return {"reports": []}

    @staticmethod
    def check_if_qualified(state: GraphState):
        """
        Check if the lead is qualified based on the lead score.

        @param state: The current state of the application.
        @return: Updated state with the qualification status.
        """
        # Checking if the lead score is 7 or higher
        print(f"Score: {state['lead_score']}")
        is_qualified = float(state["lead_score"]) >= 7
        if is_qualified:
            print(Fore.GREEN + "Lead is qualified\n" + Style.RESET_ALL)
            return "qualified"
        else:
            print(Fore.RED + "Lead is not qualified\n" + Style.RESET_ALL)
            return "not qualified"
    
    @staticmethod
    def create_outreach_materials(state: GraphState):
        return {"reports": []}
    
    def generate_custom_outreach_report(self, state: GraphState):
        print(Fore.YELLOW + "----- Crafting Custom outreach report based on gathered information -----\n" + Style.RESET_ALL)
        
        # Load reports
        reports = state["reports"]
        research_report = get_report(reports, "General Lead Research Report")

        inputs = f"""
        **Research Report:**

        {research_report}
        """

        # Generate report
        custom_outreach_report = invoke_llm(
            system_prompt=GENERATE_OUTREACH_REPORT_PROMPT,
            user_message=inputs,
            model="gemini-3.1-flash-lite"
        )

        # Call our editor/proof-reader agent
        revised_outreach_report = invoke_llm(
            system_prompt=PROOF_READER_PROMPT,
            user_message=custom_outreach_report,
            model="gemini-3.1-flash-lite"
        )
        
        # Store report into google docs and get shareable link
        new_doc = self.docs_manager.add_document(
            content=revised_outreach_report,
            doc_title="Outreach Report",
            folder_name=self.drive_folder_name,
            make_shareable=True,
            folder_shareable=True, # Set to false if only personal or true if with a team
            markdown=True
        )  
        
        return {
            "custom_outreach_report_link": new_doc["shareable_url"],
            "reports_folder_link": new_doc["folder_url"]
        }

    def generate_personalized_email(self, state: GraphState):
        """
        Generate a personalized email for the lead.

        @param state: The current state of the application.
        @return: Updated state with the generated email.
        """
        print(Fore.YELLOW + "----- Generating personalized email -----\n" + Style.RESET_ALL)
        
        # Load reports
        reports = state["reports"]
        general_lead_search_report = get_report(reports, "General Lead Research Report")
        
        lead_data = f"""
        # **Lead & company Information:**

        {general_lead_search_report}

        # Outreach report Link:

        {state["custom_outreach_report_link"]}
        """
        output = invoke_llm(
            system_prompt=PERSONALIZE_EMAIL_PROMPT,
            user_message=lead_data,
            model="gemini-3.1-flash-lite",
            response_format=EmailResponse
        )
        
        # Get relevant fields
        subject = output.subject
        personalized_email = output.email
        
        # Get lead email
        email = state["current_lead"].email
        
        # Create draft email
        gmail = GmailTools()
        gmail.create_draft_email(
            recipient=email,
            subject=subject,
            email_content=personalized_email
        )
        
        # Send email directly
        if SEND_EMAIL_DIRECTLY:
            gmail.send_email(
                recipient=email,
                subject=subject,
                email_content=personalized_email
            )
        
        # Save email with reports for reference
        personalized_email_doc = Report(
            title="Personalized Email",
            content=personalized_email,
            is_markdown=False
        )
        return {"reports": [personalized_email_doc]}

    def generate_interview_script(self, state: GraphState):
        print(Fore.YELLOW + "----- Generating interview script -----\n" + Style.RESET_ALL)
        
        # Load reports
        reports = state["reports"]
        research_report = get_report(reports, "General Lead Research Report")

        # Generating SPIN questions
        spin_questions = invoke_llm(
            system_prompt=GENERATE_SPIN_QUESTIONS_PROMPT,
            user_message=research_report,
            model="gemini-3.1-flash-lite"
        )

        inputs = f"""
        # **Lead & company Information:**

        {research_report}

        # **SPIN questions:**

        {spin_questions}
        """
        
        # Generating interview script
        interview_script = invoke_llm(
            system_prompt=WRITE_INTERVIEW_SCRIPT_PROMPT,
            user_message=inputs,
            model="gemini-3.1-flash-lite"
        )
        
        interview_script_doc = Report(
            title="Interview Script",
            content=interview_script,
            is_markdown=True
        )
        
        return {"reports": [interview_script_doc]}
    
    @staticmethod
    def await_reports_creation(state: GraphState):
        return {"reports": []}
    
    def save_reports_to_google_docs(self, state: GraphState):
        print(Fore.YELLOW + "----- Save Reports to Google Docs -----\n" + Style.RESET_ALL)
        
        # Load all reports
        reports = state["reports"]
        
        # Ensure reports are saved locally
        save_reports_locally(reports)
        
        # Save all reports to Google docs
        if SAVE_TO_GOOGLE_DOCS:
            for report in reports:
                self.docs_manager.add_document(
                    content=report.content,
                    doc_title=report.title,
                    folder_name=self.drive_folder_name,
                    markdown=report.is_markdown
                )

        return state

    def update_CRM(self, state: GraphState):
        print(Fore.YELLOW + "----- Updating CRM records -----\n" + Style.RESET_ALL)

        # The "insufficient research" and "not qualified" paths skip scoring/outreach
        # generation entirely, so those fields may never have been set - status reflects
        # which stage this lead actually stopped at rather than always claiming contact.
        if not state.get("research_sufficient", True):
            status = "NEEDS_MORE_RESEARCH"
        elif "custom_outreach_report_link" in state:
            status = "ATTEMPTED_TO_CONTACT"
        else:
            status = "NOT_QUALIFIED"

        new_data = {
            "Status": status,
            "Score": state.get("lead_score", "N/A"),
            "Analysis Reports": state.get("reports_folder_link", ""),
            "Outreach Report": state.get("custom_outreach_report_link", ""),
            "Last Contacted": get_current_date()
        }
        self.lead_loader.update_record(state["current_lead"].id, new_data)
        
        # reset reports list
        state["reports"] = []
        
        return {"number_leads": state["number_leads"] - 1}