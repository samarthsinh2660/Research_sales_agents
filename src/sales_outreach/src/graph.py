from langgraph.graph import END, StateGraph
from .nodes import OutReachAutomationNodes
from .state import GraphState
from .tools.leads_loader.lead_loader_base import LeadLoaderBase


class OutReachAutomation:
    def __init__(self, loader: LeadLoaderBase):
        # Initialize the automation workflow by building the graph
        self.app = self.build_graph(loader)

    def build_graph(self, loader):
        """
        Constructs the state graph for the outreach automation workflow.
        """
        # Create the main graph with a predefined state
        graph = StateGraph(GraphState)
        
        # Initialize the nodes with the provided lead loader
        nodes = OutReachAutomationNodes(loader)

        # **Step 1: Adding nodes to the graph**
        # Fetch new leads from the CRM
        graph.add_node("get_new_leads", nodes.get_new_leads)
        graph.add_node("check_for_remaining_leads", nodes.check_for_remaining_leads)

        # Research phase: shared open_deep_research core replaces kaymen99's own
        # LinkedIn/SERPER-based pipeline (which needed unconfigured API keys and
        # never actually worked in this project)
        graph.add_node("run_shared_research", nodes.run_shared_research)
        graph.add_node("check_research_sufficiency", nodes.check_research_sufficiency)
        graph.add_node("score_lead", nodes.score_lead)

        # Outreach preparation phase
        graph.add_node("create_outreach_materials", nodes.create_outreach_materials)
        graph.add_node("generate_custom_outreach_report", nodes.generate_custom_outreach_report)
        graph.add_node("generate_personalized_email", nodes.generate_personalized_email)
        graph.add_node("generate_interview_script", nodes.generate_interview_script)

        # Reporting and finalization
        graph.add_node("save_reports_to_google_docs", nodes.save_reports_to_google_docs)
        graph.add_node("await_reports_creation", nodes.await_reports_creation)
        graph.add_node("update_CRM", nodes.update_CRM)

        # **Step 2: Setting up edges between nodes**

        # Entry point of the graph
        graph.set_entry_point("get_new_leads")

        # Transition from fetching leads to checking if there are leads to process
        graph.add_edge("get_new_leads", "check_for_remaining_leads")

        # Conditional logic for lead availability
        graph.add_conditional_edges(
            "check_for_remaining_leads",
            nodes.check_if_there_more_leads,
            {
                "Found leads": "run_shared_research",  # Proceed if leads are found
                "No more leads": END  # Terminate if no leads remain
            }
        )

        # Research phase: run the shared research core, then gate on whether it
        # actually found enough to write a credible pitch - rather than either
        # duplicating research here or generating a weak email from thin data.
        # One retry with a gap-focused query is allowed before giving up.
        graph.add_edge("run_shared_research", "check_research_sufficiency")
        graph.add_conditional_edges(
            "check_research_sufficiency",
            nodes.check_if_research_sufficient,
            {
                "sufficient": "score_lead",
                "retry": "run_shared_research",  # One more targeted pass at the specific gap
                "insufficient": "save_reports_to_google_docs"  # Still not enough after retry - flag and stop
            }
        )

        # Scoring phase with conditional qualification check
        graph.add_conditional_edges(
            "score_lead",
            nodes.check_if_qualified,
            {
                "qualified": "generate_custom_outreach_report",  # Proceed if lead is qualified
                "not qualified": "save_reports_to_google_docs"  # Save reports and exit if lead is unqualified
            }
        )

        # Outreach material creation
        graph.add_edge("generate_custom_outreach_report", "create_outreach_materials")
        graph.add_edge("create_outreach_materials", "generate_personalized_email")
        graph.add_edge("create_outreach_materials", "generate_interview_script")

        # Await completion and finalize reports
        graph.add_edge("generate_personalized_email", "await_reports_creation")
        graph.add_edge("generate_interview_script", "await_reports_creation")
        graph.add_edge("await_reports_creation", "save_reports_to_google_docs")

        # Save reports and update the CRM
        graph.add_edge("save_reports_to_google_docs", "update_CRM")

        # Loop back to check for remaining leads
        graph.add_edge("update_CRM", "check_for_remaining_leads")
        return graph.compile()

