"""Structural tests for the compiled outreach graph - no API calls, no credentials."""
from sales_outreach.outreach_automation import outreach_automation


def test_graph_compiles_without_credentials():
    """Importing and compiling must not trigger Google OAuth.

    Regression guard: GoogleDocsManager used to be constructed in the node class
    __init__, so building the graph ran the OAuth flow and crashed with
    FileNotFoundError whenever credentials.json was missing - even for runs that
    never touched Google Docs. Reaching this assertion at all proves it is lazy now.
    """
    assert outreach_automation is not None


def test_expected_nodes_present():
    nodes = set(outreach_automation.get_graph().nodes.keys())
    expected = {
        "get_new_leads",
        "check_for_remaining_leads",
        "run_shared_research",
        "check_research_sufficiency",
        "score_lead",
        "generate_custom_outreach_report",
        "generate_personalized_email",
        "generate_interview_script",
        "save_reports_to_google_docs",
        "update_CRM",
    }
    assert expected <= nodes


def test_outreach_materials_fan_out_and_fan_in():
    """Both material generators run in parallel and rejoin at report saving."""
    edges = {(e.source, e.target) for e in outreach_automation.get_graph().edges}

    assert ("generate_custom_outreach_report", "generate_personalized_email") in edges
    assert ("generate_custom_outreach_report", "generate_interview_script") in edges
    assert ("generate_personalized_email", "save_reports_to_google_docs") in edges
    assert ("generate_interview_script", "save_reports_to_google_docs") in edges


def test_per_lead_loop_closes():
    edges = {(e.source, e.target) for e in outreach_automation.get_graph().edges}
    assert ("update_CRM", "check_for_remaining_leads") in edges


def test_research_retry_path_exists():
    edges = {(e.source, e.target) for e in outreach_automation.get_graph().edges}
    assert ("check_research_sufficiency", "run_shared_research") in edges
