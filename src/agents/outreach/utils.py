"""Utility functions and helpers for the sales outreach agent."""

import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal, Type

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agents.outreach.configuration import LeadLoaderType, SalesConfiguration
from agents.research.utils import extract_answer_text, get_api_key_for_model

##########################
# Google Auth Utils
##########################

# Resolve OAuth/report paths against the package directory rather than the process
# cwd - previously these were bare relative paths, so the app only ran correctly when
# launched from src/agents/outreach/. Mirrors LINKEDIN_SESSION_PATH in open_deep_research.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(_PACKAGE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(_PACKAGE_DIR, "token.json")
REPORTS_DIR = os.path.join(_PACKAGE_DIR, "reports")

# All four are requested in one consent flow. Narrow this if you use a non-Google lead
# source and disable Docs saving - spreadsheets/documents/drive are then unused.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]


def get_google_credentials():
    """Load cached Google OAuth credentials, running the consent flow if needed."""
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
    return creds

##########################
# Lazy Client Utils
##########################

@lru_cache(maxsize=4)
def _build_lead_loader(loader_type: str, sheet_id: str | None, base_id: str | None, table_name: str | None):
    """Build (and cache) a lead loader for a given identity tuple.

    Cached so the two nodes that need a loader share one instance - and therefore one
    Google OAuth handshake - per process. Arguments are plain strings rather than the
    config object so the cache key stays hashable.
    """
    if loader_type == LeadLoaderType.AIRTABLE.value:
        from agents.outreach.tools.leads_loader.airtable import AirtableLeadLoader
        return AirtableLeadLoader(
            access_token=os.getenv("AIRTABLE_ACCESS_TOKEN"),
            base_id=base_id,
            table_name=table_name,
        )
    if loader_type == LeadLoaderType.HUBSPOT.value:
        from agents.outreach.tools.leads_loader.hubspot import HubSpotLeadLoader
        return HubSpotLeadLoader()

    from agents.outreach.tools.leads_loader.google_sheets import GoogleSheetLeadLoader
    return GoogleSheetLeadLoader(spreadsheet_id=sheet_id)

def get_lead_loader(config: RunnableConfig):
    """Resolve the configured lead loader.

    Instantiated lazily from config rather than injected through a constructor, so the
    compiled graph takes no arguments and can be exposed directly in langgraph.json.
    Mirrors how get_all_tools(config) resolves tools in open_deep_research.

    Args:
        config: Runtime configuration naming the lead source and its identifiers

    Returns:
        A LeadLoaderBase implementation for the configured source
    """
    configurable = SalesConfiguration.from_runnable_config(config)
    loader_type = configurable.lead_loader_type
    loader_type = loader_type.value if isinstance(loader_type, LeadLoaderType) else loader_type
    return _build_lead_loader(
        loader_type,
        configurable.sheet_id,
        configurable.airtable_base_id,
        configurable.airtable_table_name,
    )

@lru_cache(maxsize=1)
def get_docs_manager():
    """Get (and cache) the Google Docs manager.

    Deliberately lazy: constructing GoogleDocsManager triggers the Google OAuth flow, so
    building it eagerly made graph *construction* fail with FileNotFoundError whenever
    credentials.json was absent - even for runs that never touch Google Docs.
    """
    from agents.outreach.tools.google_docs_tools import GoogleDocsManager
    return GoogleDocsManager()

##########################
# Model Utils
##########################

# Sales-side counterparts of open_deep_research's configurable_model/fallback_model.
# A default model= is required here (even though every call site overrides it via
# .with_config()) because LangChain internals sometimes introspect the model before
# that override lands, and a _ConfigurableModel with no default raises a confusing
# TypeError in that path.
configurable_model = init_chat_model(
    "google_genai:gemini-3.1-flash-lite",
    configurable_fields=("model", "max_tokens", "api_key"),
    max_retries=3,
)

fallback_model = init_chat_model(
    "google_genai:gemma-4-31b-it",
    configurable_fields=("model", "max_tokens", "api_key"),
    max_retries=3,
)

def get_sales_model_config(model_name: str, config: RunnableConfig, max_tokens: int) -> dict[str, Any]:
    """Build the .with_config() dict for a sales model call."""
    return {
        "model": model_name,
        "max_tokens": max_tokens,
        "api_key": get_api_key_for_model(model_name, config),
        "tags": ["langsmith:nostream"]
    }

def get_sales_fallback_configs(configurable: SalesConfiguration, config: RunnableConfig, max_tokens: int) -> list[dict[str, Any]]:
    """Build one .with_config() dict per rung of the sales fallback ladder, in order.

    sales_fallback_model holds a comma-separated list because the free-tier request limit
    is charged per model: a retry against the exhausted model always fails, while the next
    one down is a fresh quota bucket. A single name still works and gives one fallback.
    """
    return [
        get_sales_model_config(name.strip(), config, max_tokens)
        for name in configurable.sales_fallback_model.split(",")
        if name.strip()
    ]

async def invoke_llm(
    system_prompt: str,
    user_message: str,
    model_name: str,
    config: RunnableConfig,
    response_format: Type[BaseModel] | None = None,
):
    """Invoke a sales model with retry and free-tier fallback protection.

    Every call falls back to the configured free model if the primary is rate-limited or
    unavailable - the sales side previously had no fallback at all, so a single Gemini 429
    mid-pipeline would fail the whole lead. Mirrors the model chain in open_deep_research.

    Args:
        system_prompt: System prompt for the call
        user_message: User message content
        model_name: Provider-prefixed model id (e.g. "google_genai:gemini-3.1-flash-lite")
        config: Runtime configuration for API keys and fallback model selection
        response_format: Optional pydantic model for structured output

    Returns:
        The parsed pydantic object when response_format is given, otherwise plain text
    """
    configurable = SalesConfiguration.from_runnable_config(config)
    max_tokens = configurable.sales_model_max_tokens
    primary_config = get_sales_model_config(model_name, config, max_tokens)
    fallback_configs = get_sales_fallback_configs(configurable, config, max_tokens)

    if response_format:
        model = (
            configurable_model
            .with_structured_output(response_format)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(primary_config)
            .with_fallbacks([
                fallback_model.with_config(cfg).with_structured_output(response_format)
                for cfg in fallback_configs
            ])
        )
    else:
        model = (
            configurable_model
            .with_config(primary_config)
            .with_fallbacks([fallback_model.with_config(cfg) for cfg in fallback_configs])
        )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    response = await model.ainvoke(messages)

    if response_format:
        return response
    # Gemini "thinking" mode returns content as a list of typed blocks rather than a
    # plain string; extract_answer_text pulls out just the answer.
    return extract_answer_text(response.content)

##########################
# Routing Decision Utils
##########################

def has_remaining_leads(state) -> Literal["Found leads", "No more leads"]:
    """Decide whether any leads are left to process."""
    return "Found leads" if state["number_leads"] > 0 else "No more leads"

def research_sufficiency_decision(state, max_retries: int) -> Literal["sufficient", "retry", "insufficient"]:
    """Decide what to do after a research sufficiency check.

    Fails closed: if the sufficiency verdict is missing entirely, retry rather than
    proceeding to pitch on data we never validated.
    """
    if state.get("research_sufficient"):
        return "sufficient"
    if state.get("research_retry_count", 0) < max_retries:
        return "retry"
    return "insufficient"

def qualification_decision(lead_score: str, threshold: float) -> Literal["qualified", "not qualified"]:
    """Decide whether a scored lead clears the qualification threshold."""
    try:
        score = float(lead_score)
    except (TypeError, ValueError):
        return "not qualified"
    return "qualified" if score >= threshold else "not qualified"

##########################
# Report Utils
##########################

def get_report(reports, report_name: str):
    """Retrieve the content of a report by its title.

    If a title appears more than once (e.g. a research retry appending a fresher report
    under the same title), the most recent one wins, since `reports` is append-only.
    """
    for report in reversed(reports):
        if report.title == report_name:
            return report.content
    return ""

def save_reports_locally(reports):
    """Write each report to the package-local reports directory."""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    for report in reports:
        file_path = os.path.join(REPORTS_DIR, f"{report.title}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(report.content)

##########################
# Misc Utils
##########################

def get_current_date():
    """Get the current date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")
