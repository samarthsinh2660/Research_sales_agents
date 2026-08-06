"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class EntityClassification(BaseModel):
    """Classification of what kind of subject a research request is about."""

    entity_type: str = Field(
        description="The type of entity being researched: 'company', 'college', 'government_dept', 'person', or 'general' if none of those clearly apply.",
    )
    reasoning: str = Field(
        description="Brief reasoning for why this entity type was chosen.",
    )

class Subtopics(BaseModel):
    """A small set of distinct research angles to ensure comprehensive coverage of a research brief."""

    subtopics: list[str] = Field(
        description="3-5 distinct, non-overlapping research angles covering the research brief. Fewer (as low as 1) if the brief is too narrow to meaningfully split.",
    )


###################
# Contact Finding
###################

# Every reachable route we know how to write outreach for. `unreachable` is a real
# answer, but an earned one - it is only accepted once enough sources were actually tried.
ROUTE_TYPES = (
    "direct_email",
    "role_inbox_attn",
    "linkedin_dm",
    "phone",
    "contact_form",
    "postal",
    "unreachable",
)

class ContactPoint(BaseModel):
    """One contact detail, with the page it was read from.

    source_url is required rather than optional: a value with no page behind it is
    indistinguishable from one assembled from a name pattern, and that is the single
    failure mode contact finding exists to avoid.
    """

    value: str = Field(description="The address, number or URL exactly as published.")
    kind: str = Field(
        description=(
            "For an email: 'personal', 'role' (info@/csr@) or 'department'. "
            "For a phone: 'direct', 'mobile', 'switchboard' or 'department'."
        )
    )
    source_url: str = Field(description="The page this was read from. Never empty.")
    belongs_to: str = Field(
        default="", description="Named person this belongs to, if tied to one. Empty for org-wide details."
    )

class ContactPerson(BaseModel):
    """A named individual at the target organization."""

    name: str = Field(description="Full name, verbatim from a source that names them in this role.")
    role: str = Field(default="", description="Job title, e.g. 'Chief Information Officer'.")
    linkedin_url: str = Field(default="", description="Profile URL, if one was found.")
    source_url: str = Field(description="Page that names this person in this role.")

class ContactCard(BaseModel):
    """Every contact detail found for one target, plus the recommended way in.

    Structured rather than prose on purpose. Contacts used to reach the caller inside the
    research report's text, where compress_research and the report writer dropped them:
    phone numbers were extracted by the crawler and then discarded because the report
    template had no field for them. A card cannot lose a detail for lack of a slot.
    """

    organization: str = Field(default="", description="Organization these contacts belong to.")
    people: list[ContactPerson] = Field(default_factory=list, description="Named people found.")
    emails: list[ContactPoint] = Field(default_factory=list, description="Every email address found.")
    phones: list[ContactPoint] = Field(default_factory=list, description="Every phone number found.")
    linkedin_urls: list[ContactPoint] = Field(
        default_factory=list, description="LinkedIn profile or company page URLs found."
    )
    contact_page: str = Field(default="", description="URL of a contact form or contact page, if any.")
    postal_address: str = Field(default="", description="Published postal address, if any.")
    best_route: str = Field(
        description=f"The single most actionable way in. One of: {', '.join(ROUTE_TYPES)}."
    )
    best_route_value: str = Field(
        default="", description="The address, number or URL for best_route. Empty only when unreachable."
    )
    best_route_reason: str = Field(default="", description="One sentence on why this route over the others.")
    sources_checked: list[str] = Field(
        default_factory=list,
        description="Which sources were actually tried, e.g. 'official website crawl', 'AICTE disclosure'.",
    )

    def is_empty(self) -> bool:
        """Whether nothing reachable was found at all."""
        return not (self.emails or self.phones or self.linkedin_urls or self.contact_page)

class ContactPlan(BaseModel):
    """How to hunt for one particular target.

    Effort is discretionary; existence is not. The plan chooses which sources are worth
    trying and in what order, but is never asked whether to look at all - contact research
    used to be an instruction in the supervisor prompt and was silently skipped.
    """

    priority_sources: list[str] = Field(
        description="Sources to try, best first, chosen for this entity type. 2-5 entries."
    )
    queries: list[str] = Field(
        description="Specific search queries likely to surface contacts for this target. 2-5 entries."
    )
    reasoning: str = Field(default="", description="One or two sentences on why this order.")


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    entity_type: Optional[str]
    entity_guidance: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ContactAgentInputState(TypedDict, total=False):
    """What a caller hands the contact agent: who to find, and any seed we already have."""

    target_name: str
    target_website: str
    target_context: str
    entity_type: str

class ContactAgentState(TypedDict, total=False):
    """State for the contact-finding subgraph."""

    target_name: str
    target_website: str
    target_context: str
    entity_type: str
    contact_plan: ContactPlan | None
    contact_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    contact_tool_calls: int
    contact_card: ContactCard | None