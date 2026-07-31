"""State definitions for the unified agent orchestrator."""

from typing import Annotated, List, Literal

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from sales_outreach.state import Report, override_reducer

###################
# Structured Outputs
###################

class Target(BaseModel):
    """One subject to process.

    Orchestration only - deliberately carries no kind/type field. The research core
    already classifies subjects via classify_entity_type and returns entity_type;
    duplicating that here would create two sources of truth that can disagree.
    """

    name: str = Field(description="What to research, e.g. a company or person name")
    website: str | None = Field(default=None, description="Known website, used as a research seed")
    email: str | None = Field(default=None, description="Known contact email, if any")
    context: str | None = Field(
        default=None,
        description="Free-form seed facts from the source, e.g. 'CIO, Mahindra Group - speaker, ETCIO 2025'",
    )
    source: Literal["prompt", "sheet", "inline", "page"] = Field(
        description="Where this target came from. Only sheet-sourced targets get CRM write-back."
    )
    crm_row_id: str | None = Field(default=None, description="Sheet row to update; None for every other source")


class ExtractedTarget(BaseModel):
    """One organization or person pulled out of a listing page."""

    name: str = Field(description="Organization or person name, copied verbatim from the page")
    context: str = Field(
        default="",
        description=(
            "For a person: their job title and employer, e.g. 'CIO, Mahindra Group'. "
            "For an organization: its role on the page, e.g. 'Platinum Partner'. Empty if unknown."
        ),
    )

class ExtractedTargets(BaseModel):
    """Everything worth contacting that a listing page names."""

    targets: list[ExtractedTarget] = Field(
        description="Organizations and named people found on the page. Empty list if none."
    )


###################
# State Definitions
###################

def replace_reducer(current_value, new_value):
    """Replace rather than append.

    Used for `messages`: the research subgraph echoes its messages back to the parent,
    so an appending reducer accumulates duplicates on every target.
    """
    return new_value

class AgentInputState(TypedDict, total=False):
    """The prompt naming the subject, or a pre-resolved target list.

    `targets` lets a caller supply targets directly instead of having them parsed out of
    the prompt. That is what makes a long batch resumable: a rerun can pass only the
    targets that have not completed yet, keeping each one's context intact, which inline
    name parsing cannot do because job titles contain commas.
    """

    messages: Annotated[list[MessageLikeRepresentation], replace_reducer]
    targets: List[Target]

class AgentState(TypedDict):
    """Orchestrator state, carried across the per-target loop."""

    messages: Annotated[list[MessageLikeRepresentation], replace_reducer]
    targets: List[Target]
    targets_remaining: int
    current_target: Target | None
    failures: Annotated[list[str], override_reducer]
    consecutive_failures: int
    # --- per-target, reset by finish_target ---
    current_target_failed: bool
    final_report: str
    entity_type: str | None
    research_sufficient: bool
    research_gaps: str
    research_retry_count: int
    lead_score: str
    lead_qualified: bool
    reports: Annotated[list[Report], override_reducer]
    outreach_report_link: str
    send_approved: bool | None

# Every field below the marker above must be reset between targets. Enforced by test.
PER_TARGET_FIELDS = {
    "current_target_failed",
    "final_report", "entity_type", "research_sufficient", "research_gaps",
    "research_retry_count", "lead_score", "lead_qualified", "reports",
    "outreach_report_link", "send_approved",
}
