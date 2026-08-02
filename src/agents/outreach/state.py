"""Graph state definitions and data structures for the sales outreach agent."""

import operator
from typing import Annotated, List

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

###################
# Structured Outputs
###################

class Report(BaseModel):
    """A generated document produced somewhere in the outreach pipeline."""

    title: str = ""
    content: str = ""
    is_markdown: bool = False

class LeadData(BaseModel):
    """A single lead loaded from the configured lead source."""

    id: str = Field(..., description="The unique identifier for the lead being processed")
    name: str = Field(..., description="The full name of the lead")
    address: str = Field(..., description="The address of the lead")
    email: str = Field(..., description="The email address of the lead")
    phone: str = Field(..., description="The phone number of the lead")
    profile: str = Field(..., description="The lead profile summary from research")
    company_name: str = Field(default="", description="The lead's company name, from the lead sheet")
    company_website: str = Field(default="", description="The lead's company website, from the lead sheet")

class CompanyData(BaseModel):
    """Resolved details about the lead's company."""

    name: str = ""
    profile: str = ""
    website: str = ""

class EmailResponse(BaseModel):
    """Structured output for the personalized outreach email."""

    subject: str = Field(description="An engaging subject line to encourage the lead to open the email.")
    email: str = Field(description="The personalized email content tailored to the lead's profile and company information.")

class LeadScore(BaseModel):
    """Scoring verdict with the reasoning that makes it actionable.

    A bare number says a lead is worth pursuing but not why or how. The track and angle
    are what the outreach email is actually built from, so they are produced by the same
    call that assigns the score rather than re-derived later from the report.
    """

    score: float = Field(description="Overall partnership fit, 0-10.")
    track: str = Field(
        description="Best-fit partner track: 'Technology', 'Hiring', 'Challenge', or 'None' if unsuitable."
    )
    reasoning: str = Field(
        description="Two or three sentences on why this score and track, citing specifics from the research."
    )
    angle: str = Field(
        description="The single most compelling hook to open outreach with, drawn from the research. Empty if none is strong enough."
    )


class ContactRoute(BaseModel):
    """Who to address and how to reach them.

    Every field is copied from the research report. Nothing here may be constructed from
    a name pattern: an invented address silently fails or reaches a stranger.
    """

    recipient_name: str = Field(
        default="", description="Named person to address, verbatim from the research. Empty if none was found."
    )
    recipient_role: str = Field(
        default="", description="That person's job title, e.g. 'Chief Information Officer'. Empty if unknown."
    )
    email: str = Field(
        default="", description="Email address that literally appeared in a source. Empty if none was published."
    )
    route_type: str = Field(
        description="Which route this is: 'personal_email', 'role_inbox', 'linkedin', 'contact_form', or 'none'."
    )
    route_url: str = Field(
        default="", description="LinkedIn profile or contact-page URL, if that is the best route."
    )


class EmailGrounding(BaseModel):
    """Whether an outreach email's factual claims are supported by the research."""

    grounded: bool = Field(
        description="True only if every factual claim about the recipient or their organization is supported by the research."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Each claim in the email that the research does not support. Empty when grounded.",
    )


class ResearchSufficiency(BaseModel):
    """Verdict on whether research is substantial enough to write a credible pitch."""

    sufficient: bool = Field(description="Whether the research report has enough real, specific substance to write a credible, personalized partnership pitch.")
    gaps: str = Field(description="If not sufficient, a brief note on what's missing (e.g. 'no concrete company details, only generic industry claims'). Empty string if sufficient.")


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

class GraphInputState(TypedDict):
    """InputState is only the optional list of lead ids to process."""

    leads_ids: List[str]

class GraphState(TypedDict):
    """Main state for the outreach automation, carried across the per-lead loop."""

    leads_ids: List[str]
    leads_data: List[LeadData]
    current_lead: LeadData | None
    number_leads: int
    lead_score: str
    lead_qualified: bool
    company_data: CompanyData
    # override_reducer (not plain add) so update_CRM can actually clear reports between
    # leads - with a plain `add` reducer, lead 1's reports silently accumulated into
    # lead 2's prompts and Google Docs.
    reports: Annotated[list[Report], override_reducer]
    research_sufficient: bool
    research_gaps: str
    research_retry_count: int
    # Lives in state rather than on a node-class instance: it is written by one node and
    # read by two later ones, which makes it graph state, not object state.
    drive_folder_name: str
    reports_folder_link: str
    custom_outreach_report_link: str
