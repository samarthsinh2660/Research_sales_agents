from pydantic import BaseModel, Field
from typing import List, Annotated
from typing_extensions import TypedDict
from operator import add
    
class Report(BaseModel):
    title: str = ""
    content: str = ""
    is_markdown: bool = False

# Define the base data needed about the lead
class LeadData(BaseModel):
    id: str = Field(..., description="The unique identifier for the lead being processed")
    name: str = Field(..., description="The full name of the lead")
    address: str = Field(..., description="The address of the lead")
    email: str = Field(..., description="The email address of the lead")
    phone: str = Field(..., description="The phone number of the lead")
    profile: str = Field(..., description="The lead profile summary from LinkedIn data")
    company_name: str = Field(default="", description="The lead's company name, from the lead sheet")
    company_website: str = Field(default="", description="The lead's company website, from the lead sheet")

class CompanyData(BaseModel):
    name: str = ""
    profile: str = ""
    website: str = ""
    
class GraphInputState(TypedDict):
    leads_ids: List[str]

class GraphState(TypedDict):
    leads_ids: List[str]
    leads_data: List[dict]
    current_lead: LeadData
    lead_score: str = ""
    company_data: CompanyData
    reports: Annotated[list[Report], add]
    research_sufficient: bool
    research_gaps: str
    research_retry_count: int
    reports_folder_link: str
    custom_outreach_report_link: str
    personalized_email: str
    interview_script: str
    number_leads: int