from pydantic import BaseModel, Field


class EmailResponse(BaseModel):
    subject: str = Field(description="An engaging subject line to encourage the lead to open the email.")
    email: str = Field(description="The personalized email content tailored to the lead’s profile and company information.")

class ResearchSufficiency(BaseModel):
    sufficient: bool = Field(description="Whether the research report has enough real, specific substance to write a credible, personalized partnership pitch.")
    gaps: str = Field(description="If not sufficient, a brief note on what's missing (e.g. 'no concrete company details, only generic industry claims'). Empty string if sufficient.")
