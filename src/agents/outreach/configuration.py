"""Configuration management for the sales outreach system."""

import os
from enum import Enum
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class LeadLoaderType(Enum):
    """Enumeration of available lead source providers."""

    GOOGLE_SHEETS = "google_sheets"
    AIRTABLE = "airtable"
    HUBSPOT = "hubspot"

class SalesConfiguration(BaseModel):
    """Main configuration class for the sales outreach agent.

    Field names are deliberately prefixed (sales_*) where they would otherwise
    collide with open_deep_research's Configuration: from_runnable_config resolves
    every field from os.environ[FIELD.upper()] first, so two config classes sharing
    a field name would also share - and fight over - the same environment variable.
    """

    # Lead Source Configuration
    lead_loader_type: LeadLoaderType = Field(
        default=LeadLoaderType.GOOGLE_SHEETS,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "google_sheets",
                "description": "Where to load leads from.",
                "options": [
                    {"label": "Google Sheets", "value": LeadLoaderType.GOOGLE_SHEETS.value},
                    {"label": "Airtable", "value": LeadLoaderType.AIRTABLE.value},
                    {"label": "HubSpot", "value": LeadLoaderType.HUBSPOT.value},
                ]
            }
        }
    )
    sheet_id: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Google Sheets spreadsheet ID, used when lead_loader_type is google_sheets"
            }
        }
    )
    airtable_base_id: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Airtable base ID, used when lead_loader_type is airtable"
            }
        }
    )
    airtable_table_name: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Airtable table name, used when lead_loader_type is airtable"
            }
        }
    )

    # Qualification Configuration
    lead_score_threshold: float = Field(
        default=7.0,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 7.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.5,
                "description": "Minimum lead score (out of 10) required to proceed to outreach generation"
            }
        }
    )
    max_research_retries: int = Field(
        default=1,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 1,
                "min": 0,
                "max": 3,
                "step": 1,
                "description": "How many extra gap-focused research passes to attempt when the first pass returns insufficient data"
            }
        }
    )

    # Outreach Action Configuration
    send_email_directly: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Send outreach emails immediately instead of only creating Gmail drafts. Leave off until you trust the generated content."
            }
        }
    )
    save_to_google_docs: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Also save every generated report to Google Docs. Reports are always saved locally regardless."
            }
        }
    )

    # Model Configuration
    research_sufficiency_model: str = Field(
        default="google_genai:gemini-3.5-flash-lite",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google_genai:gemini-3.5-flash-lite",
                "description": "Model that judges whether research is substantial enough to pitch on"
            }
        }
    )
    lead_scoring_model: str = Field(
        default="google_genai:gemini-3.5-flash-lite",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google_genai:gemini-3.5-flash-lite",
                "description": "Model for scoring partnership fit"
            }
        }
    )
    outreach_report_model: str = Field(
        default="google_genai:gemini-3.5-flash-lite",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google_genai:gemini-3.5-flash-lite",
                "description": "Model for writing and proofreading the custom outreach report"
            }
        }
    )
    email_model: str = Field(
        default="google_genai:gemini-3.5-flash-lite",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google_genai:gemini-3.5-flash-lite",
                "description": "Model for writing the personalized outreach email"
            }
        }
    )
    interview_script_model: str = Field(
        default="google_genai:gemini-3.5-flash-lite",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google_genai:gemini-3.5-flash-lite",
                "description": "Model for generating SPIN questions and the interview script"
            }
        }
    )
    sales_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for sales outreach model calls"
            }
        }
    )
    sales_fallback_model: str = Field(
        default=(
            "google_genai:gemini-3.1-flash-lite,"
            "google_genai:gemini-2.5-flash-lite,"
            "google_genai:gemini-3.6-flash,"
            "google_genai:gemini-2.0-flash,"
            "google_genai:gemma-4-31b-it,"
            "google_genai:gemma-4-26b-a4b-it"
        ),
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google_genai:gemini-3.1-flash-lite,google_genai:gemini-2.5-flash-lite,google_genai:gemini-3.6-flash,google_genai:gemini-2.0-flash,google_genai:gemma-4-31b-it,google_genai:gemma-4-26b-a4b-it",
                "description": "Comma-separated backup models, tried in order when a configured sales model is rate-limited or unavailable. A list rather than one name because the free-tier request limit is charged per model: retrying an exhausted model always fails, while the next one down is a fresh quota bucket. Set a single name to go back to one fallback."
            }
        }
    )
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "SalesConfiguration":
        """Create a SalesConfiguration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
