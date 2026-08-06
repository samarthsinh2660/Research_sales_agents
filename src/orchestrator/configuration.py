"""Configuration management for the unified agent orchestrator."""

import os
from enum import Enum
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class OutreachIntent(Enum):
    """How deep through the pipeline a run should go."""

    RESEARCH = "research"
    QUALIFY = "qualify"
    DRAFT = "draft"
    SEND = "send"

# Intents nest: each runs the same pipeline, stopping deeper. Gates compare depth
# rather than equality so adding an intent does not mean editing every gate.
INTENT_DEPTH: dict[OutreachIntent, int] = {
    OutreachIntent.RESEARCH: 0,
    OutreachIntent.QUALIFY: 1,
    OutreachIntent.DRAFT: 2,
    OutreachIntent.SEND: 3,
}

class AgentConfiguration(BaseModel):
    """Orchestration settings: how deep to go, and how many targets to allow.

    Field names are checked against Configuration and SalesConfiguration by test,
    because from_runnable_config resolves every field from os.environ[FIELD.upper()]
    and a shared name would mean two configs fighting over one environment variable.
    """

    intent: OutreachIntent = Field(
        default=OutreachIntent.RESEARCH,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "research",
                "description": "How far to take each target. Intents nest: qualify includes research, draft includes qualify, send includes draft.",
                "options": [
                    {"label": "Research only", "value": OutreachIntent.RESEARCH.value},
                    {"label": "Research + score", "value": OutreachIntent.QUALIFY.value},
                    {"label": "Research + score + draft email", "value": OutreachIntent.DRAFT.value},
                    {"label": "Research + score + draft + send", "value": OutreachIntent.SEND.value},
                ]
            }
        }
    )
    require_send_approval: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Pause for explicit approval before each send. Selecting the send intent is authorization, but in batch one toggle would otherwise fire N real emails unattended."
            }
        }
    )
    max_targets: int = Field(
        default=25,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 25,
                "min": 1,
                "max": 200,
                "description": "Refuse runs with more targets than this, rather than discovering the cost after burning quota."
            }
        }
    )
    require_qualification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Stop at the score gate when a lead does not qualify. Turn off to draft outreach anyway - the operator may know something the research does not, and the score is advice rather than a veto."
            }
        }
    )
    max_consecutive_failures: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 5,
                "min": 1,
                "max": 100,
                "description": "Abort the run after this many targets fail back to back. A daily quota wall fails every remaining target instantly, which would otherwise finish the batch in seconds with nothing researched."
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "AgentConfiguration":
        """Create an AgentConfiguration instance from a RunnableConfig."""
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
