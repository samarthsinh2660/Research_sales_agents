"""Unit tests for SalesConfiguration - no API calls."""
from agents.outreach.configuration import LeadLoaderType, SalesConfiguration


def test_defaults_hold_for_empty_config():
    config = SalesConfiguration.from_runnable_config({})
    assert config.lead_score_threshold == 7.0
    assert config.max_research_retries == 1
    # Both action flags must default off - a bad generation should never auto-send
    # to a real prospect, and Docs saving requires OAuth that may not be set up.
    assert config.send_email_directly is False
    assert config.save_to_google_docs is False


def test_runtime_config_overrides_defaults():
    config = SalesConfiguration.from_runnable_config(
        {"configurable": {"lead_loader_type": "airtable", "lead_score_threshold": 8.5}}
    )
    assert config.lead_loader_type == LeadLoaderType.AIRTABLE
    assert config.lead_score_threshold == 8.5


def test_fallback_model_is_configured():
    # The sales side previously had no fallback at all, so one Gemini 429 killed a run.
    config = SalesConfiguration.from_runnable_config({})
    assert config.sales_fallback_model


def test_sales_fields_do_not_collide_with_research_config():
    # from_runnable_config resolves every field from os.environ[FIELD.upper()], so any
    # field name shared with open_deep_research's Configuration would fight over the
    # same env var. Guard the ones most likely to be reintroduced by accident.
    from agents.research.configuration import Configuration

    shared = set(SalesConfiguration.model_fields) & set(Configuration.model_fields)
    assert shared <= {"max_structured_output_retries"}, f"unexpected env-var collision: {shared}"
