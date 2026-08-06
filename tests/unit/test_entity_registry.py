"""Unit tests for entity_registry.py - pure lookups, no API calls."""
from agents.research.entity_registry import (
    DEFAULT_ENTITY_TYPE,
    ENTITY_REGISTRY,
    get_entity_targets,
)


def test_known_entity_types_have_targets():
    for entity_type in ("company", "college", "government_dept", "person", "general"):
        targets = get_entity_targets(entity_type)
        assert isinstance(targets, list)
        assert len(targets) > 0


def test_unknown_entity_type_falls_back_to_default():
    assert get_entity_targets("nonexistent_type") == get_entity_targets(DEFAULT_ENTITY_TYPE)


def test_default_entity_type_is_registered():
    assert DEFAULT_ENTITY_TYPE in ENTITY_REGISTRY


def test_company_targets_are_specific_not_generic():
    # Regression guard: company targets were deliberately aligned to
    # company-research-agent's proven category split, not left as vague bullets
    targets_text = " ".join(get_entity_targets("company")).lower()
    assert "industry" in targets_text
    assert "financial" in targets_text or "revenue" in targets_text
