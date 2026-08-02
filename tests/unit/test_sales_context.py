"""Guards that the sales pitch follows the entity type.

The partner tracks were written for companies, so a college was scored on whether it
could supply cloud credits or hire builders - neither of which a college does. It is
where builders come from. Scoring it against the wrong offer rates it badly for a
reason that cannot apply.
"""
import pytest

from agents.research.entity_registry import ENTITY_REGISTRY, get_sales_context


def test_every_entity_type_has_a_pitch_and_fit():
    # Adding a type must stay one dict entry - a missing pitch would silently fall back
    # to the generic one and quietly pitch the wrong thing.
    for name, entry in ENTITY_REGISTRY.items():
        assert entry.get("pitch"), f"{name} has no pitch"
        assert entry.get("fit"), f"{name} has no fit criteria"


def test_college_is_not_pitched_company_tracks():
    pitch = get_sales_context("college").lower()
    assert "students" in pitch
    assert "cloud credits" not in pitch, "a college cannot supply cloud credits"


def test_company_keeps_the_partner_tracks():
    pitch = get_sales_context("company").lower()
    assert "cloud credits" in pitch
    assert "hiring partner" in pitch


def test_government_is_pitched_challenges_not_sponsorship():
    pitch = get_sales_context("government_dept").lower()
    assert "challenge" in pitch
    assert "procurement" in pitch


def test_person_defers_to_the_employer():
    pitch = get_sales_context("person").lower()
    assert "employer" in pitch or "organization" in pitch


@pytest.mark.parametrize("unknown", ["ngo", "", "research_institute", None])
def test_unknown_types_fall_back_instead_of_failing(unknown):
    context = get_sales_context(unknown)
    assert "What to offer them" in context
    assert "What makes one worth approaching" in context


def test_context_names_the_entity_type():
    assert "college" in get_sales_context("college")
