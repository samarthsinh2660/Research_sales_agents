"""Contact card -> outreach route conversion. No API calls, no credentials."""
from agents.research.state import ContactCard, ContactPerson, ContactPoint
from orchestrator.graph import _channel_guidance, _route_from_card


def _point(value, kind="role", url="https://x.test/contact"):
    return ContactPoint(value=value, kind=kind, source_url=url)


def test_direct_email_route_carries_the_address():
    card = ContactCard(
        people=[ContactPerson(name="Asha Rao", role="CTO", source_url="https://x.test/team")],
        emails=[_point("asha@x.test", kind="personal")],
        best_route="direct_email",
        best_route_value="asha@x.test",
    )
    route = _route_from_card(card)
    assert route.email == "asha@x.test"
    assert route.recipient_name == "Asha Rao"
    assert route.recipient_role == "CTO"


def test_role_inbox_is_treated_as_a_written_route():
    # A role inbox is a normal outcome for a large organization, not a failure: it still
    # produces a sendable address, and the draft is addressed with an attention line.
    card = ContactCard(
        people=[ContactPerson(name="Asha Rao", role="CTO", source_url="https://x.test/team")],
        emails=[_point("csr@x.test")],
        best_route="role_inbox_attn",
        best_route_value="csr@x.test",
    )
    route = _route_from_card(card)
    assert route.email == "csr@x.test"
    assert "Attn:" in _channel_guidance(route)


def test_linkedin_route_yields_a_url_and_no_email():
    # The whole point of the linkedin route: no address was published, so nothing may be
    # invented to fill the gap - the profile URL is the deliverable.
    card = ContactCard(
        people=[ContactPerson(name="Anish Shah", role="CEO", source_url="https://x.test/team")],
        linkedin_urls=[_point("https://in.linkedin.com/in/anish", kind="profile")],
        best_route="linkedin_dm",
        best_route_value="https://in.linkedin.com/in/anish",
    )
    route = _route_from_card(card)
    assert route.email == ""
    assert route.route_url == "https://in.linkedin.com/in/anish"
    assert "300 characters" in _channel_guidance(route)


def test_phone_only_card_is_not_empty():
    # Phone numbers were previously extracted and then discarded for want of a field.
    # A card carrying only a phone number is a reachable target.
    card = ContactCard(
        phones=[_point("+91 135 2770137", kind="switchboard")],
        best_route="phone",
        best_route_value="+91 135 2770137",
    )
    assert not card.is_empty()
    assert _route_from_card(card).route_url == "+91 135 2770137"


def test_unreachable_card_is_empty_and_guides_generically():
    card = ContactCard(best_route="unreachable")
    assert card.is_empty()
    assert _route_from_card(card).email == ""
    assert _channel_guidance(_route_from_card(card))


def test_every_route_type_has_channel_guidance():
    # A route with no guidance silently falls back to writing an email, which is how a
    # LinkedIn note ends up with a subject line and a signature.
    from agents.research.state import ROUTE_TYPES

    for route_type in ROUTE_TYPES:
        card = ContactCard(best_route=route_type, best_route_value="x")
        assert _channel_guidance(_route_from_card(card)), route_type
