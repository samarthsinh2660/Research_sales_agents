"""Entity Schema Registry: per-entity-type research targets and outreach pitch.

Classification alone isn't enough - a college and a company both get researched
via the same generic modules, but need different target fields (a college needs
accreditation/placement, a company needs revenue/decision-maker). This registry
maps each entity type to the fields its research should target.

It also carries the sales side, because the same mismatch appeared there: the
partner tracks are written for companies, so scoring a college on "can they supply
cloud credits or hire builders" rates it badly for a reason that does not apply -
a college is where builders come from, not a technology sponsor. `pitch` says what
is actually being offered to each kind of organization, and `fit` says what makes
one worth approaching.

Adding a new entity type (NGO, research institute, etc.) is one new dict entry
here - no graph or node changes required.
"""

ENTITY_REGISTRY: dict[str, dict] = {
    "company": {
        # Mirrors company-research-agent's proven 4-category split (company/industry/
        # financial/news) - validated in production rather than invented from scratch.
        "targets": [
            "company overview: what they do, core products/services, founding info, HQ",
            "industry: market position, competitors, industry trends",
            "financial: revenue model, funding history/rounds, pricing/business model",
            "news: recent press, leadership changes, expansion/contraction signals",
            "company's own website and blog content: recent posts, product updates, thought leadership, stated focus areas",
            "decision makers and company size",
            (
                "reachable contact routes, in priority order: a published email address "
                "(a named person's, or a role inbox such as partnerships@/info@/sales@ from "
                "the company's own contact page), then LinkedIn profile URLs for named "
                "decision makers, then the contact page URL. Report only addresses actually "
                "found in a source - never guess or construct one from a name pattern."
            ),
        ],
        "pitch": (
            "Offer one of three partner tracks:\n"
            "- Technology Partner: provide cloud credits, APIs, tools or engineering support "
            "to builders, in exchange for approved technical visibility and case studies.\n"
            "- Hiring Partner: consent-based access to builders evaluated on demonstrated work "
            "rather than resumes, for internships, contract or full-time roles.\n"
            "- Challenge/Pilot Partner: bring a real business or CSR-relevant problem and get "
            "multiple independently-built solutions, with a possible route to a pilot."
        ),
        "fit": (
            "Technical capacity to contribute tools or engineering time; active hiring or "
            "growth signals; or a real operational problem worth publishing as a challenge."
        ),
    },
    "college": {
        "targets": [
            "official website and programs offered",
            "accreditation (e.g. NAAC/NBA/UGC/AICTE listing)",
            "affiliated university",
            "placement record",
            "registrar / principal / placement officer contact",
        ],
        # A college supplies builders; it does not sponsor them. Pitching cloud credits at a
        # placement officer is the wrong offer to the wrong person.
        "pitch": (
            "Offer participation, not sponsorship:\n"
            "- Talent Pipeline: their students join as builders and get 90 days of real, "
            "evidence-backed project work, with consent-based visibility to hiring partners - "
            "a placement outcome the college can point to.\n"
            "- Campus Cohort: run a cohort on campus, giving students structured project "
            "experience that strengthens NAAC/NBA documentation and placement records.\n"
            "- Faculty Mentorship: faculty mentor teams and co-author the resulting case studies."
        ),
        "fit": (
            "Engineering, technology or management programmes; an active placement cell; "
            "student strength large enough to field teams; and a reachable placement officer, "
            "training-and-placement head, registrar or principal."
        ),
    },
    "government_dept": {
        "targets": [
            "mandate and organizational structure",
            "budget",
            "schemes and programs run",
            "key officials",
            "recent announcements or public press",
        ],
        "pitch": (
            "Offer problem-solving capacity at no procurement cost:\n"
            "- Challenge Partner: publish a real departmental problem as a challenge and get "
            "multiple independently-built, evidence-backed solutions over 90 days.\n"
            "- Pilot Route: the strongest solution can move toward a supervised pilot, letting "
            "the department evaluate an approach before committing budget."
        ),
        "fit": (
            "A concrete citizen-facing or operational problem in a domain the programme covers "
            "(tourism, mobility, health, education, disaster management, waste, governance), "
            "and a named official who can sponsor a challenge."
        ),
    },
    "person": {
        "targets": [
            "current role and affiliation",
            "professional background and work history",
            "public profile (LinkedIn, personal site, etc.)",
            "recent news, public statements, or YouTube presence (what they're currently, publicly focused on)",
        ],
        "pitch": (
            "Pitch to the organization this person leads or represents, using whichever track "
            "fits that organization - they are the route in, not the counterparty."
        ),
        "fit": (
            "Seniority to sponsor a decision, and an employer that itself fits one of the tracks. "
            "Score the employer, not the individual."
        ),
    },
    "general": {
        "targets": [
            "core facts directly relevant to the research question",
        ],
        "pitch": (
            "Describe the programme and ask which form of involvement fits them, rather than "
            "asserting a track that the research does not support."
        ),
        "fit": "Any credible route to contributing problems, tools, mentorship or builders.",
    },
}

DEFAULT_ENTITY_TYPE = "general"


def get_entity_targets(entity_type: str) -> list[str]:
    """Look up research target fields for an entity type, falling back to general."""
    entry = ENTITY_REGISTRY.get(entity_type, ENTITY_REGISTRY[DEFAULT_ENTITY_TYPE])
    return entry["targets"]


def get_sales_context(entity_type: str) -> str:
    """Build the pitch guidance for an entity type: what to offer, and what fit means.

    Injected into scoring and email generation so both judge an organization by an offer
    that actually applies to it. Without this a college is scored on whether it can supply
    cloud credits, which it never can, and rated poorly for the wrong reason.

    Args:
        entity_type: Type from classify_entity_type; unknown values fall back to general

    Returns:
        A prompt-ready block naming the entity type, what to offer it, and what fit means
    """
    entry = ENTITY_REGISTRY.get(entity_type, ENTITY_REGISTRY[DEFAULT_ENTITY_TYPE])
    fallback = ENTITY_REGISTRY[DEFAULT_ENTITY_TYPE]
    return (
        f"# This target is a: {entity_type or DEFAULT_ENTITY_TYPE}\n\n"
        f"## What to offer them\n{entry.get('pitch', fallback['pitch'])}\n\n"
        f"## What makes one worth approaching\n{entry.get('fit', fallback['fit'])}"
    )
