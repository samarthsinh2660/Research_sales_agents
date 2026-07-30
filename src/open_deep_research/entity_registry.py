"""Entity Schema Registry: per-entity-type research targets.

Classification alone isn't enough - a college and a company both get researched
via the same generic modules, but need different target fields (a college needs
accreditation/placement, a company needs revenue/decision-maker). This registry
maps each entity type to the fields its research should target.

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
            "decision makers and company size",
        ],
    },
    "college": {
        "targets": [
            "official website and programs offered",
            "accreditation (e.g. NAAC/NBA/UGC/AICTE listing)",
            "affiliated university",
            "placement record",
            "registrar / principal / placement officer contact",
        ],
    },
    "government_dept": {
        "targets": [
            "mandate and organizational structure",
            "budget",
            "schemes and programs run",
            "key officials",
            "recent announcements or public press",
        ],
    },
    "person": {
        "targets": [
            "current role and affiliation",
            "professional background",
            "public profile (LinkedIn, personal site, etc.)",
            "recent news or public statements",
        ],
    },
    "general": {
        "targets": [
            "core facts directly relevant to the research question",
        ],
    },
}

DEFAULT_ENTITY_TYPE = "general"


def get_entity_targets(entity_type: str) -> list[str]:
    """Look up research target fields for an entity type, falling back to general."""
    entry = ENTITY_REGISTRY.get(entity_type, ENTITY_REGISTRY[DEFAULT_ENTITY_TYPE])
    return entry["targets"]
