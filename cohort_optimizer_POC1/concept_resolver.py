"""
Simple concept resolver (stand-in for Azure AI Search).
Maps normalized user phrases to concept_ids used in the NetworkX graph.
"""

LOOKUP = {
    "diabetes": "DIABETES",
    "type 2 diabetes": "DIABETES",
    "asthma": "ASTHMA",
    "missing eye exam": "NO_EYE_EXAM",
    "corticosteroid": "CORTICO",
    "corticosteroid use": "CORTICO",
    "retinopathy": "RETINOPATHY",
}

def resolve(term: str) -> str | None:
    t = term.strip().lower()
    return LOOKUP.get(t)