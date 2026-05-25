"""
Actionability Knowledge Graph definition for the POC.

Nodes are universal clinical concepts with cached SQL.
Edges are directed and store precomputed Unrealized Gain (lift).

Traversal is bounded to <= 2 hops by the graph engine.
"""

NODES = {
    # Anchors
    "DIABETES": {
        "display_name": "Type 2 Diabetes",
        "clinical_category": "Diagnosis",
        "cached_sql": "SELECT member_id FROM cohort_data.members WHERE diabetes=1",
    },
    "ASTHMA": {
        "display_name": "Asthma",
        "clinical_category": "Diagnosis",
        "cached_sql": "SELECT member_id FROM cohort_data.members WHERE asthma=1",
    },

    # Hop-1 drivers
    "CORTICO": {
        "display_name": "Corticosteroid Use",
        "clinical_category": "Medication",
        "cached_sql": "SELECT member_id FROM cohort_data.members WHERE cortico=1",
    },
    "NO_EYE_EXAM": {
        "display_name": "Missing Eye Exam",
        "clinical_category": "Procedure",
        "cached_sql": "SELECT member_id FROM cohort_data.members WHERE no_eye_exam=1",
    },

    # Hop-2 tiers/outcomes
    "FILL_2": {
        "display_name": "Steroids 2+ Fills",
        "clinical_category": "Medication",
        "cached_sql": "SELECT member_id FROM cohort_data.members WHERE fill_2=1",
    },
    "FILL_4": {
        "display_name": "Steroids 4+ Fills",
        "clinical_category": "Medication",
        "cached_sql": "SELECT member_id FROM cohort_data.members WHERE fill_4=1",
    },
    "FILL_6": {
        "display_name": "Steroids 6+ Fills",
        "clinical_category": "Medication",
        "cached_sql": "SELECT member_id FROM cohort_data.members WHERE fill_6=1",
    },
    "RETINOPATHY": {
        "display_name": "History of Retinopathy",
        "clinical_category": "Diagnosis",
        "cached_sql": "SELECT member_id FROM cohort_data.members WHERE retinopathy=1",
    },
}

# Directed edges with precomputed Unrealized Gains (example values aligned to your diagram)
EDGES = [
    ("DIABETES", "CORTICO", {"lift": 1.9, "hop_level": 1}),
    ("ASTHMA", "CORTICO",   {"lift": 1.8, "hop_level": 1}),

    ("CORTICO", "FILL_2",   {"lift": 1.2, "hop_level": 2}),
    ("CORTICO", "FILL_4",   {"lift": 2.5, "hop_level": 2}),
    ("CORTICO", "FILL_6",   {"lift": 4.1, "hop_level": 2}),

    ("DIABETES", "NO_EYE_EXAM", {"lift": 1.5, "hop_level": 1}),
    ("ASTHMA",   "NO_EYE_EXAM", {"lift": 1.8, "hop_level": 1}),
    ("NO_EYE_EXAM", "RETINOPATHY", {"lift": 4.5, "hop_level": 2}),
]