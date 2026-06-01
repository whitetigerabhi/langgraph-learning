from app.graph.state import GraphState


def adequacy_check_node(state: GraphState) -> GraphState:
    required_fields = state.get("required_fields", [])
    entities = state.get("entities", {})

    missing_fields = [
        f for f in required_fields
        if f not in entities or entities.get(f) in [None, ""]
    ]

    return {
        **state,
        "missing_fields": missing_fields,
        "is_adequate": len(missing_fields) == 0,
        "trace": state.get("trace", []) + ["adequacy_check_complete"],
    }