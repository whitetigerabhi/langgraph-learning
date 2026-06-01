from app.graph.state import GraphState


def intake_node(state: GraphState) -> GraphState:
    user_query = state["user_query"].strip()
    normalized_query = " ".join(user_query.split())

    return {
        **state,
        "normalized_query": normalized_query,
        "trace": state.get("trace", []) + ["intake_complete"],
    }