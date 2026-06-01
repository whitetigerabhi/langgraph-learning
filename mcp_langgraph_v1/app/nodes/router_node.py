from app.graph.state import GraphState


def router_node(state: GraphState) -> GraphState:
    if not state.get("is_adequate", False):
        route = "clarify"
    else:
        intent = state.get("intent", "clarify")
        if intent in {"analytics", "retrieval", "mixed"}:
            route = intent
        else:
            route = "clarify"

    return {
        **state,
        "route": route,
        "trace": state.get("trace", []) + [f"router_complete:{route}"],
    }