from app.graph.state import GraphState


def route_after_router(state: GraphState) -> str:
    route = state.get("route", "clarify")
    if route == "analytics":
        return "analytics"
    if route == "retrieval":
        return "retrieval"
    if route == "mixed":
        return "mixed"
    return "clarify"


def route_after_analytics(state: GraphState) -> str:
    if state.get("route") == "mixed":
        return "to_retrieval"
    return "to_synthesis"


def route_after_retrieval(state: GraphState) -> str:
    if state.get("route") == "mixed" and state.get("analytics_result"):
        return "to_fusion"
    return "to_synthesis"