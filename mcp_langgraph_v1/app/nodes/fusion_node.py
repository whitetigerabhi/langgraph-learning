from app.graph.state import GraphState


def fusion_node(state: GraphState) -> GraphState:
    return {
        **state,
        "trace": state.get("trace", []) + ["fusion_complete"],
    }