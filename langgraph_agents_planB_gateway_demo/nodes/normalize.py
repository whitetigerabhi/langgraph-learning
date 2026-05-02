from ..state import AgentState


def normalize_node(state: AgentState):
    q = (state.get("query") or "").strip()
    normalized = " ".join(q.split())
    return {"normalized_query": normalized}
