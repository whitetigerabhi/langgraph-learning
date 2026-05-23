from state import CricketState


def route_after_rerank(state: CricketState):
    retrieval = state.get("retrieval", {})
    attempt = retrieval.get("attempt", 1)
    is_relevant = retrieval.get("is_relevant", False)

    if is_relevant:
        return "package"

    if attempt < 2:
        return "retry_rewrite"

    return "package"