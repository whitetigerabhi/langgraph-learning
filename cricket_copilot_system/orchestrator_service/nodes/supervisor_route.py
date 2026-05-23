from state import CricketState


def supervisor_route_node(state: CricketState) -> CricketState:
    """
    First vertical slice: route only analytics-style queries.
    Later we will expand this to trivia / mixed / clarify.
    """
    message = (state.get("message") or "").lower()

    analytics_keywords = [
        "top", "highest", "lowest", "average", "strike rate",
        "economy", "most runs", "most wickets", "head-to-head",
        "by season", "min balls", "run rate"
    ]

    if any(k in message for k in analytics_keywords):
        return {
            "route": "analytics",
            "route_reason": "deterministic_analytics_keyword_match",
        }

    # For now everything goes to analytics path as default in this slice.
    return {
        "route": "analytics",
        "route_reason": "default_to_analytics_for_first_slice",
    }