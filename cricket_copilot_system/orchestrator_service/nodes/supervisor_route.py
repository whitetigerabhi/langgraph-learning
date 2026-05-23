from state import CricketState


def supervisor_route_node(state: CricketState) -> CricketState:
    message = (state.get("message") or "").lower()

    analytics_keywords = [
        "top", "highest", "lowest", "average", "strike rate",
        "economy", "most runs", "most wickets", "head-to-head",
        "by season", "min balls", "run rate"
    ]

    trivia_keywords = [
        "what is", "explain", "define", "meaning",
        "who is", "tell me about", "trivia", "rule", "rules"
    ]

    if any(k in message for k in analytics_keywords):
        return {
            "route": "analytics",
            "route_reason": "deterministic_analytics_keyword_match",
        }

    if any(k in message for k in trivia_keywords):
        return {
            "route": "trivia",
            "route_reason": "deterministic_trivia_keyword_match",
        }

    # First fallback: default to trivia for explanatory queries
    return {
        "route": "trivia",
        "route_reason": "default_to_trivia_for_step5",
    }