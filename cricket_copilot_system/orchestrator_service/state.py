from typing import TypedDict, List, Dict, Any, Optional


class CricketState(TypedDict, total=False):
    # Request identity
    thread_id: str
    user_role: str
    message: str

    # Memory
    history: List[Dict[str, str]]
    memory_summary: str
    memory_saved: bool

    # Routing / planning
    route: str                  # "analytics" | "trivia"
    route_reason: str

    # Analytics path
    query_id: str
    query_params: Dict[str, Any]
    action_result: Dict[str, Any]

    # Retrieval path
    retrieval: Dict[str, Any]
    # recommended structure:
    # {
    #   "query_rewritten": str,
    #   "candidates": int,
    #   "matches": [ ... ],
    #   "citations": [ ... ],
    #   "confidence": float,
    #   "is_relevant": bool
    # }

    # Final answer
    answer: str

    # Diagnostics
    error: Optional[str]