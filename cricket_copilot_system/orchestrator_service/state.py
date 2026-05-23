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
    route: str
    route_reason: str
    query_id: str
    query_params: Dict[str, Any]

    # Action API output
    action_result: Dict[str, Any]

    # Final answer
    answer: str

    # Diagnostics
    error: Optional[str]