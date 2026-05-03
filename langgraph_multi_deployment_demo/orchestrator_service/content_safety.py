from typing import TypedDict, Dict, Any, List

class AgentState(TypedDict, total=False):
    query: str
    user_role: str
    hints: Dict[str, Any]

    messages: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]     # [{"id","name","arguments"}]
    tool_results: Dict[str, Any]         # tool_name -> JSON

    step: int
    max_steps: int

    answer: str
    error: str
