from typing import TypedDict, Dict, Any, List


class AgentState(TypedDict, total=False):
    query: str
    user_role: str
    hints: Dict[str, Any]

    # tool-calling conversation
    messages: List[Dict[str, Any]]

    # tool calls + results
    tool_calls: List[Dict[str, Any]]     # [{"id","name","arguments"}]
    tool_results: Dict[str, Any]         # tool_name -> dict

    # control
    step: int
    max_steps: int

    # output
    answer: str
    error: str
