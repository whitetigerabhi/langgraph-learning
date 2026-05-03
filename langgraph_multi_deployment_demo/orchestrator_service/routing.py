from typing import Literal
from state import AgentState

def route_after_decide(state: AgentState) -> Literal["execute_tools", "finalize"]:
    """
    Route based on whether the model produced tool_calls.
    - If tool_calls exist -> execute tools
    - Else -> finalize the answer
    """
    return "execute_tools" if state.get("tool_calls") else "finalize"
