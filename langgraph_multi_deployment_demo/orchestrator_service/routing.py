from typing import Literal
 import AgentState

def route_after_decide(state: AgentState) -> Literal["execute_tools", "finalize"]:
    return "execute_tools" if state.get("tool_calls") else "finalize"