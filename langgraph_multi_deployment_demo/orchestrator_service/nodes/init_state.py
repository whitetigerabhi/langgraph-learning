from state import AgentState

def init_node(state: AgentState):
    return {
        "messages": state.get("messages", []),
        "tool_calls": state.get("tool_calls", []),
        "tool_results": state.get("tool_results", {}),
        "answer": state.get("answer", ""),
        "error": state.get("error", ""),
        "step": int(state.get("step", 0)),
        "max_steps": int(state.get("max_steps", 6)),
    }