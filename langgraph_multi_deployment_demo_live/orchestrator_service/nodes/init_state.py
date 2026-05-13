from state import AgentState


def init_node(state: AgentState):
    """
    Initialize default keys in state WITHOUT wiping existing values.
    This is important because LangGraph + SQLite checkpointer persists state across runs.
    """
    return {
        "messages": state.get("messages", []),
        "tool_calls": state.get("tool_calls", []),
        "tool_results": state.get("tool_results", {}),
        "answer": state.get("answer", ""),
        "error": state.get("error", ""),
        "step": int(state.get("step", 0)),
        "max_steps": int(state.get("max_steps", 6)),
    }
