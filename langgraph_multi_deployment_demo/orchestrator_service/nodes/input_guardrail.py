from state import AgentState
from content_safety import should_block

def input_guardrail_node(state: AgentState):
    if state.get("answer"):
        return {}
    q = (state.get("query") or "").strip()
    block, details = should_block(q)
    if block:
        return {
            "answer": "I can’t help with that. Please try again with a safe request.",
            "error": f"blocked_input:{details}",
        }
    return {}