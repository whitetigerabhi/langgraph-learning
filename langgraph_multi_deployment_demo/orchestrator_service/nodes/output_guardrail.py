from state import AgentState
 import should_block

def output_guardrail_node(state: AgentState):
    ans = (state.get("answer") or "").strip()
    if not ans:
        return {}
    block, details = should_block(ans)
    if block:
        return {
            "answer": "I’m not able to share that output. Please try a different request.",
            "error": f"blocked_output:{details}",
        }
    return {}
