
from state import AgentState
from content_safety import should_block


def output_guardrail_node(state: AgentState):
    """
    Output guardrail (defense-in-depth):
    Run Azure AI Content Safety on the final answer before returning it.
    Azure AI Content Safety can be used to moderate AI-generated text. 
    """
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
