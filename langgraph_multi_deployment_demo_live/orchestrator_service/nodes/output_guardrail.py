from state import AgentState
from content_safety import should_block


def output_guardrail_node(state: AgentState):
    """
    Output guardrail (defense-in-depth):
    Run Azure AI Content Safety on the final answer before returning it.
    Azure AI Content Safety is designed to detect harmful content in text and can be
    used on AI-generated outputs as well. [1](https://oneuptime.com/blog/post/2026-02-16-how-to-configure-content-filtering-policies-in-azure-openai-service/view)[2](https://developers.openai.com/api/reference/resources/moderations/methods/create)
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
