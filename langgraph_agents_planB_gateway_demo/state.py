from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    # input
    query: str
    normalized_query: str
    user_role: str  # "user" | "admin"

    # second-layer in-graph guardrail
    blocked: bool
    block_reason: str

    # Plan B1: plan once per thread
    plan: Dict[str, Any]          # {"goal": "...", "steps": [...]}
    step_index: int               # current step index into plan.steps

    # pausing mechanisms
    waiting_for_user: bool
    clarify_key: str              # e.g. "location"
    clarification_question: str
    clarified: Dict[str, str]
    pending_step_type: str        # "ASK_CLARIFICATION" | "REQUIRE_APPROVAL" | ""

    needs_approval: bool
    approved: bool
    approval_token: str

    # tool results and errors
    results: Dict[str, Any]
    error: str

    # final
    answer: str
    history: List[Dict[str, str]]