from typing import Dict, Any
from ..state import AgentState


def init_node(state: AgentState) -> Dict[str, Any]:
    """
    Do NOT wipe state because we use checkpointing across runs.
    Only fill defaults when keys are missing.
    """
    updates: Dict[str, Any] = {}

    if "blocked" not in state:
        updates["blocked"] = False
    if "block_reason" not in state:
        updates["block_reason"] = ""

    if "plan" not in state:
        updates["plan"] = {}
    if "step_index" not in state:
        updates["step_index"] = 0

    if "waiting_for_user" not in state:
        updates["waiting_for_user"] = False
    if "clarify_key" not in state:
        updates["clarify_key"] = ""
    if "clarification_question" not in state:
        updates["clarification_question"] = ""
    if "clarified" not in state:
        updates["clarified"] = {}
    if "pending_step_type" not in state:
        updates["pending_step_type"] = ""

    if "needs_approval" not in state:
        updates["needs_approval"] = False
    if "approved" not in state:
        updates["approved"] = False
    if "approval_token" not in state:
        updates["approval_token"] = ""

    if "results" not in state:
        updates["results"] = {}
    if "error" not in state:
        updates["error"] = ""

    if "answer" not in state:
        updates["answer"] = ""
    if "history" not in state:
        updates["history"] = []

    return updates