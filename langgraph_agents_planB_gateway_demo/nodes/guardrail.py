import re
from typing import Literal
from ..state import AgentState


DESTRUCTIVE_PATTERNS = [
    r"\bdrop\s+table\b",
    r"\btruncate\s+table\b",
    r"\bdelete\s+table\b",
    r"\bdestroy\s+repo\b",
    r"\brm\s+-rf\b",
    r"\bwipe\b",
]

UNSAFE_PATTERNS = [
    r"\bmake\s+a\s+bomb\b",
    r"\bbuild\s+a\s+bomb\b",
]


def guardrail_node(state: AgentState):
    q = (state.get("normalized_query") or "").lower()
    role = (state.get("user_role") or "user").lower()

    if any(re.search(p, q) for p in UNSAFE_PATTERNS):
        return {
            "blocked": True,
            "block_reason": "BLOCK: I can’t help with that. Please try again with a safe request."
        }

    if any(re.search(p, q) for p in DESTRUCTIVE_PATTERNS) and role != "admin":
        return {
            "blocked": True,
            "block_reason": "BLOCK: Destructive actions are restricted to admins. Please try again."
        }

    return {"blocked": False, "block_reason": ""}


def route_after_guardrail(state: AgentState) -> Literal["plan_once", "end"]:
    return "end" if state.get("blocked") else "plan_once"