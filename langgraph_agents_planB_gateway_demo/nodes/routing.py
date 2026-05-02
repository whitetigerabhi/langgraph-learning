from typing import Literal
from ..state import AgentState


def route_loop(state: AgentState) -> Literal["execute_step", "end"]:
    """
    Loop until:
      - blocked
      - waiting_for_user
      - waiting_for_approval
      - plan finished
    """
    if state.get("blocked"):
        return "end"
    if state.get("waiting_for_user"):
        return "end"
    if state.get("needs_approval") and not state.get("approved"):
        return "end"

    steps = (state.get("plan") or {}).get("steps") or []
    idx = state.get("step_index", 0)
    return "end" if idx >= len(steps) else "execute_step"
