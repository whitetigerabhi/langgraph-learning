from state import CricketState
from tools.action_api_client import memory_load as memory_load_api


def memory_load_node(state: CricketState) -> CricketState:
    thread_id = state["thread_id"]

    try:
        mem = memory_load_api(thread_id=thread_id, limit=20)
        return {
            "history": mem.get("history", []),
            "memory_summary": mem.get("summary", ""),
        }
    except Exception as e:
        # Graceful degradation: system still works without memory
        return {
            "history": [],
            "memory_summary": "",
            "error": f"memory_load_failed: {str(e)}",
        }