from state import CricketState
from tools.action_api_client import memory_save as memory_save_api


def memory_save_node(state: CricketState) -> CricketState:
    thread_id = state["thread_id"]
    message = state["message"]
    answer = state.get("answer", "")

    try:
        # Save user turn
        memory_save_api(
            thread_id=thread_id,
            role="user",
            content=message,
            metadata={"route": state.get("route"), "query_id": state.get("query_id")},
        )

        # Save assistant turn
        memory_save_api(
            thread_id=thread_id,
            role="assistant",
            content=answer,
            metadata={"trace": state.get("action_result", {}).get("trace_id")},
        )

        return {}
    except Exception as e:
        return {"error": f"memory_save_failed: {str(e)}"}