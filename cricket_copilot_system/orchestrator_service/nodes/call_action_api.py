from state import CricketState
from tools.action_api_client import stats_query


def call_action_api_node(state: CricketState) -> CricketState:
    try:
        result = stats_query(
            query_id=state["query_id"],
            params=state["query_params"],
            thread_id=state["thread_id"],
            user_role=state.get("user_role", "user"),
        )
        return {"action_result": result}
    except Exception as e:
        return {
            "action_result": {
                "status": "error",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "trace_id": "",
                "latency_ms": 0,
            },
            "error": f"action_api_failed: {str(e)}",
        }
