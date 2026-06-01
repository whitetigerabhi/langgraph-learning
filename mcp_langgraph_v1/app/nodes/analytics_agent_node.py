from app.graph.state import GraphState
from app.agents.analytics_agent import AnalyticsAgent
from app.mcp_clients.analytics_client import AnalyticsMCPClient


analytics_agent = AnalyticsAgent()
analytics_client = AnalyticsMCPClient()


async def analytics_agent_node(state: GraphState) -> GraphState:
    plan = analytics_agent.plan(
        user_query=state["user_query"],
        entities=state.get("entities", {}),
        sub_intent=state.get("sub_intent"),
    )

    tool_call = {
        "server": "analytics_mcp_server",
        "tool_name": "run_stats_query",
        "arguments": {
            "query_id": plan["query_id"],
            "params": plan["params"],
        },
        "status": "pending",
    }

    try:
        result = await analytics_client.run_stats_query(
            query_id=plan["query_id"],
            params=plan["params"],
        )

        tool_call["status"] = "success"
        tool_call["raw_result"] = result

        evidence_item = {
            "source_type": "analytics",
            "source_id": plan["query_id"],
            "content": result,
        }

        return {
            **state,
            "planned_tool_calls": state.get("planned_tool_calls", []) + [tool_call],
            "executed_tool_calls": state.get("executed_tool_calls", []) + [tool_call],
            "analytics_result": result,
            "evidence": state.get("evidence", []) + [evidence_item],
            "trace": state.get("trace", []) + ["analytics_complete"],
        }

    except Exception as e:
        tool_call["status"] = "failed"
        tool_call["error"] = str(e)

        return {
            **state,
            "planned_tool_calls": state.get("planned_tool_calls", []) + [tool_call],
            "executed_tool_calls": state.get("executed_tool_calls", []) + [tool_call],
            "errors": state.get("errors", []) + [f"analytics_error:{str(e)}"],
            "trace": state.get("trace", []) + ["analytics_failed"],
        }