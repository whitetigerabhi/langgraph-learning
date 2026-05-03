import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from state import _run_tool(name: str, args: dict):from state import AgentState
    if name == "get_weather":
        return fetch_current_weather(args.get("location", ""))
    if name == "get_cricket":
        return fetch_live_cricket(args.get("team", ""))
    return {"error": f"Unknown tool {name}"}


def execute_tools_node(state: AgentState):
    calls = list(state.get("tool_calls") or [])
    if not calls:
        return {}

    # Tool-arg guardrail
    for tc in calls:
        args = json.loads(tc.get("arguments") or "{}")
        block, details = should_block(json.dumps(args))
        if block:
            return {
                "answer": "Tool input flagged as unsafe. Please try again with a safe request.",
                "error": f"blocked_tool_args:{details}",
                "tool_calls": [],
            }

    tool_results = dict(state.get("tool_results") or {})
    messages = list(state.get("messages") or [])

    with ThreadPoolExecutor(max_workers=min(4, len(calls))) as ex:
        futures = {}
        for tc in calls:
            name = tc["name"]
            args = json.loads(tc.get("arguments") or "{}")
            futures[ex.submit(_run_tool, name, args)] = tc

        for fut in as_completed(futures):
            tc = futures[fut]
            name = tc["name"]
            try:
                out = fut.result()
            except Exception as e:
                out = {"error": str(e)}

            tool_results[name] = out

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(out),
            })

    return {"tool_results": tool_results, "messages": messages, "tool_calls": []}
from tools.weather_api import fetch_current_weather
from tools.cricket_api import fetch_live_cricket
from content_safety import should_block


