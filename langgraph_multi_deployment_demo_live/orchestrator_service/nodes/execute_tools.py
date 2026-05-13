import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from state import AgentState
from tools.weather_api import fetch_current_weather
from tools.cricket_api import fetch_live_cricket
from rag_retriever import retrieve_docs
from content_safety import should_block


def _run_tool(name: str, args: dict):
    if name == "get_weather":
        return fetch_current_weather(args.get("location", ""))
    if name == "get_cricket":
        return fetch_live_cricket(args.get("team", ""))
    if name == "retrieve_docs":
        return retrieve_docs(args.get("query", ""), int(args.get("top_k", 4)))
    return {"error": f"Unknown tool {name}"}


def execute_tools_node(state: AgentState):
    calls = list(state.get("tool_calls") or [])
    if not calls:
        return {}

    # Tool-arg guardrail (defense-in-depth)
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
    rag_meta = dict(state.get("rag_meta") or {})

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

            # If this was retrieval, store quality metrics in state for later routing/decisions
            if name == "retrieve_docs" and isinstance(out, dict):
                rag_meta = {
                    "is_relevant": out.get("is_relevant", False),
                    "max_score": out.get("max_score", 0.0),
                    "threshold_used": out.get("threshold_used", None),
                    "top_k": out.get("top_k", None),
                    "docs": list({m.get("doc") for m in out.get("matches", []) if isinstance(m, dict)}),
                }

            # Append tool output to conversation for the model
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(out),
            })

    return {
        "tool_results": tool_results,
        "messages": messages,
        "tool_calls": [],
        "rag_meta": rag_meta,
    }
