import os
import re
from typing import Dict, Any
from openai import AzureOpenAI

from ..state import AgentState
from ..tools.weather_api import fetch_current_weather
from ..tools.cricket_api import fetch_live_cricket


UK_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )


def _deployment() -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")


def _resolve_placeholders(args: Dict[str, Any], clarified: Dict[str, str]) -> Dict[str, Any]:
    """
    Supports placeholders like "$location" or "$team".
    """
    out: Dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and v.startswith("$"):
            key = v[1:].lower()
            out[k] = clarified.get(key, v)
        else:
            out[k] = v
    return out


def execute_step_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute one plan step. This is deterministic.
    It either:
      - pauses (waiting_for_user / needs_approval)
      - calls a tool
      - synthesizes
    """
    if state.get("blocked"):
        return {"answer": state.get("block_reason", "BLOCKED. Please try again.")}

    if state.get("waiting_for_user"):
        return {"answer": state.get("clarification_question", "Please provide more details.")}

    if state.get("needs_approval") and not state.get("approved"):
        token = state.get("approval_token", "")
        return {"answer": f"Approval required. approval_token={token}"}

    plan = state.get("plan") or {}
    steps = plan.get("steps") or []
    idx = state.get("step_index", 0)

    if idx >= len(steps):
        return {}

    step = steps[idx]
    step_type = (step.get("type") or "").strip()

    # 1) Ask clarification (pause)
    if step_type == "ASK_CLARIFICATION":
        key = (step.get("key") or "other").strip()
        qn = (step.get("question") or "Could you clarify?").strip()
        return {
            "waiting_for_user": True,
            "clarify_key": key,
            "clarification_question": qn,
            "pending_step_type": "ASK_CLARIFICATION",
            "answer": qn
        }

    # 2) Require approval (pause)
    if step_type == "REQUIRE_APPROVAL":
        token = state.get("approval_token") or os.urandom(8).hex()
        return {
            "needs_approval": True,
            "approved": False,
            "approval_token": token,
            "pending_step_type": "REQUIRE_APPROVAL",
            "answer": f"Approval required. approval_token={token}"
        }

    # 3) Tool call
    if step_type == "CALL_TOOL":
        tool = step.get("tool")
        args = _resolve_placeholders(step.get("args") or {}, state.get("clarified") or {})
        results = dict(state.get("results") or {})

        try:
            if tool == "weather_api":
                location = (args.get("location") or "").strip()
                if not location:
                    m = UK_POSTCODE_RE.search(state.get("normalized_query", ""))
                    location = m.group(0).upper().strip() if m else state.get("normalized_query", "")
                results["weather"] = fetch_current_weather(location)

            elif tool == "cricket_api":
                team = (args.get("team") or args.get("query") or "").strip()
                if not team:
                    team = "CSK" if "csk" in (state.get("normalized_query") or "").lower() else ""
                results["cricket"] = fetch_live_cricket(team)

            else:
                return {"error": f"Unknown tool in plan: {tool}", "step_index": idx + 1, "results": results}

        except Exception as e:
            return {"error": f"{tool}_error: {e}", "step_index": idx + 1, "results": results}

        return {"results": results, "step_index": idx + 1}

    # 4) Synthesize final answer
    if step_type == "SYNTHESIZE":
        style = (step.get("style") or "short").strip()
        q = state.get("normalized_query", "")
        results = state.get("results", {})
        err = state.get("error", "")

        client = _client()
        dep = _deployment()
        resp = client.chat.completions.create(
            model=dep,
            messages=[
                {"role": "system", "content": f"Write a {style} answer. Use only provided results; do not hallucinate."},
                {"role": "user", "content": f"User query: {q}\nResults: {results}\nError: {err}\n"},
            ],
            max_tokens=300,
        )
        ans = resp.choices[0].message.content.strip()

        history = list(state.get("history") or [])
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})

        return {"answer": ans, "history": history, "step_index": idx + 1}

    # unknown step
    return {"error": f"Unknown step type: {step_type}", "step_index": idx + 1}