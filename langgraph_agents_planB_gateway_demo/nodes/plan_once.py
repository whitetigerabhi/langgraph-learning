import os
import re
import json
from typing import Dict, Any
from openai import AzureOpenAI
from ..state import AgentState


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )


def _deployment() -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")


def _extract_json_obj(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}


def plan_once_node(state: AgentState) -> Dict[str, Any]:
    """
    Create plan ONCE per thread. If already present, no-op.
    Plan includes NON-TOOL steps:
      - ASK_CLARIFICATION
      - REQUIRE_APPROVAL
      - CALL_TOOL
      - SYNTHESIZE
    """
    if state.get("blocked"):
        return {}

    plan = state.get("plan") or {}
    if plan.get("steps"):
        return {}

    q = state.get("normalized_query", "")

    client = _client()
    dep = _deployment()

    prompt = f"""
Return ONLY JSON with schema:
{{
  "goal": "<short goal>",
  "steps": [
    {{
      "type": "ASK_CLARIFICATION",
      "key": "location|team|other",
      "question": "<question>"
    }},
    {{
      "type": "CALL_TOOL",
      "tool": "weather_api|cricket_api",
      "args": {{...}}
    }},
    {{
      "type": "REQUIRE_APPROVAL",
      "reason": "<why approval>"
    }},
    {{
      "type": "SYNTHESIZE",
      "style": "short|detailed"
    }}
  ]
}}

Rules:
- Use ASK_CLARIFICATION if required tool args missing.
- Use REQUIRE_APPROVAL only when user requests sending/external action.
- End with SYNTHESIZE.

User query: {q}
"""

    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": "Return only JSON. No markdown, no prose."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=450,
    )

    raw = resp.choices[0].message.content.strip()
    plan_obj = _extract_json_obj(raw)

    if not plan_obj.get("steps"):
        plan_obj = {"goal": "Answer the user", "steps": [{"type": "SYNTHESIZE", "style": "short"}]}

    return {"plan": plan_obj, "step_index": 0}