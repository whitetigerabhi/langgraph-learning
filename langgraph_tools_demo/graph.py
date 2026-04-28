import os
import re
import json
import sqlite3
from typing import TypedDict, Literal, List, Dict, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import AzureOpenAI

from tools.weather_api import fetch_current_weather
from tools.cricket_api import fetch_live_cricket


# ---------
# State
# ---------
class ToolState(TypedDict, total=False):
    query: str
    normalized_query: str

    # routing + extracted params
    intent: Literal["weather", "cricket", "general"]
    location: str
    cricket_query: str

    # tool outputs
    weather: dict
    cricket: dict
    error: str

    # final response
    answer: str

    # conversation memory (persisted via checkpointer)
    history: List[Dict[str, str]]


# ---------
# Helpers
# ---------
UK_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)

def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())

def _azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )

def _deployment() -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")

def _extract_json_obj(raw: str) -> dict:
    """Best-effort JSON extraction that never throws."""
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

def _extract_weather_location(text: str) -> Optional[str]:
    """
    Extract a location for weather:
    - If UK postcode exists, return the postcode (best).
    - Else try 'weather in <place>' or 'weather for <place>'.
    - Else None.
    """
    m = UK_POSTCODE_RE.search(text or "")
    if m:
        # normalize postcode spacing/case
        return m.group(0).upper().strip()

    m2 = re.search(r"\b(?:weather\s+in|weather\s+for|in|for)\s+(.+)$", text or "", re.IGNORECASE)
    if m2:
        candidate = m2.group(1).strip()
        # strip trailing punctuation
        candidate = re.sub(r"[?.!,;:]+$", "", candidate).strip()
        return candidate if candidate else None

    return None

def _extract_cricket_keyword(text: str) -> str:
    """
    Simple extractor: if common team keyword is present, use it.
    Otherwise return original query (API can return top live matches).
    """
    t = (text or "").lower()
    # Expandable list
    for kw in ["csk", "mi", "rcb", "kkr", "dc", "rr", "gt", "lsg", "srh", "pbks", "ipl"]:
        if kw in t:
            return kw.upper()
    return text or ""


# ---------
# Nodes
# ---------
def init_node(state: ToolState) -> ToolState:
    return {
        "history": state.get("history", []),
        "error": "",
        "answer": "",
    }

def normalize_node(state: ToolState) -> ToolState:
    return {"normalized_query": _normalize(state.get("query", ""))}

def plan_node(state: ToolState) -> ToolState:
    """
    Planner/classifier node (still NOT an agent):
    - determines intent: weather/cricket/general
    - extracts parameters needed by tool nodes
    - robust to LLM output formatting errors
    """
    q = state.get("normalized_query", "")
    qlow = q.lower()

    # ---------
    # Deterministic fast-paths (cheap, reliable)
    # ---------
    loc = _extract_weather_location(q)
    if "weather" in qlow or loc:
        return {"intent": "weather", "location": loc or q}

    if "cricket" in qlow or "score" in qlow or "live" in qlow or "csk" in qlow:
        return {"intent": "cricket", "cricket_query": _extract_cricket_keyword(q)}

    # ---------
    # LLM classifier fallback
    # ---------
    client = _azure_client()
    dep = _deployment()

    prompt = f"""
Classify the query into one intent: weather | cricket | general.
Return ONLY JSON:
{{"intent":"weather|cricket|general","location":"", "cricket_query":""}}

Rules:
- If weather: location should be the city/postcode only (not the whole sentence)
- If cricket: cricket_query should be a team/keyword if present, else empty

Query: {q}
"""

    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": "Output only JSON. No markdown, no prose."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=160,
    )
    raw = resp.choices[0].message.content.strip()
    data = _extract_json_obj(raw)

    intent = data.get("intent", "general")
    location = (data.get("location") or "").strip()
    cricket_query = (data.get("cricket_query") or "").strip()

    # Final safety: if LLM says weather but location is empty, attempt deterministic extraction
    if intent == "weather" and not location:
        location = _extract_weather_location(q) or q

    # Final safety: if cricket intent but empty query, attempt keyword extraction
    if intent == "cricket" and not cricket_query:
        cricket_query = _extract_cricket_keyword(q)

    if intent not in ("weather", "cricket", "general"):
        intent = "general"

    return {
        "intent": intent,
        "location": location,
        "cricket_query": cricket_query,
    }

def route_intent(state: ToolState) -> Literal["weather_api", "cricket_api", "general_llm"]:
    intent = state.get("intent", "general")
    if intent == "weather":
        return "weather_api"
    if intent == "cricket":
        return "cricket_api"
    return "general_llm"

def weather_api_node(state: ToolState) -> ToolState:
    try:
        loc = (state.get("location") or "").strip()
        if not loc:
            # fall back to normalized query, but this should rarely happen after fix
            loc = state.get("normalized_query", "")
        weather = fetch_current_weather(loc)
        return {"weather": weather, "error": ""}
    except Exception as e:
        return {"error": f"weather_api_error: {e}"}

def cricket_api_node(state: ToolState) -> ToolState:
    try:
        cq = (state.get("cricket_query") or "").strip()
        cricket = fetch_live_cricket(cq)
        return {"cricket": cricket, "error": ""}
    except Exception as e:
        return {"error": f"cricket_api_error: {e}"}

def general_llm_node(state: ToolState) -> ToolState:
    client = _azure_client()
    dep = _deployment()

    q = state.get("normalized_query", "")
    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": q},
        ],
        max_tokens=250,
    )
    return {"answer": resp.choices[0].message.content.strip()}

def synthesize_node(state: ToolState) -> ToolState:
    """
    Create a final answer. If a tool ran, summarize tool results.
    Always append to history and persist via checkpointing.
    """
    q = state.get("normalized_query", state.get("query", ""))
    weather = state.get("weather")
    cricket = state.get("cricket")
    err = state.get("error", "")

    # If general answer exists and no tools ran, use it
    if state.get("answer") and not weather and not cricket:
        ans = state["answer"]
    else:
        # Synthesize using LLM for nicer output (still not an agent)
        client = _azure_client()
        dep = _deployment()
        resp = client.chat.completions.create(
            model=dep,
            messages=[
                {"role": "system", "content": "Format tool results into a short helpful answer. Do not hallucinate."},
                {"role": "user", "content": f"Query: {q}\nWeather: {weather}\nCricket: {cricket}\nError: {err}\n"},
            ],
            max_tokens=220,
        )
        ans = resp.choices[0].message.content.strip()

    history = state.get("history", [])
    history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    return {"answer": ans, "history": history}


# ---------
# Build graph + SQLite checkpointing
# ---------
def build_graph():
    g = StateGraph(ToolState)

    g.add_node("init", init_node)
    g.add_node("normalize", normalize_node)
    g.add_node("plan", plan_node)

    g.add_node("weather_api", weather_api_node)
    g.add_node("cricket_api", cricket_api_node)
    g.add_node("general_llm", general_llm_node)

    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "init")
    g.add_edge("init", "normalize")
    g.add_edge("normalize", "plan")

    g.add_conditional_edges("plan", route_intent)

    g.add_edge("weather_api", "synthesize")
    g.add_edge("cricket_api", "synthesize")
    g.add_edge("general_llm", "synthesize")
    g.add_edge("synthesize", END)

    # SQLite checkpoint DB path (ensure directory exists)
    db_path = os.environ.get("CHECKPOINT_DB", "./storage/checkpoints.sqlite")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return g.compile(checkpointer=checkpointer)

GRAPH = build_graph()