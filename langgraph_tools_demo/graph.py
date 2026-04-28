import os
import re
import json
import sqlite3
from typing import TypedDict, Literal, List, Dict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import AzureOpenAI

from tools.weather_api import fetch_current_weather
from tools.cricket_api import fetch_live_cricket


class ToolState(TypedDict, total=False):
    query: str
    normalized_query: str

    # routing + extracted params
    intent: Literal["weather", "cricket", "general"]
    location: str
    cricket_query: str

    # outputs
    answer: str
    weather: dict
    cricket: dict
    error: str

    # conversation memory
    history: List[Dict[str, str]]


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def init_node(state: ToolState) -> ToolState:
    return {"history": state.get("history", []), "error": ""}


def normalize_node(state: ToolState) -> ToolState:
    return {"normalized_query": _normalize(state.get("query", ""))}


def plan_node(state: ToolState) -> ToolState:
    """
    Not an agent: just a planner/classifier.
    Tries to classify intent + extract parameters.
    Falls back deterministically if JSON is messy.
    """
    q = state.get("normalized_query", "")
    qlow = q.lower()

    # Deterministic fast-path
    if "weather" in qlow or re.search(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", q, re.IGNORECASE):
        return {"intent": "weather", "location": q}
    if "cricket" in qlow or "score" in qlow or "csk" in qlow:
        return {"intent": "cricket", "cricket_query": q}

    # Otherwise use LLM classifier
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")

    prompt = f"""
Classify the query into one intent: weather | cricket | general.
Return ONLY JSON:
{{"intent":"weather|cricket|general","location":"", "cricket_query":""}}

Query: {q}
"""
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": "Output only JSON."}, {"role": "user", "content": prompt}],
        max_tokens=120,
    )
    raw = resp.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {"intent": "general"}

    return {
        "intent": data.get("intent", "general"),
        "location": (data.get("location") or "").strip(),
        "cricket_query": (data.get("cricket_query") or "").strip(),
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
        loc = state.get("location") or state.get("normalized_query") or ""
        weather = fetch_current_weather(loc)
        return {"weather": weather}
    except Exception as e:
        return {"error": f"weather_api_error: {e}"}


def cricket_api_node(state: ToolState) -> ToolState:
    try:
        cq = state.get("cricket_query") or ""
        cricket = fetch_live_cricket(cq)
        return {"cricket": cricket}
    except Exception as e:
        return {"error": f"cricket_api_error: {e}"}


def general_llm_node(state: ToolState) -> ToolState:
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")

    q = state.get("normalized_query", "")
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": "Answer concisely."}, {"role": "user", "content": q}],
        max_tokens=250,
    )
    return {"answer": resp.choices[0].message.content.strip()}


def synthesize_node(state: ToolState) -> ToolState:
    """
    Convert structured tool results into a clean final answer.
    (LLM formatting, still not an agent.)
    """
    q = state.get("normalized_query", state.get("query", ""))
    weather = state.get("weather")
    cricket = state.get("cricket")
    err = state.get("error", "")

    # If general answer already exists and no tools ran
    if state.get("answer") and not weather and not cricket:
        ans = state["answer"]
    else:
        # Format using LLM to avoid ugly JSON output
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="2024-10-21",
        )
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Format tool results into a short helpful answer. Do not hallucinate."},
                {"role": "user", "content": f"Query: {q}\nWeather: {weather}\nCricket: {cricket}\nError: {err}\n"}
            ],
            max_tokens=220,
        )
        ans = resp.choices[0].message.content.strip()

    history = state.get("history", [])
    history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    return {"answer": ans, "history": history}


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

    # SQLite checkpoint DB (create directory if missing)
    db_path = os.environ.get("CHECKPOINT_DB", "./storage/checkpoints.sqlite")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return g.compile(checkpointer=checkpointer)


GRAPH = build_graph()