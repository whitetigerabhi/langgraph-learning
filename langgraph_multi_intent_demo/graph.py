import os
import re
import json
import sqlite3
import operator
from typing import TypedDict, List, Dict, Any, Annotated, Literal, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import AzureOpenAI

from tools.weather_api import fetch_current_weather
from tools.cricket_api import fetch_live_cricket

UK_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)

# --- Reducer for dict merge (results from parallel branches) ---
def merge_dict(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> Dict[str, Any]:
    a = a or {}
    b = b or {}
    out = dict(a)
    out.update(b)
    return out

class MultiState(TypedDict, total=False):
    # input
    query: str
    normalized_query: str

    # planner output
    tasks: Annotated[List[str], operator.add]          # e.g., ["weather_api","cricket_api"]
    location: str
    cricket_query: str
    general_query: str

    # branch completion + outputs
    done: Annotated[List[str], operator.add]           # e.g., ["weather_api"]
    results: Annotated[Dict[str, Any], merge_dict]     # {"weather": {...}, "cricket": {...}}

    # final
    answer: str
    error: str

    # conversation memory persisted via checkpointing
    history: Annotated[List[Dict[str, str]], operator.add]


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())

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

def init_node(state: MultiState) -> MultiState:
    return {
        "tasks": [],
        "done": [],
        "results": {},
        "history": [],
        "error": "",
        "answer": "",
        "location": "",
        "cricket_query": "",
        "general_query": ""
    }

def normalize_node(state: MultiState) -> MultiState:
    return {"normalized_query": _normalize(state.get("query", ""))}

def plan_node(state: MultiState) -> MultiState:
    """
    Multi-intent planner (NOT an agent):
    Returns a list of tasks to run and extracted parameters.
    """
    q = state.get("normalized_query", "")
    qlow = q.lower()

    # Deterministic detection (cheap & reliable)
    tasks: List[str] = []
    location = ""
    cricket_query = ""
    general_query = ""

    pc = UK_POSTCODE_RE.search(q)
    if "weather" in qlow or pc:
        tasks.append("weather_api")
        if pc:
            location = pc.group(0).upper().strip()

    if "cricket" in qlow or "score" in qlow or "csk" in qlow or "ipl" in qlow:
        tasks.append("cricket_api")
        cricket_query = "CSK" if "csk" in qlow else q

    if "explain" in qlow or "what is" in qlow:
        tasks.append("general_llm")
        general_query = q

    # If deterministic found something, return it
    if tasks:
        return {"tasks": tasks, "location": location, "cricket_query": cricket_query, "general_query": general_query}

    # LLM planner fallback
    client = _client()
    dep = _deployment()

    prompt = f"""
Determine which tasks are needed:
- weather_api (needs location/postcode)
- cricket_api (needs team keyword)
- general_llm (general explanation)

Return ONLY JSON:
{{
  "tasks": ["weather_api","cricket_api","general_llm"],
  "location": "",
  "cricket_query": "",
  "general_query": ""
}}

User query: {q}
"""
    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": "Output only JSON. No prose."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=220,
    )
    data = _extract_json_obj(resp.choices[0].message.content.strip())

    allowed = {"weather_api", "cricket_api", "general_llm"}
    tasks = [t for t in (data.get("tasks") or []) if t in allowed]

    # Final safe defaults
    if not tasks:
        tasks = ["general_llm"]

    return {
        "tasks": tasks,
        "location": (data.get("location") or "").strip(),
        "cricket_query": (data.get("cricket_query") or "").strip(),
        "general_query": (data.get("general_query") or "").strip(),
    }

def dispatch_node(state: MultiState):
    """
    Fan-out router: return list of node names to execute.
    Conditional edges can route dynamically based on state. [2](https://dasroot.net/posts/2026/03/remote-development-dev-containers-github-codespaces/)
    """
    return state.get("tasks", [])

def weather_api_node(state: MultiState) -> MultiState:
    loc = (state.get("location") or "").strip()
    if not loc:
        # fall back to query if planner didn't extract
        loc = state.get("normalized_query", "")
    try:
        w = fetch_current_weather(loc)
        return {"results": {"weather": w}, "done": ["weather_api"]}
    except Exception as e:
        return {"results": {"weather_error": str(e)}, "done": ["weather_api"], "error": f"weather_api_error: {e}"}

def cricket_api_node(state: MultiState) -> MultiState:
    cq = (state.get("cricket_query") or "").strip()
    try:
        c = fetch_live_cricket(cq)
        return {"results": {"cricket": c}, "done": ["cricket_api"]}
    except Exception as e:
        return {"results": {"cricket_error": str(e)}, "done": ["cricket_api"], "error": f"cricket_api_error: {e}"}

def general_llm_node(state: MultiState) -> MultiState:
    q = (state.get("general_query") or state.get("normalized_query") or "").strip()
    client = _client()
    dep = _deployment()
    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": "Answer clearly and concisely."},
            {"role": "user", "content": q},
        ],
        max_tokens=250,
    )
    ans = resp.choices[0].message.content.strip()
    return {"results": {"general": ans}, "done": ["general_llm"]}

def join_node(state: MultiState) -> MultiState:
    """
    Fan-in point. After branches, state should have merged results/done.
    LangGraph merges node updates between steps; nodes are functions. [1](https://docs.azure.cn/en-us/container-apps/quickstart-repo-to-cloud)
    """
    return {}

def route_after_join(state: MultiState) -> Literal["synthesize", "end_now"]:
    tasks = state.get("tasks", [])
    done = state.get("done", [])
    if tasks and set(done) >= set(tasks):
        return "synthesize"
    return "end_now"

def synthesize_node(state: MultiState) -> MultiState:
    q = state.get("normalized_query", "")
    results = state.get("results", {})
    err = state.get("error", "")

    client = _client()
    dep = _deployment()
    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": "Combine tool results into one short answer. Do not hallucinate."},
            {"role": "user", "content": f"User query: {q}\nResults: {results}\nError: {err}\nReturn a single combined answer."},
        ],
        max_tokens=280,
    )
    ans = resp.choices[0].message.content.strip()

    history = state.get("history", [])
    history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    return {"answer": ans, "history": history}

def build_graph():
    g = StateGraph(MultiState)

    g.add_node("init", init_node)
    g.add_node("normalize", normalize_node)
    g.add_node("plan", plan_node)

    g.add_node("weather_api", weather_api_node)
    g.add_node("cricket_api", cricket_api_node)
    g.add_node("general_llm", general_llm_node)

    g.add_node("join", join_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "init")
    g.add_edge("init", "normalize")
    g.add_edge("normalize", "plan")

    # fan-out
    g.add_conditional_edges("plan", dispatch_node)

    # fan-in
    g.add_edge("weather_api", "join")
    g.add_edge("cricket_api", "join")
    g.add_edge("general_llm", "join")

    g.add_conditional_edges("join", route_after_join, {"synthesize": "synthesize", "end_now": END})
    g.add_edge("synthesize", END)

    # SQLite checkpointing: persists state per thread_id [3](https://pypi.org/project/postcodes-io-api/)[4](https://microsoftlearning.github.io/mslearn-azure-ai/instructions/container-hosting/01-acr-tasks.html)
    db_path = os.environ.get("CHECKPOINT_DB", "./storage/checkpoints.sqlite")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    # check_same_thread=False is safe for SqliteSaver usage pattern [4](https://microsoftlearning.github.io/mslearn-azure-ai/instructions/container-hosting/01-acr-tasks.html)[5](https://postcodes.io/docs/api/api-reference-postcodes-io/)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # compile step required; attaches checkpointer for persistence [1](https://docs.azure.cn/en-us/container-apps/quickstart-repo-to-cloud)[3](https://pypi.org/project/postcodes-io-api/)
    return g.compile(checkpointer=checkpointer)

GRAPH = build_graph()
