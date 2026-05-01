import os
import re
import json
import sqlite3
import operator
from typing import TypedDict, List, Dict, Any, Annotated, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import AzureOpenAI

from tools.weather_api import fetch_current_weather
from tools.cricket_api import fetch_live_cricket

UK_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)

def merge_dict(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> Dict[str, Any]:
    a = a or {}
    b = b or {}
    out = dict(a)
    out.update(b)
    return out

class State(TypedDict, total=False):
    query: str
    normalized_query: str

    # multi-intent orchestration
    tasks: Annotated[List[str], operator.add]
    done: Annotated[List[str], operator.add]
    results: Annotated[Dict[str, Any], merge_dict]

    location: str
    cricket_query: str
    general_query: str

    # approval gate
    needs_approval: bool
    approved: bool
    approval_token: str
    proposed_answer: str

    # final output + memory
    answer: str
    history: Annotated[List[Dict[str, str]], operator.add]
    error: str

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

def init_node(state: State) -> State:
    return {
        "tasks": [],
        "done": [],
        "results": {},
        "history": [],
        "error": "",
        "needs_approval": False,
        "approved": False,
        "approval_token": "",
        "proposed_answer": "",
        "answer": "",
        "location": "",
        "cricket_query": "",
        "general_query": "",
    }

def normalize_node(state: State) -> State:
    return {"normalized_query": _normalize(state.get("query", ""))}

def plan_node(state: State) -> State:
    q = state.get("normalized_query", "")
    qlow = q.lower()

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

    if tasks:
        return {"tasks": tasks, "location": location, "cricket_query": cricket_query, "general_query": general_query}

    # LLM fallback planner (still non-agentic)
    client = _client()
    dep = _deployment()
    prompt = f"""
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
        messages=[{"role": "system", "content": "Output only JSON."}, {"role": "user", "content": prompt}],
        max_tokens=200,
    )
    data = _extract_json_obj(resp.choices[0].message.content.strip())
    allowed = {"weather_api", "cricket_api", "general_llm"}
    tasks = [t for t in (data.get("tasks") or []) if t in allowed] or ["general_llm"]

    return {
        "tasks": tasks,
        "location": (data.get("location") or "").strip(),
        "cricket_query": (data.get("cricket_query") or "").strip(),
        "general_query": (data.get("general_query") or "").strip(),
    }

def dispatch_node(state: State):
    return state.get("tasks", [])

def weather_api_node(state: State) -> State:
    loc = (state.get("location") or "").strip() or state.get("normalized_query", "")
    try:
        w = fetch_current_weather(loc)
        return {"results": {"weather": w}, "done": ["weather_api"]}
    except Exception as e:
        return {"results": {"weather_error": str(e)}, "done": ["weather_api"], "error": f"weather_api_error: {e}"}

def cricket_api_node(state: State) -> State:
    cq = (state.get("cricket_query") or "").strip()
    try:
        c = fetch_live_cricket(cq)
        return {"results": {"cricket": c}, "done": ["cricket_api"]}
    except Exception as e:
        return {"results": {"cricket_error": str(e)}, "done": ["cricket_api"], "error": f"cricket_api_error: {e}"}

def general_llm_node(state: State) -> State:
    q = (state.get("general_query") or state.get("normalized_query") or "").strip()
    client = _client()
    dep = _deployment()
    resp = client.chat.completions.create(
        model=dep,
        messages=[{"role": "system", "content": "Answer clearly and concisely."}, {"role": "user", "content": q}],
        max_tokens=250,
    )
    ans = resp.choices[0].message.content.strip()
    return {"results": {"general": ans}, "done": ["general_llm"]}

def join_node(state: State) -> State:
    return {}

def route_after_join(state: State) -> Literal["approval_gate", "end_now"]:
    tasks = state.get("tasks", [])
    done = state.get("done", [])
    if tasks and set(done) >= set(tasks):
        return "approval_gate"
    return "end_now"

def approval_gate_node(state: State) -> State:
    """
    Simulate HITL approval requirement:
    If query contains 'send' or 'approve', require approval before finalizing.
    """
    q = state.get("normalized_query", "").lower()
    if "send" in q or "approve" in q:
        token = state.get("approval_token") or os.urandom(8).hex()
        return {"needs_approval": True, "approved": False, "approval_token": token}
    return {"needs_approval": False, "approved": True}

def route_after_approval(state: State) -> Literal["finalize", "stop_for_approval"]:
    if state.get("needs_approval") and not state.get("approved"):
        return "stop_for_approval"
    return "finalize"

def stop_for_approval_node(state: State) -> State:
    """
    Produce a proposed answer but do not finalize. Caller must approve.
    """
    q = state.get("normalized_query", "")
    results = state.get("results", {})
    err = state.get("error", "")

    client = _client()
    dep = _deployment()
    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": "Draft a response that requires human approval before sending."},
            {"role": "user", "content": f"Query: {q}\nResults: {results}\nError: {err}\nDraft response:"},
        ],
        max_tokens=220,
    )
    proposed = resp.choices[0].message.content.strip()
    return {"proposed_answer": proposed}

def finalize_node(state: State) -> State:
    q = state.get("normalized_query", "")
    results = state.get("results", {})
    err = state.get("error", "")

    # If approval required, use proposed_answer as final once approved
    if state.get("needs_approval"):
        ans = state.get("proposed_answer") or "Approved."
    else:
        client = _client()
        dep = _deployment()
        resp = client.chat.completions.create(
            model=dep,
            messages=[
                {"role": "system", "content": "Combine results into one short answer. Do not hallucinate."},
                {"role": "user", "content": f"Query: {q}\nResults: {results}\nError: {err}\nReturn final:"},
            ],
            max_tokens=280,
        )
        ans = resp.choices[0].message.content.strip()

    history = state.get("history", [])
    history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    return {"answer": ans, "history": history}

def build_graph():
    g = StateGraph(State)

    g.add_node("init", init_node)
    g.add_node("normalize", normalize_node)
    g.add_node("plan", plan_node)

    g.add_node("weather_api", weather_api_node)
    g.add_node("cricket_api", cricket_api_node)
    g.add_node("general_llm", general_llm_node)

    g.add_node("join", join_node)
    g.add_node("approval_gate", approval_gate_node)
    g.add_node("stop_for_approval", stop_for_approval_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "init")
    g.add_edge("init", "normalize")
    g.add_edge("normalize", "plan")

    g.add_conditional_edges("plan", dispatch_node)
    g.add_edge("weather_api", "join")
    g.add_edge("cricket_api", "join")
    g.add_edge("general_llm", "join")

    g.add_conditional_edges("join", route_after_join, {"approval_gate": "approval_gate", "end_now": END})
    g.add_conditional_edges("approval_gate", route_after_approval, {"stop_for_approval": "stop_for_approval", "finalize": "finalize"})
    g.add_edge("stop_for_approval", END)
    g.add_edge("finalize", END)

    db_path = os.environ.get("CHECKPOINT_DB", "./storage/checkpoints.sqlite")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)  # lightweight sync saver for demos [3](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-quickstart-task-cli)[7](https://microsoftlearning.github.io/mslearn-azure-ai/instructions/container-hosting/01-acr-tasks.html)

    return g.compile(checkpointer=checkpointer)  # enables threads+history [2](https://code.visualstudio.com/docs/remote/codespaces)[5](https://postcodes.io/docs/api/api-reference-postcodes-io/)

GRAPH = build_graph()