import os
import sqlite3
from typing import TypedDict, Literal, List, Dict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import AzureOpenAI


class GraphState(TypedDict, total=False):
    query: str
    normalized_query: str
    answer: str
    history: List[Dict[str, str]]  # persisted chat history inside state


def init_node(state: GraphState) -> GraphState:
    # Ensure history exists
    return {"history": state.get("history", [])}


def normalize_node(state: GraphState) -> GraphState:
    q = (state.get("query") or "").strip()
    return {"normalized_query": " ".join(q.split())}


def route_node(state: GraphState) -> Literal["clarify", "llm_call"]:
    q = state.get("normalized_query") or ""
    return "clarify" if len(q) < 4 else "llm_call"


def clarify_node(state: GraphState) -> GraphState:
    msg = "Please provide a bit more detail (at least a few words)."
    q = state.get("normalized_query") or state.get("query") or ""
    history = state.get("history", [])
    history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": msg}]
    return {"answer": msg, "history": history}


def llm_call_node(state: GraphState) -> GraphState:
    # Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")

    q = state.get("normalized_query") or state.get("query") or ""
    history = state.get("history", [])

    messages = [{"role": "system", "content": "You are a helpful assistant. Answer concisely."}]
    messages.extend(history)
    messages.append({"role": "user", "content": q})

    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=250,
    )
    ans = resp.choices[0].message.content.strip()

    history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    return {"answer": ans, "history": history}


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("init", init_node)
    g.add_node("normalize", normalize_node)
    g.add_node("clarify", clarify_node)
    g.add_node("llm_call", llm_call_node)

    g.add_edge(START, "init")
    g.add_edge("init", "normalize")
    g.add_conditional_edges("normalize", route_node)
    g.add_edge("clarify", END)
    g.add_edge("llm_call", END)

    # SQLite checkpointer (file-based)
    db_path = os.environ.get("CHECKPOINT_DB", "./storage/checkpoints.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return g.compile(checkpointer=checkpointer)


GRAPH = build_graph()
