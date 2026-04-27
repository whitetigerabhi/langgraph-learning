import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from openai import AzureOpenAI

class GraphState(TypedDict, total=False):
    query: str
    normalized_query: str
    answer: str

def normalize_node(state: GraphState) -> GraphState:
    q = (state.get("query") or "").strip()
    return {"normalized_query": " ".join(q.split())}

def route_node(state: GraphState) -> Literal["clarify", "llm_call"]:
    q = state.get("normalized_query") or ""
    return "clarify" if len(q) < 4 else "llm_call"

def clarify_node(state: GraphState) -> GraphState:
    return {"answer": "Please provide a bit more detail (at least a few words)."}

def llm_call_node(state: GraphState) -> GraphState:
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")

    q = state.get("normalized_query") or state.get("query") or ""
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": q}
        ],
        max_tokens=250,
    )
    return {"answer": resp.choices[0].message.content.strip()}

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("normalize", normalize_node)
    g.add_node("clarify", clarify_node)
    g.add_node("llm_call", llm_call_node)

    g.add_edge(START, "normalize")
    g.add_conditional_edges("normalize", route_node)
    g.add_edge("clarify", END)
    g.add_edge("llm_call", END)

    return g.compile()

GRAPH = build_graph()
