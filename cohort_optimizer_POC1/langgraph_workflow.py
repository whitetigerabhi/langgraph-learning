from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from concept_resolver import resolve
from runtime_engine import run_suggest_flow

class KGState(TypedDict, total=False):
    user_query: str
    terms: List[str]
    anchor_concepts: List[str]
    output: Dict[str, Any]

def parse_query(state: KGState) -> KGState:
    q = state["user_query"].lower()
    terms = []
    if "diabetes" in q:
        terms.append("diabetes")
    if "asthma" in q:
        terms.append("asthma")
    return {**state, "terms": terms}

def resolve_concepts(state: KGState) -> KGState:
    anchors = [resolve(t) for t in state.get("terms", [])]
    anchors = [a for a in anchors if a]
    return {**state, "anchor_concepts": anchors}

def compute(state: KGState) -> KGState:
    out = run_suggest_flow(state.get("anchor_concepts", []))
    return {**state, "output": out}

def build_workflow():
    g = StateGraph(KGState)
    g.add_node("parse_query", parse_query)
    g.add_node("resolve_concepts", resolve_concepts)
    g.add_node("compute", compute)

    g.set_entry_point("parse_query")
    g.add_edge("parse_query", "resolve_concepts")
    g.add_edge("resolve_concepts", "compute")
    g.add_edge("compute", END)
    return g.compile()

if __name__ == "__main__":
    wf = build_workflow()
    res = wf.invoke({"user_query": "Show members with diabetes and asthma"})
    print(res["output"])