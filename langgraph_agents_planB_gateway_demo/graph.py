import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import AgentState

from nodes.init_state import init_node
from nodes.normalize import normalize_node
from nodes.guardrail import guardrail_node, route_after_guardrail
from nodes.plan_once import plan_once_node
from nodes.execute_step import execute_step_node
from nodes.routing import route_loop


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("init", init_node)
    g.add_node("normalize", normalize_node)
    g.add_node("guardrail", guardrail_node)
    g.add_node("plan_once", plan_once_node)
    g.add_node("execute_step", execute_step_node)

    g.add_edge(START, "init")
    g.add_edge("init", "normalize")
    g.add_edge("normalize", "guardrail")

    g.add_conditional_edges("guardrail", route_after_guardrail, {
        "plan_once": "plan_once",
        "end": END,
    })

    g.add_edge("plan_once", "execute_step")

    g.add_conditional_edges("execute_step", route_loop, {
        "execute_step": "execute_step",
        "end": END,
    })

    # SQLite checkpointer (lightweight sync persistence). 
    db_path = os.environ.get("CHECKPOINT_DB", "./storage/checkpoints.sqlite")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return g.compile(checkpointer=checkpointer)


GRAPH = build_graph()