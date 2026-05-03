cd ~/langgraph-learning/langgraph_multi_deployment_demo/orchestrator_service
cat > graph.py <<'PY'
import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import AgentState
from routing import route_after_decide

from nodes.init_state import init_node
from nodes.input_guardrail import input_guardrail_node
from nodes.agent_decide import agent_decide_node
from nodes.execute_tools import execute_tools_node
from nodes.finalize import finalize_node
from nodes.output_guardrail import output_guardrail_node


def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("init", init_node)
    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("agent_decide", agent_decide_node)
    g.add_node("execute_tools", execute_tools_node)
    g.add_node("finalize", finalize_node)
    g.add_node("output_guardrail", output_guardrail_node)

    # Edges
    g.add_edge(START, "init")
    g.add_edge("init", "input_guardrail")
    g.add_edge("input_guardrail", "agent_decide")

    # Conditional routing: if model produced tool_calls → execute_tools else → finalize
    # LangGraph supports dynamic routing via conditional edges. [1](https://oneuptime.com/blog/post/2026-02-16-how-to-use-azure-openai-function-calling-to-build-tool-using-ai-agents/view)
    g.add_conditional_edges(
        "agent_decide",
        route_after_decide,
        {
            "execute_tools": "execute_tools",
            "finalize": "finalize",
        },
    )

    g.add_edge("execute_tools", "finalize")
    g.add_edge("finalize", "output_guardrail")
    g.add_edge("output_guardrail", END)

    # SQLite checkpointer (memory persistence)
    db_path = os.environ.get("CHECKPOINT_DB", "./storage/checkpoints.sqlite")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return g.compile(checkpointer=checkpointer)


GRAPH = build_graph()
PY
