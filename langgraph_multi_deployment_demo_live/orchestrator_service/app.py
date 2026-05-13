from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4
from typing import Any, Dict

from graph import GRAPH

app = FastAPI(title="Orchestrator Service (LangGraph Agent)", version="1.0")


class RunRequest(BaseModel):
    thread_id: str
    query: str
    user_role: str = "user"
    hints: Dict[str, Any] = {}


def cfg(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/threads/new")
def new_thread():
    return {"thread_id": str(uuid4())}


@app.post("/run")
def run(req: RunRequest):
    result = GRAPH.invoke(
        {"query": req.query, "user_role": req.user_role, "hints": req.hints},
        config=cfg(req.thread_id),
    )
    return {
        "thread_id": req.thread_id,
        "answer": result.get("answer", ""),
        "meta": {
            "tool_results_keys": list((result.get("tool_results") or {}).keys()),
            "step": result.get("step", 0),
            "error": result.get("error", ""),
        },
    }