from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4

from graph import GRAPH


app = FastAPI(title="Cricket Copilot Orchestrator", version="1.0")


class RunRequest(BaseModel):
    thread_id: str | None = None
    message: str
    user_role: str = "user"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/threads/new")
def new_thread():
    return {"thread_id": str(uuid4())}


@app.post("/run")
def run(req: RunRequest):
    thread_id = req.thread_id or str(uuid4())

    initial_state = {
        "thread_id": thread_id,
        "user_role": req.user_role,
        "message": req.message,
    }

    result = GRAPH.invoke(initial_state)

    return {
        "thread_id": thread_id,
        "answer": result.get("answer", ""),
        "meta": {
            "route": result.get("route"),
            "query_id": result.get("query_id"),
            "query_params": result.get("query_params"),
            "error": result.get("error"),
            "trace_id": result.get("action_result", {}).get("trace_id"),
        },
    }