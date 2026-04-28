from fastapi import FastAPI
from pydantic import BaseModel
from graph import GRAPH

app = FastAPI(title="LangGraph Tools Demo", version="0.1")

class RunRequest(BaseModel):
    thread_id: str
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
def run(req: RunRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = GRAPH.invoke({"query": req.query}, config=config)
    return {"thread_id": req.thread_id, "query": req.query, "answer": result.get("answer",""), "state": result}

@app.get("/state/{thread_id}")
def get_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = GRAPH.get_state(config)
    return {"thread_id": thread_id, "values": snapshot.values, "next": list(snapshot.next)}
