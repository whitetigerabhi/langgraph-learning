from fastapi import FastAPI
from pydantic import BaseModel
from graph import GRAPH

app = FastAPI(title="LangGraph API Demo", version="0.1")

class RunRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
def run(req: RunRequest):
    result = GRAPH.invoke({"query": req.query})
    return {"query": req.query, "answer": result.get("answer", ""), "state": result}
