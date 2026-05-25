from fastapi import FastAPI
from pydantic import BaseModel
from concept_resolver import resolve
from runtime_engine import run_suggest_flow, run_finalize_flow

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class FinalizeRequest(BaseModel):
    query: str
    accepted_nodes: list[str]

def simple_split(query: str):
    q = query.lower()
    terms = []
    if "diabetes" in q:
        terms.append("diabetes")
    if "asthma" in q:
        terms.append("asthma")
    return terms

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/query")
def query(req: QueryRequest):
    terms = simple_split(req.query)
    anchors = [resolve(t) for t in terms]
    anchors = [a for a in anchors if a]
    return run_suggest_flow(anchors)

@app.post("/finalize")
def finalize(req: FinalizeRequest):
    terms = simple_split(req.query)
    anchors = [resolve(t) for t in terms]
    anchors = [a for a in anchors if a]
    return run_finalize_flow(anchors, req.accepted_nodes)
