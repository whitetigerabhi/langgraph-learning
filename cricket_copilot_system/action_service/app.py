import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from db.postgres import fetch_all
from query_catalog.catalog import validate_params
from query_catalog.templates import TEMPLATES
from memory.store import load_memory, save_message, upsert_summary

app = FastAPI(title="Cricket Copilot Action API", version="1.0")


class Context(BaseModel):
    thread_id: str
    user_role: str = "user"
    user_id: Optional[str] = None


class StatsQueryRequest(BaseModel):
    query_id: str
    params: Dict[str, Any] = {}
    context: Context


class StatsQueryResponse(BaseModel):
    status: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    trace_id: str
    latency_ms: int


class MemoryLoadRequest(BaseModel):
    thread_id: str
    limit: int = 20


class MemorySaveRequest(BaseModel):
    thread_id: str
    role: str
    content: str
    metadata: Dict[str, Any] = {}
    # optional summary update
    summary: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/stats/query", response_model=StatsQueryResponse)
def stats_query(req: StatsQueryRequest):
    # 1) Allowlist + params validation
    try:
        cleaned = validate_params(req.query_id, req.params)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid params: {str(e)}")

    # 2) Retrieve SQL template
    if req.query_id not in TEMPLATES:
        raise HTTPException(status_code=500, detail=f"Template missing for query_id={req.query_id}")

    sql = TEMPLATES[req.query_id]

    # 3) Execute safely (parameterized)
    trace_id = str(uuid.uuid4())
    t0 = time.time()
    cols, rows = fetch_all(sql, cleaned)
    latency_ms = int((time.time() - t0) * 1000)

    # Normalize rows to JSON-friendly lists
    rows_out = [list(r) for r in rows]

    return StatsQueryResponse(
        status="ok",
        columns=cols,
        rows=rows_out,
        row_count=len(rows_out),
        trace_id=trace_id,
        latency_ms=latency_ms,
    )


@app.post("/memory/load")
def memory_load(req: MemoryLoadRequest):
    return load_memory(req.thread_id, limit=req.limit)


@app.post("/memory/save")
def memory_save(req: MemorySaveRequest):
    # Save message
    save_message(req.thread_id, req.role, req.content, metadata=req.metadata)

    # Optionally update rolling summary
    if req.summary is not None:
        upsert_summary(req.thread_id, req.summary)

    return {"status": "ok"}