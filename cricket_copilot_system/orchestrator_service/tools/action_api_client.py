import os
import httpx

ACTION_API_BASE = os.environ.get("ACTION_API_BASE", "http://localhost:8050")


def memory_load(thread_id: str, limit: int = 20) -> dict:
    r = httpx.post(
        f"{ACTION_API_BASE}/memory/load",
        json={"thread_id": thread_id, "limit": limit},
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


def memory_save(thread_id: str, role: str, content: str, metadata: dict | None = None, summary: str | None = None) -> dict:
    payload = {
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
    if summary is not None:
        payload["summary"] = summary

    r = httpx.post(
        f"{ACTION_API_BASE}/memory/save",
        json=payload,
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


def stats_query(query_id: str, params: dict, thread_id: str, user_role: str = "user") -> dict:
    payload = {
        "query_id": query_id,
        "params": params,
        "context": {
            "thread_id": thread_id,
            "user_role": user_role,
        },
    }

    r = httpx.post(
        f"{ACTION_API_BASE}/stats/query",
        json=payload,
        timeout=20.0,
    )
    r.raise_for_status()
    return r.json()