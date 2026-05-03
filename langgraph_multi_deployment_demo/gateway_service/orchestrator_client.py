import os
import requests

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8040")


def run_orchestrator(thread_id: str, query: str, user_role: str, hints: dict) -> dict:
    r = requests.post(
        f"{ORCHESTRATOR_URL}/run",
        json={"thread_id": thread_id, "query": query, "user_role": user_role, "hints": hints},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()